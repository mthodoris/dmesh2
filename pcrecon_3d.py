import torch as th
import numpy as np
import os
import time
import argparse
import yaml
import sys
import trimesh
from tqdm import tqdm

from exp.utils.utils import *
from exp.utils.dmesh import *
from exp.utils.logging import get_logger
from exp.utils.common import *
from exp.utils.mlflow_utils import init_mlflow_run, MetricWriter

from input.common import DOMAIN

from mindiffdt.qface import qface_knn_spatial, qface_dt
from mindiffdt.projection import knn_search, projection
from mindiffdt.minball import MB3_V0
from mindiffdt.tgrid import TetGrid

from torch.utils.tensorboard import SummaryWriter

from easydict import EasyDict as edict

from torch_scatter import scatter

DEVICE = 'cuda:0'

MAX_KNN_K = 40              # maximum k for knn in computing expected chamfer distance
NEAR_THRESH = 1e-3          # distance threshold to be used in preal initialization

### saving settings
RENDERING = False           # whether to dump intermediate meshes during optimization
MESH_FORMAT = 'obj'

class PCRecon3D:
    '''
    3D counterpart of PCRecon2D (pcrecon_2d.py): reconstructs a triangle mesh
    from a 3D point cloud using the same two-stage differentiable pipeline
    (point-wise real value initialization, then point position refinement),
    with faces (3-point simplexes) taking the role that edges (2-point
    simplexes) play in 2D.

    Unlike pcrecon_2d.py, this does not include the (optional, off-by-default
    in the 2D config) RL-ball point-weight pruning stage: it relies on an
    exact point-to-segment distance for its reward, and porting that reward
    to point-to-triangle distance while keeping the batched reward computation
    correct would need a new geometric primitive not present elsewhere in the
    codebase. The two stages implemented here already perform the full
    point-cloud-to-mesh extraction.
    '''

    def __init__(self,

                logdir,
                logger,

                # target points;
                target_point_positions,

                # sample interval for our mesh;
                our_sample_interval,

                # init method;
                init_args,

                # lr;
                lr_settings,

                # init preal;
                init_preal_settings,

                # optimize ppos;
                optimize_ppos_settings,

                use_mlflow: bool = False,):

        self.logger = logger

        self.target_point_positions = target_point_positions
        self.our_sample_interval = our_sample_interval
        self.init_args = init_args

        '''
        Grid
        '''
        self.tgrid = TetGrid(DEVICE)
        self.ppos: th.Tensor = None
        self.preal: th.Tensor = None
        self.dtfaces: th.Tensor = None

        '''
        Logdir
        '''
        self.logdir = logdir
        self.writer = MetricWriter(SummaryWriter(logdir), use_mlflow)

        '''
        LR
        '''
        self.lr = lr_settings
        self.ppos_lr = float(self.lr.pos)
        self.preal_lr = float(self.lr.real)

        '''
        Init point reals
        '''
        self.init_preal_settings = init_preal_settings
        self.init_preal_settings.num_steps = int(float(self.init_preal_settings.num_steps))
        self.init_preal_settings.vis_steps = int(float(self.init_preal_settings.vis_steps))
        self.init_preal_settings.real_reg_weight = float(self.init_preal_settings.real_reg_weight)

        '''
        Optimize point positions
        '''
        self.optimize_ppos_settings = optimize_ppos_settings
        self.optimize_ppos_settings.num_steps = int(float(self.optimize_ppos_settings.num_steps))
        self.optimize_ppos_settings.vis_steps = int(float(self.optimize_ppos_settings.vis_steps))

        '''
        Etc
        '''
        self.global_optim_start_time = 0.0

    '''
    Initialization and refinement
    '''
    def init_grid(self):
        grid_size = self.init_args.get("grid_size", 5e-2)
        self.tgrid.init((-DOMAIN, -DOMAIN, -DOMAIN), (DOMAIN, DOMAIN, DOMAIN), grid_size)

        self.ppos = self.tgrid.verts.clone()
        self.preal = th.zeros((self.ppos.shape[0],), dtype=th.float32, device=DEVICE)


    '''
    Saving
    '''
    @th.no_grad()
    def save_mesh(self, ppos, faces, path):

        os.makedirs(path, exist_ok=True)

        mesh = trimesh.Trimesh(
            vertices=ppos.cpu().numpy(),
            faces=faces.cpu().numpy(),
            process=False,
        )
        mesh.export(os.path.join(path, f"mesh.{MESH_FORMAT}"))

        ### save timestamp
        with open(os.path.join(path, "time_sec.txt"), "w") as f:
            f.write(str(time.time() - self.global_optim_start_time))

        ### save points and faces
        np.save(os.path.join(path, "points.npy"), ppos.cpu().numpy())
        np.save(os.path.join(path, "faces.npy"), faces.cpu().numpy())

        ### save num points and faces
        with open(os.path.join(path, "mesh_info.txt"), "w") as f:
            f.write(f"num_points: {ppos.shape[0]}\n")
            f.write(f"num_faces: {faces.shape[0]}\n")

    '''
    Losses
    '''
    def compute_topology_regularizer(self, preal: th.Tensor):
        '''
        Remove redundant faces by penalizing the existence of faces.
        '''
        reg = th.mean(preal)
        return reg

    def compute_geometry_regularizer(self, preal: th.Tensor, coef: float):
        '''
        Remove redundant faces by penalizing the existence of faces.
        '''
        reg = th.mean(preal) * coef
        return reg

    def compute_eval_loss(self, positions: th.Tensor, faces: th.Tensor):
        '''
        Compute evaluation loss.
        '''
        raise NotImplementedError()

    '''
    Updates during optimization
    '''

    def update_lr(self, lr: float, lr_schedule: str, step: int, num_steps: int, optimizer: th.optim.Optimizer):

        if lr_schedule == "linear":
            lr = lr * (1.0 - (step / num_steps))
        elif lr_schedule == "exp":
            min_log_lr = np.log(MIN_LR)
            max_log_lr = np.log(lr)
            curr_log_lr = max_log_lr + (min_log_lr - max_log_lr) * (step / num_steps)
            lr = np.exp(curr_log_lr)
        elif lr_schedule == "constant":
            lr = lr
        else:
            raise ValueError(f"Invalid lr schedule: {lr_schedule}")

        lr = max(lr, MIN_LR)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    '''
    Sampling
    '''
    def sample_points_from_faces(self, point_positions: th.Tensor, face_idx: th.Tensor):
        '''
        Area-weighted random barycentric sampling. Triangles have no closed-form
        even spacing the way edges do (arc-length parametrization in 2D), so each
        face gets a sample count proportional to its area, with a random
        barycentric coordinate per sample; every face gets at least one sample.
        '''
        v0 = point_positions[face_idx[:, 0]]
        v1 = point_positions[face_idx[:, 1]]
        v2 = point_positions[face_idx[:, 2]]
        e1 = v1 - v0
        e2 = v2 - v0

        with th.no_grad():
            area = 0.5 * th.norm(th.cross(e1, e2, dim=-1), dim=-1)
            ref_area = (np.sqrt(3.0) / 4.0) * (self.our_sample_interval ** 2)
            num_samples = th.clamp((area / ref_area).round().long(), min=1)

            face_id = th.repeat_interleave(th.arange(face_idx.shape[0], device=DEVICE), num_samples)

            r1 = th.rand((face_id.shape[0],), device=DEVICE)
            r2 = th.rand((face_id.shape[0],), device=DEVICE)
            sqrt_r1 = th.sqrt(r1)
            bary_u = (1.0 - sqrt_r1).unsqueeze(-1)
            bary_v = (r2 * sqrt_r1).unsqueeze(-1)

        sample_pos = v0[face_id] + e1[face_id] * bary_u + e2[face_id] * bary_v

        return sample_pos, face_id

    '''
    Main Optimization
    '''
    def optimize(self):

        self.global_optim_start_time = time.time()

        self.init_grid()
        self.logger.info(f"Initialized grid with {self.ppos.shape[0]} points.")

        self.logger.info(f"Start preal initialization.")
        self.init_preal()

        self.logger.info(f"Start ppos optimization.")
        self.optimize_ppos()

        ### save final result
        save_dir = os.path.join(self.writer.log_dir, f"result")
        self.save_mesh(self.ppos, self.dtfaces, save_dir)

    '''
    Point-wise real initialization (fixed point-positions).
    '''
    def _refresh_preal_optimizer(self):

        preal = self.preal.clone()
        preal.requires_grad = True

        preal_lr = self.preal_lr
        optimizer = th.optim.Adam([preal], lr=preal_lr)

        return optimizer, preal

    def init_preal(self):

        '''
        Real settings
        '''
        target_sample_points = self.target_point_positions

        '''
        Refresh optimizer and variables
        '''
        optimizer, preal = self._refresh_preal_optimizer()
        ppos = self.ppos.clone()

        '''
        Gather fixed faces.
        '''
        face_idx = self.tgrid.tri_idx

        '''
        Gather point idx of which reals are fixed to 0.
        Those points that are located far from the target points are fixed to 0 real value.
        '''
        with th.no_grad():
            face_v0 = ppos[face_idx[:, 0]]
            face_v1 = ppos[face_idx[:, 1]]
            face_v2 = ppos[face_idx[:, 2]]

            face_ball, face_ball_stable = MB3_V0.forward(face_v0, face_v1, face_v2)
            face_ball_cen = face_ball.center
            face_ball_rad = face_ball.radius

            face_ball_cen_nn_dist = run_knn(face_ball_cen, self.target_point_positions, 1)[1]
            face_ball_cen_nn_dist = face_ball_cen_nn_dist.squeeze(-1)
            face_ball_nn_dist = face_ball_cen_nn_dist - face_ball_rad

            possible_faces = face_ball_stable & (face_ball_nn_dist <= (NEAR_THRESH * 2 * DOMAIN))
            possible_face_idx = face_idx[possible_faces]
            possible_face_verts = possible_face_idx.unique()

            # @bugfix: have to update [possible_face_idx] again
            face_comprised_of_possible_face_verts = th.all(th.isin(face_idx, possible_face_verts), dim=-1)
            possible_face_idx = face_idx[face_comprised_of_possible_face_verts]

            fixed_zero_idx = th.ones_like(ppos[:, 0], dtype=th.bool)
            fixed_zero_idx[possible_face_verts] = False

            preal[fixed_zero_idx] = 0.0
            preal[~fixed_zero_idx] = 1.0

        '''
        Sample points from faces that possibly exist.
        '''
        with th.no_grad():
            our_sample_points_pos, our_sample_points_face = self.sample_points_from_faces(ppos, possible_face_idx)

        '''
        Find K nearest points.
        '''
        with th.no_grad():
            num_knn = MAX_KNN_K
            num_knn = min(num_knn, len(our_sample_points_pos))
            tgt_to_our_knn_idx, tgt_to_our_knn_dist = run_knn(target_sample_points, our_sample_points_pos, num_knn)
            tgt_to_our_face_knn_idx = our_sample_points_face[tgt_to_our_knn_idx]

            num_knn = 1
            our_to_tgt_knn_idx, our_to_tgt_knn_dist = run_knn(our_sample_points_pos, target_sample_points, num_knn)
            our_to_tgt_knn_dist = our_to_tgt_knn_dist.squeeze(-1)

        '''
        Warmup steps: Exclude faces that are not close to the target points using direct differentiation.
        '''
        num_steps = self.init_preal_settings.num_steps
        vis_steps = self.init_preal_settings.vis_steps
        init_lr = self.preal_lr
        lr_schedule = self.init_preal_settings.lr_schedule
        real_reg_weight = self.init_preal_settings.real_reg_weight

        start_event = th.cuda.Event(enable_timing=True)
        end_event = th.cuda.Event(enable_timing=True)

        bar = tqdm(range(num_steps))
        for step in bar:

            curr_lr = self.update_lr(init_lr, lr_schedule, step, num_steps, optimizer)

            '''
            Evaluate probability based on preals
            '''
            possible_face_preal = preal[possible_face_idx.to(dtype=th.long)]
            possible_face_prob = dmin(possible_face_preal, k=DMIN_K)
            our_sample_points_prob = possible_face_prob[our_sample_points_face]

            '''
            2. Compute losses
            '''

            '''
            2-1. CD loss from GT to OURS
            '''

            start_event.record()
            if True:

                dist = tgt_to_our_knn_dist                                      # [# gt sample points, # k]
                prob_mat = possible_face_prob[tgt_to_our_face_knn_idx]          # [# tgt sample points, # k]

                '''
                Sorting: If a sample point from gt mesh finds a near point from a certain face,
                we do not consider another point from the same face in computing the loss.
                '''

                # =========

                sorted_indices = th.argsort(tgt_to_our_face_knn_idx, dim=1, stable=True)

                # Step 2: Rearrange A using sorted indices
                sorted_A = th.gather(prob_mat, 1, sorted_indices)

                # Step 3: Identify duplicates in sorted B
                sorted_B = th.gather(tgt_to_our_face_knn_idx, 1, sorted_indices)
                duplicate_mask = sorted_B[:, 1:] == sorted_B[:, :-1]
                # Pad the mask to match the shape of A and B
                padded_mask = th.cat([th.zeros(duplicate_mask.shape[0], 1, dtype=th.bool, device=DEVICE), duplicate_mask], dim=1)

                # Step 4: Revert A to the original order, applying the duplicate mask
                # First, set duplicates in sorted_A to 0
                sorted_A[padded_mask] = 0.0

                # Then, invert the sorted indices to get the original order
                inverse_indices = th.argsort(sorted_indices, dim=1)
                original_order_A = th.gather(sorted_A, 1, inverse_indices)

                prob_mat = original_order_A

                # =========

                # append one more column: fall back for all miss case;
                dist_n_col = th.ones((dist.shape[0], 1), dtype=th.float32, device=DEVICE) * (DOMAIN * 10)
                prob_n_col = th.ones((dist.shape[0], 1), dtype=th.float32, device=DEVICE)

                dist = th.cat([dist, dist_n_col], dim=-1)
                prob_mat = th.cat([prob_mat, prob_n_col], dim=-1)

                # =========

                n_prob_mat = 1.0 - prob_mat
                n_prob_mat_prod = th.cumprod(n_prob_mat, dim=-1)

                prob_mat[:, 1:] = prob_mat[:, 1:].clone() * n_prob_mat_prod[:, :-1]

                loss_0 = th.sum(prob_mat * dist, dim=-1)         # [# gt sample points,]
                loss_0 = loss_0.mean()

            '''
            2-2. CD loss from OURS to GT.
            '''
            if True:
                dist = our_to_tgt_knn_dist
                loss_1 = (our_sample_points_prob * dist).mean()
                loss_1 = loss_1.mean()

            end_event.record()
            th.cuda.synchronize()
            recon_loss_time = start_event.elapsed_time(end_event) / 1000.0

            '''
            2-3. Regularizers.
            '''
            start_event.record()

            real_regularizer = self.compute_topology_regularizer(
                preal
            )

            end_event.record()
            th.cuda.synchronize()
            real_loss_time = start_event.elapsed_time(end_event) / 1000.0

            recon_loss = loss_0 + loss_1
            loss = recon_loss + (real_regularizer * real_reg_weight)

            '''
            Update points.
            '''
            with th.no_grad():
                prev_preal = preal.clone()

            start_event.record()

            optimizer.zero_grad()
            loss.backward()

            end_event.record()
            th.cuda.synchronize()
            loss_backward_time = start_event.elapsed_time(end_event) / 1000.0

            # clip grads;
            with th.no_grad():
                preal_grad = preal.grad if preal.grad is not None else th.zeros_like(preal)

                # fix for nan grads;
                preal_grad_nan_idx = th.isnan(preal_grad)
                preal_grad[preal_grad_nan_idx] = 0.0

                if preal.grad is not None:
                    preal.grad.data = preal_grad

                preal_nan_grad_ratio = th.count_nonzero(preal_grad_nan_idx) / preal_grad_nan_idx.shape[0]

            optimizer.step()

            '''
            Prev mesh we got.
            '''
            with th.no_grad():
                # previous (non-differentiable) mesh we got;
                prev_mesh_faces = face_idx[prev_preal[face_idx].min(dim=-1).values > INIT_PREAL_THRESH]

                prev_num_points_on_mesh = th.unique(prev_mesh_faces).shape[0]
                prev_num_faces_on_mesh = prev_mesh_faces.shape[0]

            '''
            Bounding.
            '''
            with th.no_grad():
                preal.data = th.clamp(preal.data, min=0.0, max=1.0)
                preal.data[fixed_zero_idx] = 0.0

                # update points;
                self.preal = preal.clone()

                assert th.any(th.isnan(preal)) == False, "point real contains nan."
                assert th.any(th.isinf(preal)) == False, "point real contains inf."

            '''
            Logging
            '''
            with th.no_grad():
                self.writer.add_scalar(f"init_preal/loss", loss, step)
                self.writer.add_scalar(f"init_preal/recon_loss", recon_loss, step)
                self.writer.add_scalar(f"init_preal/real_regularizer", real_regularizer, step)

                self.writer.add_scalar(f"init_preal_info/num_faces_on_mesh", prev_num_faces_on_mesh, step)
                self.writer.add_scalar(f"init_preal_info/num_points_on_mesh", prev_num_points_on_mesh, step)

                # nan grad;
                self.writer.add_scalar(f"init_preal_nan/nan_grad_ratio", preal_nan_grad_ratio, step)

                # time;
                self.writer.add_scalar(f"init_preal_time/recon_loss_time", recon_loss_time, step)
                self.writer.add_scalar(f"init_preal_time/real_loss_time", real_loss_time, step)
                self.writer.add_scalar(f"init_preal_time/loss_backward_time", loss_backward_time, step)

                bar.set_description("loss: {:.4f}".format(loss))

            '''
            Saving
            '''
            if step % vis_steps == 0 or step == num_steps - 1:

                save_dir = os.path.join(self.writer.log_dir, f"save/init_preal")

                if RENDERING:

                    os.makedirs(save_dir, exist_ok=True)
                    self.save_mesh(
                        ppos,
                        prev_mesh_faces,
                        os.path.join(
                            save_dir,
                            f"step_{step}"
                        )
                    )

        # change preal to 0 or 1
        with th.no_grad():
            preal.data[preal > INIT_PREAL_THRESH] = 1.0
            preal.data[preal <= INIT_PREAL_THRESH] = 0.0
            self.preal = preal.detach().clone()

        # remove unnecessary points
        # only points with preal == 1.0 or adjacent to points with preal == 1.0 are kept
        max_adjacency = 2
        adjacency_edges = self.tgrid.tri_idx[:, [0, 1, 1, 2, 0, 2]].view(-1, 2)
        real_verts = th.where(preal == 1.0)[0]
        adjacency_counter = th.full_like(preal, max_adjacency, dtype=th.long)
        adjacency_counter[real_verts] = 0
        for _ in range(max_adjacency - 1):
            edge_adj = adjacency_counter[adjacency_edges]
            edge_adj_0 = edge_adj[:, 0]
            edge_adj_1 = edge_adj[:, 1]

            edge_vid_0 = adjacency_edges[:, 0]
            edge_vid_1 = adjacency_edges[:, 1]

            case_0 = (edge_adj_0 < edge_adj_1)
            case_1 = (edge_adj_0 > edge_adj_1)

            # case 0: set adjacency counter of edge_vid_1 to that of edge_vid_0 + 1
            tmp_counter = adjacency_counter[edge_vid_0] + 1
            tmp_counter[~case_0] = max_adjacency
            adjacency_counter = scatter(tmp_counter, edge_vid_1, out=adjacency_counter, dim=0, reduce='min')

            # case 1: set adjacency counter of edge_vid_0 to that of edge_vid_1 + 1
            tmp_counter = adjacency_counter[edge_vid_1] + 1
            tmp_counter[~case_1] = max_adjacency
            adjacency_counter = scatter(tmp_counter, edge_vid_0, out=adjacency_counter, dim=0, reduce='min')

        valid_verts = adjacency_counter < max_adjacency
        valid_verts_idx = th.where(valid_verts)[0]

        ppos = ppos[valid_verts_idx]
        preal = preal[valid_verts_idx]

        self.ppos = ppos.detach().clone()
        self.preal = preal.detach().clone()

        valid_verts_ratio = valid_verts.sum() / valid_verts.shape[0]
        self.logger.info(f"Point-wise real value initialization done: {ppos.shape[0]} points remain ({valid_verts_ratio * 100:.2f} % remain).")

    '''
    Point-wise position optimization with Minimum-Ball algorithm (fixed point-reals).
    '''
    def _refresh_ppos_optimizer(self):
        ppos = self.ppos.clone()
        ppos.requires_grad = True

        ppos_lr = self.ppos_lr
        optimizer = th.optim.Adam([
            {'params': [ppos], 'lr': ppos_lr},
        ])

        return optimizer, ppos

    def _geometry_sdist_to_prob(self, sdist: th.Tensor, sdist_unit: float, sigmoid_T: float):
        normalized_sdist = sdist / (sdist_unit)                     # [-sdist_unit, sdist_unit] -> [-1.0, 1.0]
        return th.sigmoid(normalized_sdist / sigmoid_T)

    def optimize_ppos(self):

        '''
        Refresh optimizer and variables
        '''
        optimizer, ppos = self._refresh_ppos_optimizer()

        num_steps = self.optimize_ppos_settings.num_steps
        vis_steps = self.optimize_ppos_settings.vis_steps

        '''
        Thresholds for signed distance used for probability computation
        '''
        sdist_unit = self.tgrid.apex_circumball_dist
        assert sdist_unit > 0, "[sdist_unit] should be positive."

        # if [sdist] is equal to [sdist_unit], it corresponds to sigmoid(MAX_INPUT) probability
        ppos_sigmoid_max_input = PPOS_SIGMOID_MAX_INPUT
        ppos_sigmoid_T = 1.0 / ppos_sigmoid_max_input   # temperature parameter for sigmoid function
        ppos_update_thresh = ((sdist_unit / 3.0) / ppos_sigmoid_max_input)

        '''
        Find query faces
        '''
        is_real_point = (self.preal == 1.0)
        real_points_idx = th.where(is_real_point)[0]
        real_points = ppos[real_points_idx]

        # we only care about these faces...
        with th.no_grad():
            qfaces_0 = qface_knn_spatial(real_points, QFACE_KNN_SPATIAL_K, 2)
            qfaces_0 = real_points_idx[qfaces_0]
            qfaces_1 = qface_dt(ppos, is_real_point)

            qfaces = th.cat([qfaces_0, qfaces_1], dim=0)
            qfaces = th.sort(qfaces, dim=-1)[0]
            qfaces = th.unique(qfaces, dim=0)
            prev_qfaces_nearest = None

        start_event = th.cuda.Event(enable_timing=True)
        end_event = th.cuda.Event(enable_timing=True)

        bar = tqdm(range(num_steps))
        for step in bar:

            '''
            Evaluate probability of query faces.
            '''
            start_event.record()

            qfaces_minball, qfaces_stable = MB3_V0.forward(ppos[qfaces[:, 0]], ppos[qfaces[:, 1]], ppos[qfaces[:, 2]])
            if prev_qfaces_nearest is None:
                qfaces_nearest, _ = knn_search(qfaces, qfaces_minball.center, qfaces_minball.radius, ppos)
                prev_qfaces_nearest = qfaces_nearest
            else:
                # first use prev nearest to cull out faces with very low probability;
                with th.no_grad():
                    qfaces_sdist = projection(qfaces, qfaces_minball.center, qfaces_minball.radius, ppos, prev_qfaces_nearest)
                    qfaces_probs = self._geometry_sdist_to_prob(qfaces_sdist, sdist_unit, ppos_sigmoid_T)
                    qfaces_probs = th.where(qfaces_stable, qfaces_probs, th.zeros_like(qfaces_probs))
                    likely_qfaces = qfaces_probs > PROB_THRESH

                    likely_qfaces_nearest, _ = knn_search(
                        qfaces[likely_qfaces],
                        qfaces_minball.center[likely_qfaces],
                        qfaces_minball.radius[likely_qfaces],
                        ppos
                    )
                    prev_qfaces_nearest[likely_qfaces] = likely_qfaces_nearest

            qfaces_sdist = projection(qfaces, qfaces_minball.center, qfaces_minball.radius, ppos, prev_qfaces_nearest)
            qfaces_probs = self._geometry_sdist_to_prob(qfaces_sdist, sdist_unit, ppos_sigmoid_T)
            qfaces_probs = th.where(qfaces_stable, qfaces_probs, th.zeros_like(qfaces_probs))

            curr_faces = qfaces[qfaces_probs > PROB_THRESH]
            curr_face_probs = qfaces_probs[qfaces_probs > PROB_THRESH]

            end_event.record()
            th.cuda.synchronize()
            prob_time = start_event.elapsed_time(end_event) / 1000.0

            '''
            Sample points from the faces.
            '''
            start_event.record()

            our_sample_points_pos, our_sample_points_face = self.sample_points_from_faces(ppos, curr_faces)
            our_sample_points_prob = curr_face_probs[our_sample_points_face]

            end_event.record()
            th.cuda.synchronize()
            sample_time = start_event.elapsed_time(end_event) / 1000.0

            '''
            Compute CD loss.
            '''
            start_event.record()

            '''
            2-1. CD loss from GT to OURS
            '''

            num_knn = MAX_KNN_K if len(our_sample_points_pos) > MAX_KNN_K else len(our_sample_points_pos)
            tgt_to_our_knn_idx, tgt_to_our_knn_dist = run_knn(
                self.target_point_positions,
                our_sample_points_pos,
                num_knn
            )
            tgt_to_our_face_knn_idx = our_sample_points_face[tgt_to_our_knn_idx]

            dist = tgt_to_our_knn_dist                                      # [# gt sample points, # k]
            prob_mat = curr_face_probs[tgt_to_our_face_knn_idx]             # [# tgt sample points, # k]

            '''
            Sorting: If a sample point from gt mesh finds a near point from a certain face,
            we do not consider another point from the same face in computing the loss.
            '''

            # =========

            sorted_indices = th.argsort(tgt_to_our_face_knn_idx, dim=1, stable=True)

            # Step 2: Rearrange A using sorted indices
            sorted_A = th.gather(prob_mat, 1, sorted_indices)

            # Step 3: Identify duplicates in sorted B
            sorted_B = th.gather(tgt_to_our_face_knn_idx, 1, sorted_indices)
            duplicate_mask = sorted_B[:, 1:] == sorted_B[:, :-1]
            # Pad the mask to match the shape of A and B
            padded_mask = th.cat([th.zeros(duplicate_mask.shape[0], 1, dtype=th.bool, device=DEVICE), duplicate_mask], dim=1)

            # Step 4: Revert A to the original order, applying the duplicate mask
            # First, set duplicates in sorted_A to 0
            sorted_A[padded_mask] = 0.0

            # Then, invert the sorted indices to get the original order
            inverse_indices = th.argsort(sorted_indices, dim=1)
            original_order_A = th.gather(sorted_A, 1, inverse_indices)

            prob_mat = original_order_A

            # =========

            # append one more column: fall back for all miss case;
            dist_n_col = th.ones((dist.shape[0], 1), dtype=th.float32, device=DEVICE) * (DOMAIN * 10)
            prob_n_col = th.ones((dist.shape[0], 1), dtype=th.float32, device=DEVICE)

            dist = th.cat([dist, dist_n_col], dim=-1)
            prob_mat = th.cat([prob_mat, prob_n_col], dim=-1)

            # =========

            n_prob_mat = 1.0 - prob_mat
            n_prob_mat_prod = th.cumprod(n_prob_mat, dim=-1)

            prob_mat[:, 1:] = prob_mat[:, 1:].clone() * n_prob_mat_prod[:, :-1]

            loss_0 = th.sum(prob_mat * dist, dim=-1)         # [# gt sample points,]
            loss_0 = loss_0.mean()

            '''
            2-2. CD loss from OURS to GT.
            '''
            num_knn = 1
            our_to_tgt_knn_idx, our_to_tgt_knn_dist = run_knn(
                our_sample_points_pos,
                self.target_point_positions,
                num_knn
            )

            dist = our_to_tgt_knn_dist.reshape(our_sample_points_prob.shape)
            loss_1 = (our_sample_points_prob * dist).mean()
            loss_1 = loss_1.mean()

            recon_loss = loss_0 + loss_1

            end_event.record()
            th.cuda.synchronize()
            recon_loss_time = start_event.elapsed_time(end_event) / 1000.0

            loss = recon_loss

            '''
            Update points.
            '''
            with th.no_grad():
                prev_ppos = ppos.clone()

            optimizer.zero_grad()

            start_event.record()
            loss.backward()
            end_event.record()
            th.cuda.synchronize()
            loss_backward_time = start_event.elapsed_time(end_event) / 1000.0

            # clip grads;
            with th.no_grad():
                ppos_grad = ppos.grad if ppos.grad is not None else th.zeros_like(ppos)

                # fix for nan grads;
                ppos_grad_nan_idx = th.any(th.isnan(ppos_grad), dim=-1)
                ppos_grad[ppos_grad_nan_idx] = 0.0

                if ppos.grad is not None:
                    ppos.grad.data = ppos_grad

                ppos_nan_grad_ratio = th.count_nonzero(ppos_grad_nan_idx) / ppos_grad_nan_idx.shape[0]

            optimizer.step()

            '''
            Bounding.
            '''
            start_event.record()
            with th.no_grad():
                # ppos
                ppos_curr_perturb = ppos - prev_ppos
                ppos_curr_perturb_len = th.norm(ppos_curr_perturb, dim=-1)
                ppos_curr_perturb_dir = ppos_curr_perturb / (ppos_curr_perturb_len.unsqueeze(-1) + 1e-6)

                ppos_to_bound = ppos_curr_perturb_len > ppos_update_thresh
                ppos_safe_perturb = ppos_curr_perturb
                ppos_safe_perturb[ppos_to_bound] = \
                    ppos_curr_perturb_dir[ppos_to_bound] * ppos_update_thresh

                ppos_safe_perturb_len = th.norm(ppos_safe_perturb, dim=-1)
                assert th.all(ppos_safe_perturb_len <= ppos_update_thresh), "Safe perturbation range is not satisfied."

                ppos.data = prev_ppos + ppos_safe_perturb

                self.ppos = ppos.clone()

            end_event.record()
            th.cuda.synchronize()
            bound_time = start_event.elapsed_time(end_event) / 1000.0

            '''
            Logging
            '''

            with th.no_grad():
                self.writer.add_scalar(f"update_ppos_loss/loss", loss, step)
                self.writer.add_scalar(f"update_ppos_loss/recon_loss", recon_loss, step)
                self.writer.add_scalar(f"update_ppos_loss/loss_0", loss_0, step)
                self.writer.add_scalar(f"update_ppos_loss/loss_1", loss_1, step)

                # nan grad;
                self.writer.add_scalar(f"update_ppos_nan/ppos_nan_grad_ratio", ppos_nan_grad_ratio, step)

                # time;
                self.writer.add_scalar(f"update_ppos_time/recon_loss_time", recon_loss_time, step)
                self.writer.add_scalar(f"update_ppos_time/loss_backward_time", loss_backward_time, step)
                self.writer.add_scalar(f"update_ppos_time/bound_time", bound_time, step)
                self.writer.add_scalar(f"update_ppos_time/prob_time", prob_time, step)
                self.writer.add_scalar(f"update_ppos_time/sample_time", sample_time, step)

                bar.set_description("loss: {:.4f}".format(loss))

            '''
            Saving
            '''
            if step % vis_steps == 0 or step == num_steps - 1:

                if RENDERING:
                    save_dir = os.path.join(self.writer.log_dir, f"save/optimize_ppos")
                    os.makedirs(save_dir, exist_ok=True)

                    # extract faces
                    valid_qfaces = qfaces_sdist > 0
                    vis_faces = qfaces[valid_qfaces]

                    self.save_mesh(
                        prev_ppos,
                        vis_faces,
                        os.path.join(
                            save_dir,
                            f"step_{step}"
                        )
                    )

        # extract faces / save final faces
        valid_qfaces = qfaces_sdist > 0
        vis_faces = qfaces[valid_qfaces]
        self.logger.info(f"Point-wise position optimization done: {vis_faces.shape[0]} faces remain.")
        self.dtfaces = vis_faces.clone()

        # save final points
        ppos = prev_ppos.detach().clone()
        self.ppos = ppos

        save_dir = os.path.join(self.writer.log_dir, f"save/optimize_ppos")
        self.save_mesh(
            ppos,
            vis_faces,
            os.path.join(save_dir, "final")
        )

        # set [preal] based on [vis_faces]
        points_on_vis_faces = th.unique(vis_faces)
        preal = th.zeros_like(ppos[:, 0])
        preal[points_on_vis_faces] = 1.0
        self.preal = preal


def subsample_point_cloud_with_grid(pc: th.Tensor, grid_cell_size: float):
    '''
    From input point cloud (pc), subsample subset of points using a regular grid.
    For each cell in the grid, select only one point among the points in the cell.
    '''
    pc_x = th.floor((pc[:, 0] - (-DOMAIN)) / grid_cell_size).long()
    pc_y = th.floor((pc[:, 1] - (-DOMAIN)) / grid_cell_size).long()
    pc_z = th.floor((pc[:, 2] - (-DOMAIN)) / grid_cell_size).long()
    pc_cell_id = th.stack([pc_x, pc_y, pc_z], dim=-1)
    pc_cell_id_pos = th.cat([pc_cell_id, pc], dim=-1)
    pc_cell_id_pos_unique = th.unique(pc_cell_id_pos, dim=0)
    _, u_pc_cell_id_cnt = th.unique(pc_cell_id_pos_unique[:, :3], return_counts=True, dim=0)
    u_pc_cell_id_cnt_cumsum = th.cumsum(u_pc_cell_id_cnt, dim=0)

    pc = pc_cell_id_pos_unique[u_pc_cell_id_cnt_cumsum - 1][:, 3:]

    return pc

def rescale_point_cloud_to_domain(pc: th.Tensor, margin: float = 0.9):
    '''
    Center by centroid and rescale to fit a sphere of radius [margin * DOMAIN],
    matching the normalization convention used elsewhere in the repo (see
    import_non_textured_mesh / import_textured_mesh in
    input/generate_mvrecon_3d_input.py).
    '''
    center = pc.mean(dim=0, keepdim=True)
    pc = pc - center
    max_norm = th.norm(pc, dim=-1).max()
    pc = (pc / max_norm) * (margin * DOMAIN)
    return pc

def load_input_point_cloud(init_tgrid_size: float, input_path: str, logdir: str, auto_rescale: bool = False):

    if input_path.endswith(".npy"):
        pc = np.load(input_path)
    elif input_path.endswith((".ply", ".obj", ".xyz", ".pcd")):
        loaded = trimesh.load(input_path, process=False)
        pc = np.array(loaded.vertices) if hasattr(loaded, "vertices") else np.array(loaded)
    else:
        raise ValueError(f"Unsupported point cloud file type: {input_path}")

    pc = th.tensor(pc[:, :3], device=DEVICE, dtype=th.float32)

    if auto_rescale:
        pc = rescale_point_cloud_to_domain(pc)

    pc_abs_max = th.max(th.abs(pc))
    if pc_abs_max >= DOMAIN:
        raise ValueError("Input point cloud should be within the domain [-DOMAIN, DOMAIN]. Please rescale your point cloud, or pass --auto-rescale.")

    # reduce number of points using grid for faster optimization;
    # set the grid cell size to be slightly smaller than the initial grid size;
    subsample_grid_cell_size = init_tgrid_size / 1.1
    pc = subsample_point_cloud_with_grid(pc, subsample_grid_cell_size)

    return pc

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="exp/config/d3/pcrecon.yaml")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--input-path", type=str, default="input/3d/pcrecon/example.npy")
    parser.add_argument("--no-log-time", action='store_true')
    parser.add_argument("--render", action='store_true')
    parser.add_argument("--auto-rescale", action='store_true', help="center and rescale the input point cloud to fit within [-DOMAIN, DOMAIN]")
    parser.add_argument("--use-mlflow", action='store_true')
    parser.add_argument("--mlflow-experiment", type=str, default="pcrecon_3d")
    args = parser.parse_args()

    # load settings from yaml file;
    with open(args.config, "r") as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)

    DEVICE = settings['device']
    settings['args']['seed'] = args.seed

    if args.render:
        RENDERING = True

    '''
    Setup log dir and logger.
    '''
    # setup logdir;
    logdir = settings['log_dir']
    if not args.no_log_time:
        logdir = logdir + time.strftime("/%Y_%m_%d_%H_%M_%S")
    logdir = setup_logdir(logdir)

    # setup logger;
    logger = get_logger("pcrecon_3d", os.path.join(logdir, "run.log"))

    # save settings;
    with open(os.path.join(logdir, "config.yaml"), "w") as f:
        yaml.dump(settings, f)
    th.random.manual_seed(args.seed)

    if args.use_mlflow:
        init_mlflow_run(
            experiment_name=args.mlflow_experiment,
            run_name=os.path.basename(logdir.rstrip("/")),
            settings=settings,
        )

    '''
    Arguments
    '''
    # initial grid size;
    init_tgrid_size = float(settings['args']['init_args']['grid_size'])

    # the unit interval to sample points from DMesh;
    # set this to be shorter than the grid size to guarantee at least one sample per DMesh face as much as possible;
    our_sample_points_interval = init_tgrid_size * 0.5

    '''
    Input point cloud
    '''
    input_path = args.input_path
    try:
        gt_pc = load_input_point_cloud(init_tgrid_size, input_path, logdir, auto_rescale=args.auto_rescale)
    except ValueError as e:
        logger.exception(str(e))
        sys.exit(1)

    logger.info(f"Num. Input Point Cloud: {len(gt_pc)}")
    logger.info(f"Init. Tet Grid Size: {init_tgrid_size}")

    '''
    Initialize optimizer
    '''
    init_args = settings['args']['init_args']
    lr_settings = edict(settings['args']['lr'])
    init_preal_settings = edict(settings['args']['init_preal'])
    optimize_ppos_settings = edict(settings['args']['optimize_ppos'])

    optimizer = PCRecon3D(
        logdir,
        logger,

        gt_pc,
        our_sample_points_interval,

        init_args,

        lr_settings,

        init_preal_settings,
        optimize_ppos_settings,

        args.use_mlflow,
    )

    try:
        optimizer.optimize()
    except Exception as e:
        logger.exception(str(e))
        sys.exit(1)
    finally:
        if args.use_mlflow:
            import mlflow
            mlflow.end_run()
