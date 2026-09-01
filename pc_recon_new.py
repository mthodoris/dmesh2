'''
DMesh++ point-cloud -> triangle-mesh reconstruction.

This is a from-scratch implementation of the 3D point-cloud reconstruction task
described in the DMesh++ paper (Son et al.), Sec. 4 and Appendices 7-9, but for
which no code was released with the paper. It follows the paper as literally as
the primitives in this repo allow:

  * Initialization (App. 8.2.1 "Point Cloud Init", App. 9.3.1)
      - estimate the input point-cloud density (mean nearest-neighbour distance)
      - voxel-downsample the input cloud at that density
      - seed a body-centred-cubic tetrahedral grid whose edge length is
        3x the density; every face of this grid trivially satisfies the
        Minimum-Ball condition (Def. 3.1 / Fig. 16)

  * Step 1 - Real-value initialization (App. 8.2.1):
      fix point positions, optimize per-point real values psi with the expected
      Chamfer Distance loss (inherited unchanged from DMesh [42], App. 8.1.1)
      plus lambda_real * mean(psi).

  * Step 2 - Position optimization (Algorithm 2, App. 7.1/7.4):
      fix psi, optimize point positions. Face probability is Lambda(F) =
      Lambda_min(F) * Lambda_real(F) with Lambda_min from the Minimum-Ball
      condition (Eq. 2-5). Query faces are refreshed every n_1 steps and their
      K nearest neighbours are cached (App. 7.4). Loss is Eq. (11):
      L_recon + lambda_qual * L_qual (Eq. 14) + lambda_real * L_real (Eq. 15).

  * Step 3 - Real-value re-optimization (App. 8.2.3), optional (0 steps for 3D
      point clouds per App. 9.3.1): re-optimize face real values on the faces of
      the Delaunay triangulation that satisfy the Minimum-Ball condition
      (Lemma 3.2).

  * Subdivision between epochs (App. 8.2.4). Only runs when num_epochs > 1.

The final mesh is extracted (Lemma 3.2) by taking the Delaunay triangulation of
the optimized points and keeping the faces whose vertices are all real and whose
minimum bounding ball contains no other point.

The Reinforce-Ball point-weight pruning of App. 10 is intentionally NOT included:
it is 2D-only in the paper and disabled for 3D point clouds.
'''

import os
import sys
import time
import argparse

import numpy as np
import torch as th
import trimesh
import yaml
from easydict import EasyDict as edict
from tqdm import tqdm
from torch_scatter import scatter
from torch.utils.tensorboard import SummaryWriter

from exp.utils.utils import *
from exp.utils.dmesh import *
from exp.utils.common import *
from exp.utils.logging import get_logger
from exp.utils.mlflow_utils import init_mlflow_run, MetricWriter

from input.common import DOMAIN

from mindiffdt.tgrid import TetGrid
from mindiffdt.qface import qface_knn_spatial, qface_dt
from mindiffdt.minball import MB3_V0, Ball
from mindiffdt.projection import (
    knn_search, knn_search_multi, projection, projection_multi,
)
from mindiffdt.cgaldt import CGALDTStruct
from mindiffdt.utils import tensor_subtract_1

DEVICE = 'cuda:0'

MAX_KNN_K = 40                    # k for the GT->ours term of the expected Chamfer Distance
MINBALL_CHUNK_SIZE = int(5e5)    # max query faces per MB3_V0.forward call (bounds peak memory)
MESH_FORMAT = 'obj'
RENDERING = False                # dump intermediate meshes during optimization


def minball_forward_chunked(p1: th.Tensor, p2: th.Tensor, p3: th.Tensor, chunk_size: int = None):
    '''
    Chunked wrapper around [MB3_V0.forward]: each face's minimum bounding ball is
    independent, so we process the faces in fixed-size chunks and concatenate,
    bounding peak memory by [chunk_size].
    '''
    if chunk_size is None:
        chunk_size = MINBALL_CHUNK_SIZE

    n = p1.shape[0]
    if n <= chunk_size:
        return MB3_V0.forward(p1, p2, p3)

    centers, radii, stable_masks = [], [], []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        ball, stable = MB3_V0.forward(p1[start:end], p2[start:end], p3[start:end])
        centers.append(ball.center)
        radii.append(ball.radius)
        stable_masks.append(stable)

    return Ball(th.cat(centers, dim=0), th.cat(radii, dim=0)), th.cat(stable_masks, dim=0)


class PCReconNew:

    def __init__(self,
                 logdir,
                 logger,
                 target_point_positions: th.Tensor,
                 density: float,
                 our_sample_interval: float,
                 init_args: dict,
                 lr_settings: edict,
                 num_epochs: int,
                 init_preal_settings: edict,
                 optimize_ppos_settings: edict,
                 optimize_freal_settings: edict,
                 use_mlflow: bool = False):

        self.logger = logger
        self.logdir = logdir
        self.writer = MetricWriter(SummaryWriter(logdir), use_mlflow)

        self.target_point_positions = target_point_positions
        self.density = float(density)
        self.our_sample_interval = float(our_sample_interval)
        self.init_args = edict(init_args)
        self.num_epochs = int(num_epochs)

        self.ppos_lr = float(lr_settings.pos)
        self.preal_lr = float(lr_settings.real)

        self.init_preal_settings = edict(init_preal_settings)
        self.init_preal_settings.num_steps = int(float(self.init_preal_settings.num_steps))
        self.init_preal_settings.vis_steps = int(float(self.init_preal_settings.vis_steps))
        self.init_preal_settings.real_reg_weight = float(self.init_preal_settings.real_reg_weight)

        self.optimize_ppos_settings = edict(optimize_ppos_settings)
        for k in ("num_steps", "vis_steps", "qface_refresh_steps", "nn_cache_steps", "nn_cache_size"):
            self.optimize_ppos_settings[k] = int(float(self.optimize_ppos_settings[k]))
        self.optimize_ppos_settings.quality_reg_weight = float(self.optimize_ppos_settings.quality_reg_weight)
        self.optimize_ppos_settings.real_reg_weight = float(self.optimize_ppos_settings.real_reg_weight)

        self.optimize_freal_settings = edict(optimize_freal_settings)
        for k in ("num_steps", "vis_steps"):
            self.optimize_freal_settings[k] = int(float(self.optimize_freal_settings[k]))
        self.optimize_freal_settings.real_reg_weight = float(self.optimize_freal_settings.real_reg_weight)

        # mesh state
        self.tgrid = TetGrid(DEVICE)
        self.ppos: th.Tensor = None       # [V, 3]
        self.preal: th.Tensor = None      # [V]  in {0, 1} after step 1
        self.dtfaces: th.Tensor = None    # [F, 3]

        self.global_optim_start_time = 0.0

    # ------------------------------------------------------------------ #
    # Initialization (App. 8.2.1)                                        #
    # ------------------------------------------------------------------ #
    def init_grid(self):
        '''
        BCC tetrahedral grid with edge length = grid_size_density_scale * density.
        Every face of this grid satisfies the Minimum-Ball condition (Fig. 16).
        '''
        grid_size = self.init_args.grid_size_density_scale * self.density

        # Cap the grid resolution: "3x density" on a dense cloud can ask for tens of
        # millions of grid vertices, which blows up the per-grid-face tensors in
        # step 1 and the query-face enumeration / CGAL DT in step 2. The paper's
        # grids are ~128^3 (App. 9.4.1). max_grid_res bounds it (default 192).
        max_grid_res = int(self.init_args.get("max_grid_res", 192))
        min_grid_size = (2.0 * DOMAIN) / max_grid_res
        if grid_size < min_grid_size:
            self.logger.warning(
                f"Requested grid edge {grid_size:.5f} exceeds the {max_grid_res}^3 "
                f"resolution cap; clamping to {min_grid_size:.5f}. Raise "
                f"init_args.max_grid_res (more memory) or grid_size_density_scale "
                f"if you need finer detail.")
            grid_size = min_grid_size

        self.tgrid.init((-DOMAIN, -DOMAIN, -DOMAIN), (DOMAIN, DOMAIN, DOMAIN), grid_size)

        self.ppos = self.tgrid.verts.clone()
        self.preal = th.zeros((self.ppos.shape[0],), dtype=th.float32, device=DEVICE)
        self.logger.info(f"Init grid: edge length {grid_size:.5f} ({self.ppos.shape[0]} points).")

    # ------------------------------------------------------------------ #
    # Sampling / losses                                                  #
    # ------------------------------------------------------------------ #
    def sample_points_from_faces(self, ppos: th.Tensor, faces: th.Tensor):
        '''
        Area-weighted random barycentric sampling. Each face gets a sample count
        proportional to its area (relative to an equilateral triangle of edge
        [our_sample_interval]), with a minimum of one sample per face.
        '''
        v0 = ppos[faces[:, 0]]
        v1 = ppos[faces[:, 1]]
        v2 = ppos[faces[:, 2]]
        e1 = v1 - v0
        e2 = v2 - v0

        with th.no_grad():
            area = 0.5 * th.norm(th.cross(e1, e2, dim=-1), dim=-1)
            ref_area = (np.sqrt(3.0) / 4.0) * (self.our_sample_interval ** 2)
            num_samples = th.clamp((area / ref_area).round().long(), min=1)
            face_id = th.repeat_interleave(th.arange(faces.shape[0], device=DEVICE), num_samples)

            r1 = th.rand((face_id.shape[0],), device=DEVICE)
            r2 = th.rand((face_id.shape[0],), device=DEVICE)
            sqrt_r1 = th.sqrt(r1)
            bary_u = (1.0 - sqrt_r1).unsqueeze(-1)
            bary_v = (r2 * sqrt_r1).unsqueeze(-1)

        sample_pos = v0[face_id] + e1[face_id] * bary_u + e2[face_id] * bary_v
        return sample_pos, face_id

    def _cd_gt_to_ours(self, dist: th.Tensor, face_knn_idx: th.Tensor, face_probs: th.Tensor):
        '''
        Expected Chamfer Distance, GT -> ours direction (DMesh [42], App. 8.1.1).

        For each GT sample point we alpha-composite over its k nearest candidate
        sample points (sorted so each face contributes at most once), weighting
        each by the probability of the face it was sampled from, with a "miss"
        fall-back column.
        @ dist:          [G, k]  distances to the k nearest "our" sample points
        @ face_knn_idx:  [G, k]  face id of each of those nearest sample points
        @ face_probs:    [F]     probability of every current candidate face
        '''
        prob_mat = face_probs[face_knn_idx]                                     # [G, k]

        # keep only the first occurrence of each face id within a row
        sorted_indices = th.argsort(face_knn_idx, dim=1, stable=True)
        sorted_prob = th.gather(prob_mat, 1, sorted_indices)
        sorted_face = th.gather(face_knn_idx, 1, sorted_indices)
        dup = sorted_face[:, 1:] == sorted_face[:, :-1]
        dup = th.cat([th.zeros((dup.shape[0], 1), dtype=th.bool, device=DEVICE), dup], dim=1)
        sorted_prob = sorted_prob.clone()
        sorted_prob[dup] = 0.0
        inv = th.argsort(sorted_indices, dim=1)
        prob_mat = th.gather(sorted_prob, 1, inv)

        # miss fall-back column
        dist = th.cat([dist, th.full((dist.shape[0], 1), DOMAIN * 10.0, device=DEVICE)], dim=-1)
        prob_mat = th.cat([prob_mat, th.ones((dist.shape[0], 1), device=DEVICE)], dim=-1)

        n_prob_prod = th.cumprod(1.0 - prob_mat, dim=-1)
        prob_mat = prob_mat.clone()
        prob_mat[:, 1:] = prob_mat[:, 1:].clone() * n_prob_prod[:, :-1]

        return th.sum(prob_mat * dist, dim=-1).mean()

    def _cd_ours_to_gt(self, dist: th.Tensor, sample_probs: th.Tensor):
        '''
        Expected Chamfer Distance, ours -> GT direction: distance from each of our
        sample points to the nearest GT point, weighted by its face probability.
        '''
        return (sample_probs * dist.reshape(sample_probs.shape)).mean()

    @staticmethod
    def _sdist_to_prob(sdist: th.Tensor, sdist_unit: float, sigmoid_T: float):
        '''
        Lambda_min: map the Minimum-Ball signed distance to a probability with a
        sigmoid (Eq. 5). A signed distance of +sdist_unit maps to
        sigmoid(PPOS_SIGMOID_MAX_INPUT).
        '''
        return th.sigmoid((sdist / sdist_unit) / sigmoid_T)

    def compute_quality_loss(self, ppos: th.Tensor, faces: th.Tensor, face_probs: th.Tensor):
        '''
        L_qual (Eq. 14): probability-weighted mean triangle aspect ratio.
        '''
        if faces.shape[0] == 0:
            return th.zeros((), device=DEVICE)
        ar = triangle_aspect_ratio(ppos, faces)
        return (face_probs * ar).sum() / (face_probs.sum().detach() + 1e-6)

    # ------------------------------------------------------------------ #
    # Saving / final extraction                                          #
    # ------------------------------------------------------------------ #
    @th.no_grad()
    def save_mesh(self, ppos: th.Tensor, faces: th.Tensor, path: str):
        os.makedirs(path, exist_ok=True)
        mesh = trimesh.Trimesh(vertices=ppos.detach().cpu().numpy(),
                               faces=faces.detach().cpu().numpy(), process=False)
        mesh.export(os.path.join(path, f"mesh.{MESH_FORMAT}"))
        with open(os.path.join(path, "time_sec.txt"), "w") as f:
            f.write(str(time.time() - self.global_optim_start_time))
        with open(os.path.join(path, "mesh_info.txt"), "w") as f:
            f.write(f"num_points: {ppos.shape[0]}\n")
            f.write(f"num_faces: {faces.shape[0]}\n")

    @th.no_grad()
    def _extract_faces_minball(self, ppos: th.Tensor, real_mask: th.Tensor):
        '''
        Lemma 3.2 / Step 3: the Minimum-Ball faces are a subset of the Delaunay
        triangulation. Compute the DT of the points and keep the faces whose
        vertices are all real and whose minimum bounding ball contains no other
        point (signed distance > 0).
        '''
        dt = CGALDTStruct.forward(ppos)
        tets = dt.dsimp_point_id.to(dtype=th.long)
        faces = tets[:, [0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3]].reshape(-1, 3)
        faces = th.sort(faces, dim=-1)[0]
        faces = th.unique(faces, dim=0)

        faces = faces[real_mask[faces].all(dim=-1)]
        if faces.shape[0] == 0:
            return faces

        ball, stable = minball_forward_chunked(ppos[faces[:, 0]], ppos[faces[:, 1]], ppos[faces[:, 2]])
        nearest, _ = knn_search(faces, ball.center, ball.radius, ppos)
        sdist = projection(faces, ball.center, ball.radius, ppos, nearest)
        return faces[stable & (sdist > 0.0)]

    # ------------------------------------------------------------------ #
    # Main loop                                                          #
    # ------------------------------------------------------------------ #
    def optimize(self):
        self.global_optim_start_time = time.time()

        self.init_grid()

        self.logger.info("=== Step 1: real-value initialization ===")
        self.init_preal()

        for epoch in range(self.num_epochs):
            self.logger.info(f"=== Epoch {epoch}: position optimization ===")
            if self.optimize_ppos_settings.num_steps > 0:
                self.optimize_ppos(epoch)

            if self.optimize_freal_settings.num_steps > 0:
                self.logger.info(f"=== Epoch {epoch}: real-value re-optimization ===")
                self.optimize_freal(epoch)

            if epoch < self.num_epochs - 1:
                self.logger.info(f"=== Epoch {epoch}: subdivision ===")
                self.subdivide(epoch)

        # Final mesh. If step 3 ran on the last epoch, [self.dtfaces] already holds
        # its post-processed result (App. 8.2.3); otherwise extract via Lemma 3.2.
        if self.optimize_freal_settings.num_steps <= 0:
            self.dtfaces = self._extract_faces_minball(self.ppos, self.preal == 1.0)
        save_dir = os.path.join(self.writer.log_dir, "result")
        self.save_mesh(self.ppos, self.dtfaces, save_dir)
        self.logger.info(f"Done: {self.ppos.shape[0]} points, {self.dtfaces.shape[0]} faces -> {save_dir}")

    # ------------------------------------------------------------------ #
    # Step 1: real-value initialization                                  #
    # ------------------------------------------------------------------ #
    def init_preal(self):
        s = self.init_preal_settings
        ppos = self.ppos.clone()
        target = self.target_point_positions

        preal = self.preal.clone()
        preal.requires_grad = True
        optimizer = th.optim.Adam([preal], lr=self.preal_lr)

        face_idx = self.tgrid.tri_idx.to(dtype=th.long)
        near_thresh = self.init_args.near_thresh_scale * self.density

        # --- seed: faces whose minimum bounding ball is near the input cloud ---
        with th.no_grad():
            ball, stable = minball_forward_chunked(
                ppos[face_idx[:, 0]], ppos[face_idx[:, 1]], ppos[face_idx[:, 2]])
            ball_nn_dist = run_knn(ball.center, target, 1)[1].squeeze(-1) - ball.radius
            possible = stable & (ball_nn_dist <= near_thresh)

            possible_face_verts = face_idx[possible].unique()
            # a face is a candidate only if all three of its vertices are candidates
            # (boolean-mask lookup, not th.isin: the latter can broadcast to a
            #  [num_faces*3, num_candidate_verts] tensor and OOM on a fine grid)
            is_candidate_vert = th.zeros_like(ppos[:, 0], dtype=th.bool)
            is_candidate_vert[possible_face_verts] = True
            possible_face_idx = face_idx[is_candidate_vert[face_idx].all(dim=-1)]

            fixed_zero = th.ones_like(ppos[:, 0], dtype=th.bool)
            fixed_zero[possible_face_verts] = False
            preal.data[fixed_zero] = 0.0
            preal.data[~fixed_zero] = 1.0

            # sample our mesh once (positions are fixed in step 1)
            our_pos, our_face = self.sample_points_from_faces(ppos, possible_face_idx)

            num_knn = min(MAX_KNN_K, len(our_pos))
            t2o_idx, t2o_dist = run_knn(target, our_pos, num_knn)
            t2o_face_knn_idx = our_face[t2o_idx]
            o2t_dist = run_knn(our_pos, target, 1)[1].squeeze(-1)

        bar = tqdm(range(s.num_steps))
        for step in bar:
            self._set_lr(optimizer, self.preal_lr, s.lr_schedule, step, s.num_steps)

            face_prob = dmin(preal[possible_face_idx], k=DMIN_K)               # Lambda_real
            our_sample_prob = face_prob[our_face]

            loss_0 = self._cd_gt_to_ours(t2o_dist, t2o_face_knn_idx, face_prob)
            loss_1 = self._cd_ours_to_gt(o2t_dist, our_sample_prob)
            real_reg = preal.mean()
            recon_loss = loss_0 + loss_1
            loss = recon_loss + s.real_reg_weight * real_reg

            optimizer.zero_grad()
            loss.backward()
            with th.no_grad():
                if preal.grad is not None:
                    preal.grad.nan_to_num_(0.0)
            optimizer.step()

            with th.no_grad():
                preal.data.clamp_(0.0, 1.0)
                preal.data[fixed_zero] = 0.0
                self.preal = preal.detach().clone()

            self.writer.add_scalar("init_preal/loss", loss, step)
            self.writer.add_scalar("init_preal/recon_loss", recon_loss, step)
            self.writer.add_scalar("init_preal/real_reg", real_reg, step)
            bar.set_description(f"[step1] loss {loss.item():.5f}")
            if step % 50 == 0 or step == s.num_steps - 1:
                self.logger.info(f"[step1] {step}/{s.num_steps} loss={loss.item():.5f}")

        # binarize psi and drop points that are neither real nor adjacent to a real point
        with th.no_grad():
            preal = preal.detach()
            preal = th.where(preal > INIT_PREAL_THRESH,
                             th.ones_like(preal), th.zeros_like(preal))

            keep = self._keep_real_and_adjacent(preal, face_idx)
            self.ppos = ppos[keep].detach().clone()
            self.preal = preal[keep].detach().clone()

        ratio = 100.0 * self.preal.numel() / preal.numel()
        self.logger.info(f"[step1] done: {self.ppos.shape[0]} points remain ({ratio:.1f}%), "
                         f"{int((self.preal == 1.0).sum())} real.")
        if RENDERING:
            self.save_mesh(self.ppos, self._extract_faces_minball(self.ppos, self.preal == 1.0),
                           os.path.join(self.writer.log_dir, "save/init_preal/final"))

    @staticmethod
    @th.no_grad()
    def _keep_real_and_adjacent(preal: th.Tensor, face_idx: th.Tensor):
        '''Keep points with psi == 1 or sharing a grid edge with such a point.'''
        edges = face_idx[:, [0, 1, 1, 2, 0, 2]].reshape(-1, 2)
        counter = th.full_like(preal, 2, dtype=th.long)
        counter[preal == 1.0] = 0
        c0 = counter[edges[:, 0]] + 1
        counter = scatter(c0, edges[:, 1], out=counter.clone(), dim=0, reduce='min')
        c1 = counter[edges[:, 1]] + 1
        counter = scatter(c1, edges[:, 0], out=counter.clone(), dim=0, reduce='min')
        return th.where(counter < 2)[0]

    # ------------------------------------------------------------------ #
    # Step 2: position optimization (Algorithm 2)                        #
    # ------------------------------------------------------------------ #
    def optimize_ppos(self, epoch: int):
        s = self.optimize_ppos_settings
        label = f"e{epoch}_optimize_ppos"

        ppos = self.ppos.clone()
        ppos.requires_grad = True
        optimizer = th.optim.Adam([{'params': [ppos], 'lr': self.ppos_lr}])

        # signed-distance unit halves every epoch (coarse-to-fine)
        sdist_unit = self.tgrid.apex_circumball_dist / (2 ** epoch)
        assert sdist_unit > 0
        sigmoid_T = 1.0 / PPOS_SIGMOID_MAX_INPUT
        update_thresh = (sdist_unit / 4.0) / PPOS_SIGMOID_MAX_INPUT

        is_real_point = (self.preal == 1.0)
        real_points_idx = th.where(is_real_point)[0]

        K = max(1, s.nn_cache_size)

        @th.no_grad()
        def gather_qfaces(pos):
            q0 = real_points_idx[qface_knn_spatial(pos[real_points_idx], QFACE_KNN_SPATIAL_K, 2)]
            q1 = qface_dt(pos, is_real_point)
            q = th.cat([q0, q1], dim=0)
            return th.unique(th.sort(q, dim=-1)[0], dim=0)

        qfaces = gather_qfaces(ppos.detach())
        qfaces_nearest = th.zeros_like(qfaces[:, 0], dtype=th.long)

        likely_qfaces = qfaces[:0].clone()
        likely_nearest = th.zeros((0, K), dtype=th.long, device=DEVICE)
        likely_valid = th.zeros((0, K), dtype=th.bool, device=DEVICE)

        @th.no_grad()
        def rebuild_cache():
            nonlocal likely_qfaces, likely_nearest, likely_valid, qfaces_nearest
            if qfaces.shape[0] == 0:
                return
            ball, _ = minball_forward_chunked(ppos[qfaces[:, 0]], ppos[qfaces[:, 1]], ppos[qfaces[:, 2]])
            # coarse cull with the previously cached nearest
            sdist = projection(qfaces, ball.center, ball.radius, ppos, qfaces_nearest)
            like0 = self._sdist_to_prob(sdist, sdist_unit, sigmoid_T) > PROB_THRESH
            if not like0.any():
                likely_qfaces = qfaces[:0].clone()
                likely_nearest = th.zeros((0, K), dtype=th.long, device=DEVICE)
                likely_valid = th.zeros((0, K), dtype=th.bool, device=DEVICE)
                return
            # true nearest for the survivors
            n0, _ = knn_search(qfaces[like0], ball.center[like0], ball.radius[like0], ppos)
            qfaces_nearest = qfaces_nearest.clone()
            qfaces_nearest[like0] = n0
            sdist0 = projection(qfaces[like0], ball.center[like0], ball.radius[like0], ppos, n0)
            keep = self._sdist_to_prob(sdist0, sdist_unit, sigmoid_T) > PROB_THRESH

            likely_qfaces = qfaces[like0][keep]
            lc = ball.center[like0][keep]
            lr = ball.radius[like0][keep]
            if K > 1:
                likely_nearest, likely_valid = knn_search_multi(likely_qfaces, lc, lr, ppos, K)
            else:
                likely_nearest = n0[keep].unsqueeze(-1)
                likely_valid = th.ones_like(likely_nearest, dtype=th.bool)

        rebuild_cache()

        bar = tqdm(range(s.num_steps))
        for step in bar:
            # --- refresh query faces (Algorithm 2, line 8) ---
            if step > 0 and step % s.qface_refresh_steps == 0:
                with th.no_grad():
                    new_q = gather_qfaces(ppos.detach())
                    added = tensor_subtract_1(new_q, qfaces)
                    if added.shape[0] > 0:
                        qfaces = th.cat([qfaces, added], dim=0)
                        qfaces_nearest = th.cat(
                            [qfaces_nearest, th.zeros_like(added[:, 0], dtype=th.long)], dim=0)
                rebuild_cache()
            elif step % s.nn_cache_steps == 0:
                rebuild_cache()

            # --- differentiable probability of the likely query faces ---
            if likely_qfaces.shape[0] == 0:
                self.logger.warning(f"[{label}] step {step}: no likely query faces.")
                break

            ball, stable = minball_forward_chunked(
                ppos[likely_qfaces[:, 0]], ppos[likely_qfaces[:, 1]], ppos[likely_qfaces[:, 2]])
            lq = likely_qfaces[stable]
            sdist = projection_multi(lq, ball.center[stable], ball.radius[stable],
                                     ppos, likely_nearest[stable], likely_valid[stable])
            probs = self._sdist_to_prob(sdist, sdist_unit, sigmoid_T)
            keep = probs > PROB_THRESH
            curr_faces = lq[keep]
            curr_face_probs = probs[keep]
            if curr_faces.shape[0] == 0:
                self.logger.warning(f"[{label}] step {step}: no faces above probability threshold.")
                continue

            # --- Eq. (11) loss ---
            our_pos, our_face = self.sample_points_from_faces(ppos, curr_faces)
            our_sample_prob = curr_face_probs[our_face]

            num_knn = min(MAX_KNN_K, len(our_pos))
            t2o_idx, t2o_dist = run_knn(self.target_point_positions, our_pos, num_knn)
            t2o_face_knn_idx = our_face[t2o_idx]
            o2t_dist = run_knn(our_pos, self.target_point_positions, 1)[1].squeeze(-1)

            loss_0 = self._cd_gt_to_ours(t2o_dist, t2o_face_knn_idx, curr_face_probs)
            loss_1 = self._cd_ours_to_gt(o2t_dist, our_sample_prob)
            recon_loss = loss_0 + loss_1
            qual_loss = self.compute_quality_loss(ppos, curr_faces, curr_face_probs)
            real_loss = self.preal.mean()                                    # psi fixed in step 2

            loss = (recon_loss
                    + s.quality_reg_weight * qual_loss
                    + s.real_reg_weight * real_loss)

            with th.no_grad():
                prev_ppos = ppos.clone()
            optimizer.zero_grad()
            loss.backward()
            with th.no_grad():
                if ppos.grad is not None:
                    ppos.grad.nan_to_num_(0.0)
            optimizer.step()

            # --- bound the per-step displacement (keeps the DT/Minimum-Ball valid) ---
            with th.no_grad():
                delta = ppos - prev_ppos
                dlen = th.norm(delta, dim=-1, keepdim=True)
                scale = th.clamp(update_thresh / (dlen + 1e-9), max=1.0)
                ppos.data = prev_ppos + delta * scale
                self.ppos = ppos.detach().clone()

            self.writer.add_scalar(f"{label}/loss", loss, step)
            self.writer.add_scalar(f"{label}/recon_loss", recon_loss, step)
            self.writer.add_scalar(f"{label}/quality_loss", qual_loss, step)
            self.writer.add_scalar(f"{label}/num_faces", curr_faces.shape[0], step)
            self.writer.add_scalar(f"{label}/num_qfaces", qfaces.shape[0], step)
            bar.set_description(f"[step2] loss {loss.item():.5f}")
            if step % 100 == 0 or step == s.num_steps - 1:
                self.logger.info(f"[step2] {step}/{s.num_steps} loss={loss.item():.5f} "
                                 f"faces={curr_faces.shape[0]}")

            if RENDERING and (step % s.vis_steps == 0 or step == s.num_steps - 1):
                self.save_mesh(self.ppos, self._extract_faces_minball(self.ppos, is_real_point),
                               os.path.join(self.writer.log_dir, f"save/{label}/step_{step}"))

        # commit result of this epoch
        self.dtfaces = self._extract_faces_minball(self.ppos, is_real_point)
        points_on_mesh = th.unique(self.dtfaces) if self.dtfaces.numel() > 0 else th.tensor([], dtype=th.long, device=DEVICE)
        preal = th.zeros_like(self.ppos[:, 0])
        preal[points_on_mesh] = 1.0
        self.preal = preal
        self.logger.info(f"[step2] epoch {epoch} done: {self.dtfaces.shape[0]} faces.")

    # ------------------------------------------------------------------ #
    # Step 3: real-value re-optimization (App. 8.2.3)                    #
    # ------------------------------------------------------------------ #
    def optimize_freal(self, epoch: int):
        s = self.optimize_freal_settings
        label = f"e{epoch}_optimize_freal"

        ppos = self.ppos.clone()
        faces = self._extract_faces_minball(ppos, self.preal == 1.0)
        if faces.shape[0] == 0:
            self.logger.warning(f"[{label}] no candidate faces; skipping.")
            return

        freal = th.ones((faces.shape[0],), dtype=th.float32, device=DEVICE, requires_grad=True)
        optimizer = th.optim.Adam([freal], lr=self.preal_lr)

        bar = tqdm(range(s.num_steps))
        for step in bar:
            face_prob = th.clamp(freal, 0.0, 1.0)
            our_pos, our_face = self.sample_points_from_faces(ppos, faces)
            our_sample_prob = face_prob[our_face]

            num_knn = min(MAX_KNN_K, len(our_pos))
            t2o_idx, t2o_dist = run_knn(self.target_point_positions, our_pos, num_knn)
            t2o_face_knn_idx = our_face[t2o_idx]
            o2t_dist = run_knn(our_pos, self.target_point_positions, 1)[1].squeeze(-1)

            loss_0 = self._cd_gt_to_ours(t2o_dist, t2o_face_knn_idx, face_prob)
            loss_1 = self._cd_ours_to_gt(o2t_dist, our_sample_prob)
            recon_loss = loss_0 + loss_1
            loss = recon_loss + s.real_reg_weight * face_prob.mean()

            optimizer.zero_grad()
            loss.backward()
            with th.no_grad():
                if freal.grad is not None:
                    freal.grad.nan_to_num_(0.0)
            optimizer.step()
            with th.no_grad():
                freal.data.clamp_(0.0, 1.0)

            self.writer.add_scalar(f"{label}/loss", loss, step)
            bar.set_description(f"[step3] loss {loss.item():.5f}")
            if step % 100 == 0 or step == s.num_steps - 1:
                self.logger.info(f"[step3] {step}/{s.num_steps} loss={loss.item():.5f}")

        with th.no_grad():
            kept = faces[freal > 0.5]
            self.dtfaces = kept.clone()
            preal = th.zeros_like(self.ppos[:, 0])
            preal[th.unique(kept)] = 1.0
            self.preal = preal
        self.logger.info(f"[step3] epoch {epoch} done: {self.dtfaces.shape[0]} faces.")

    # ------------------------------------------------------------------ #
    # Subdivision between epochs (App. 8.2.4)                            #
    # ------------------------------------------------------------------ #
    @th.no_grad()
    def subdivide(self, epoch: int):
        ppos = self.ppos
        preal = self.preal
        faces = self.dtfaces
        if faces.shape[0] == 0:
            self.logger.warning("[subdiv] empty mesh; skipping.")
            return

        # 1. insert psi = 0 points at the circumcentres of real DT faces that are
        #    NOT in the current mesh, to suppress those undesirable faces (Fig. 17)
        real_dt_faces = self._extract_faces_minball(ppos, preal == 1.0)
        undesirable = tensor_subtract_1(real_dt_faces, faces)
        new_pos = []
        new_real = []
        if undesirable.shape[0] > 0:
            ball, _ = minball_forward_chunked(
                ppos[undesirable[:, 0]], ppos[undesirable[:, 1]], ppos[undesirable[:, 2]])
            new_pos.append(ball.center)
            new_real.append(th.zeros((ball.center.shape[0],), device=DEVICE))

        # 2. insert psi = 1 midpoints on every edge of the current mesh
        edges = th.unique(th.sort(faces[:, [0, 1, 1, 2, 0, 2]].reshape(-1, 2), dim=-1)[0], dim=0)
        new_pos.append(0.5 * (ppos[edges[:, 0]] + ppos[edges[:, 1]]))
        new_real.append(th.ones((edges.shape[0],), device=DEVICE))

        self.ppos = th.cat([ppos] + new_pos, dim=0).detach().clone()
        self.preal = th.cat([preal] + new_real, dim=0).detach().clone()
        self.dtfaces = self._extract_faces_minball(self.ppos, self.preal == 1.0)
        self.logger.info(f"[subdiv] {ppos.shape[0]} -> {self.ppos.shape[0]} points, "
                         f"{self.dtfaces.shape[0]} faces.")

    # ------------------------------------------------------------------ #
    def _set_lr(self, optimizer, base_lr, schedule, step, num_steps):
        if schedule == "linear":
            lr = base_lr * (1.0 - step / num_steps)
        elif schedule == "exp":
            lr = np.exp(np.log(base_lr) + (np.log(MIN_LR) - np.log(base_lr)) * (step / num_steps))
        else:
            lr = base_lr
        lr = max(lr, MIN_LR)
        for g in optimizer.param_groups:
            g['lr'] = lr
        return lr


# ---------------------------------------------------------------------------- #
# Input handling                                                               #
# ---------------------------------------------------------------------------- #
def rescale_point_cloud_to_domain(pc: th.Tensor, margin: float = 0.9):
    center = pc.mean(dim=0, keepdim=True)
    pc = pc - center
    pc = (pc / th.norm(pc, dim=-1).max()) * (margin * DOMAIN)
    return pc


def estimate_density(pc: th.Tensor, sample_size: int = 50000):
    '''Mean nearest-neighbour distance of the input cloud (App. 8.2.1).'''
    with th.no_grad():
        if pc.shape[0] > sample_size:
            idx = th.randperm(pc.shape[0], device=pc.device)[:sample_size]
            src = pc[idx]
        else:
            src = pc
        nn_dist = run_knn(src, pc, 2)[1][:, 1]
        return nn_dist.mean().item()


def voxel_downsample(pc: th.Tensor, cell_size: float):
    '''One representative point per occupied voxel.'''
    key = th.floor((pc - (-DOMAIN)) / cell_size).long()
    key_pos = th.cat([key, pc], dim=-1)
    key_pos = th.unique(key_pos, dim=0)
    _, cnt = th.unique(key_pos[:, :3], dim=0, return_counts=True)
    return key_pos[th.cumsum(cnt, dim=0) - 1][:, 3:]


def load_input_point_cloud(path: str, auto_rescale: bool):
    if path.endswith(".npy"):
        pc = np.load(path)
    elif path.endswith((".ply", ".obj", ".xyz", ".pcd")):
        loaded = trimesh.load(path, process=False)
        pc = np.array(loaded.vertices) if hasattr(loaded, "vertices") else np.array(loaded)
    else:
        raise ValueError(f"Unsupported point cloud file type: {path}")

    pc = th.tensor(pc[:, :3], device=DEVICE, dtype=th.float32)
    if auto_rescale:
        pc = rescale_point_cloud_to_domain(pc)
    if th.max(th.abs(pc)) >= DOMAIN:
        raise ValueError("Input point cloud must lie within [-DOMAIN, DOMAIN]; pass --auto-rescale.")
    return pc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="exp/config/d3/pcrecon_new.yaml")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--input-path", type=str, default="input/3d/pcrecon/example.npy")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="where to save results/logs (overrides 'log_dir' in the config)")
    parser.add_argument("--no-log-time", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--auto-rescale", action="store_true")
    parser.add_argument("--minball-chunk-size", type=int, default=None)
    parser.add_argument("--use-mlflow", action="store_true")
    parser.add_argument("--mlflow-experiment", type=str, default="pc_recon_new")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)

    DEVICE = settings["device"]
    settings["args"]["seed"] = args.seed
    if args.render:
        RENDERING = True
    if args.minball_chunk_size is not None:
        MINBALL_CHUNK_SIZE = args.minball_chunk_size

    logdir = args.output_dir if args.output_dir is not None else settings["log_dir"]
    if not args.no_log_time:
        logdir = logdir + time.strftime("/%Y_%m_%d_%H_%M_%S")
    logdir = setup_logdir(logdir)
    logger = get_logger("pc_recon_new", os.path.join(logdir, "run.log"))
    with open(os.path.join(logdir, "config.yaml"), "w") as f:
        yaml.dump(settings, f)
    th.random.manual_seed(args.seed)

    if args.use_mlflow:
        init_mlflow_run(experiment_name=args.mlflow_experiment,
                        run_name=os.path.basename(logdir.rstrip("/")), settings=settings)

    a = edict(settings["args"])
    try:
        gt_pc = load_input_point_cloud(args.input_path, args.auto_rescale)
    except ValueError as e:
        logger.exception(str(e))
        sys.exit(1)

    density = estimate_density(gt_pc)
    gt_pc = voxel_downsample(gt_pc, a.init_args["downsample_density_scale"] * density)
    logger.info(f"Input cloud: density {density:.5f}, {gt_pc.shape[0]} points after down-sampling.")

    our_sample_interval = 0.5 * a.init_args["grid_size_density_scale"] * density

    optimizer = PCReconNew(
        logdir, logger,
        gt_pc, density, our_sample_interval,
        a.init_args, edict(a.lr), int(a.num_epochs),
        edict(a.init_preal), edict(a.optimize_ppos), edict(a.optimize_freal),
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
