import os
import numpy as np
import trimesh
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

IN_PATH = "data/3d/garden_clean.ply"
OUT_DIR = "data/3d/garden_clean_voxel_mesh_3d"
INDIV_DIR = os.path.join(OUT_DIR, "individual")
os.makedirs(INDIV_DIR, exist_ok=True)

N_BINS = 10           # subdivisions per axis -> N_BINS^3 voxels
MIN_POINTS = 15       # need at least this many points in a voxel to attempt reconstruction

pc = trimesh.load(IN_PATH)
pts = pc.vertices
colors = pc.colors[:, :3] if pc.colors is not None else None
n_pts = len(pts)
print(f"Loaded {n_pts} points")

xyz_min = pts.min(axis=0)
xyz_max = pts.max(axis=0)
edges = [np.linspace(xyz_min[d], xyz_max[d], N_BINS + 1) for d in range(3)]

bin_idx = np.zeros((n_pts, 3), dtype=int)
for d in range(3):
    idx = np.searchsorted(edges[d], pts[:, d], side="right") - 1
    idx = np.clip(idx, 0, N_BINS - 1)
    bin_idx[:, d] = idx

all_mesh_verts = []
all_mesh_tris = []
vert_offset = 0
voxel_colors_for_combined = []

n_attempted = 0
n_reconstructed = 0
n_skipped_sparse = 0
n_failed = 0
total_triangles = 0

rng = np.random.default_rng(0)

for i in range(N_BINS):
    for j in range(N_BINS):
        for k in range(N_BINS):
            mask = (bin_idx[:, 0] == i) & (bin_idx[:, 1] == j) & (bin_idx[:, 2] == k)
            count = int(mask.sum())
            if count == 0:
                continue

            if count < MIN_POINTS:
                n_skipped_sparse += 1
                continue

            n_attempted += 1
            voxel_pts = pts[mask]
            voxel_colors = colors[mask] if colors is not None else None

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(voxel_pts)
            if voxel_colors is not None:
                pcd.colors = o3d.utility.Vector3dVector(voxel_colors / 255.0)

            try:
                dists = pcd.compute_nearest_neighbor_distance()
                avg_dist = np.mean(dists)
                if avg_dist <= 0 or not np.isfinite(avg_dist):
                    raise ValueError("degenerate nearest-neighbor distance")

                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=avg_dist * 4, max_nn=30)
                )
                pcd.orient_normals_consistent_tangent_plane(k=min(15, count - 1))

                radii = [avg_dist * f for f in (1.5, 2.0, 3.0, 4.0)]
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                    pcd, o3d.utility.DoubleVector(radii)
                )
                mesh.remove_duplicated_vertices()
                mesh.remove_degenerate_triangles()
                mesh.remove_duplicated_triangles()
                mesh.remove_unreferenced_vertices()

                n_tris = len(mesh.triangles)
                if n_tris == 0:
                    raise ValueError("ball pivoting produced no triangles")

            except Exception as e:
                n_failed += 1
                print(f"  voxel ({i},{j},{k}): FAILED ({count} pts) - {e}")
                continue

            n_reconstructed += 1
            total_triangles += n_tris

            voxel_name = f"voxel_{i:02d}_{j:02d}_{k:02d}"
            mesh_path = os.path.join(INDIV_DIR, f"{voxel_name}.ply")
            o3d.io.write_triangle_mesh(mesh_path, mesh)

            mv = np.asarray(mesh.vertices)
            mt = np.asarray(mesh.triangles)
            all_mesh_verts.append(mv)
            all_mesh_tris.append(mt + vert_offset)
            vert_offset += len(mv)
            voxel_color = rng.uniform(0.2, 0.9, 3)
            voxel_colors_for_combined.append(np.tile(voxel_color, (len(mt), 1)))

            print(f"  voxel ({i},{j},{k}): {count} pts -> {n_tris} triangles")

print()
print(f"Attempted: {n_attempted}, reconstructed: {n_reconstructed}, "
      f"failed: {n_failed}, skipped (<{MIN_POINTS} pts): {n_skipped_sparse}")
print(f"Total triangles across all voxels: {total_triangles}")

# ============================================================
# combined mesh export + visualization
# ============================================================
if all_mesh_verts:
    combined_verts = np.concatenate(all_mesh_verts, axis=0)
    combined_tris = np.concatenate(all_mesh_tris, axis=0)
    combined_face_colors = np.concatenate(voxel_colors_for_combined, axis=0)

    combined_mesh = trimesh.Trimesh(vertices=combined_verts, faces=combined_tris, process=False)
    combined_path = os.path.join(OUT_DIR, "garden_clean_voxel_combined_mesh.ply")
    combined_mesh.export(combined_path)
    print(f"Saved combined mesh to {combined_path}")

    def set_equal_3d_limits(ax):
        center = (xyz_max + xyz_min) / 2
        radius = (xyz_max - xyz_min).max() / 2
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)

    fig = plt.figure(figsize=(14, 7))

    ax_pts = fig.add_subplot(1, 2, 1, projection="3d")
    ax_pts.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1.5, c=(colors / 255.0 if colors is not None else "tab:blue"),
                   alpha=0.6, depthshade=False)
    set_equal_3d_limits(ax_pts)
    ax_pts.view_init(elev=20, azim=-50)
    ax_pts.set_xlabel("X")
    ax_pts.set_ylabel("Y")
    ax_pts.set_zlabel("Z")
    ax_pts.set_title(f"Input point cloud\n({n_pts} pts)")

    ax_mesh = fig.add_subplot(1, 2, 2, projection="3d")
    face_verts = combined_verts[combined_tris]
    collection = Poly3DCollection(face_verts, facecolor=combined_face_colors, edgecolor="k", linewidths=0.03, alpha=0.95)
    ax_mesh.add_collection3d(collection)
    set_equal_3d_limits(ax_mesh)
    ax_mesh.view_init(elev=20, azim=-50)
    ax_mesh.set_xlabel("X")
    ax_mesh.set_ylabel("Y")
    ax_mesh.set_zlabel("Z")
    ax_mesh.set_title(
        f"Per-voxel ball-pivoting mesh ({N_BINS}³ grid)\n"
        f"{n_reconstructed} voxels reconstructed, {total_triangles} triangles total"
    )

    fig.tight_layout()
    render_path = os.path.join(OUT_DIR, "garden_clean_voxel_combined_render.png")
    fig.savefig(render_path, dpi=150)
    plt.close(fig)
    print(f"Saved render to {render_path}")
else:
    print("No voxels had enough points to reconstruct a mesh.")
