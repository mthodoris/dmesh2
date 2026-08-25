import os
import itertools
from collections import Counter

import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Delaunay, cKDTree

ALPHA_FACTOR = 3.0  # max allowed tetrahedron edge, as a multiple of local median NN distance

pc = trimesh.load("data/3d/umbrella_pointcloud.ply")
verts = pc.vertices
n_pts = len(verts)
print(f"Loaded {n_pts} points")

# --- 3D Delaunay tetrahedralization of the whole point cloud ---
tri3d = Delaunay(verts)
tets = tri3d.simplices
print(f"Delaunay produced {len(tets)} tetrahedra")

edge_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
edge_lens = np.stack(
    [np.linalg.norm(verts[tets[:, i]] - verts[tets[:, j]], axis=1) for i, j in edge_pairs],
    axis=1,
)
max_edge = edge_lens.max(axis=1)

tree = cKDTree(verts)
nn_dist, _ = tree.query(verts, k=2)
median_nn = np.median(nn_dist[:, 1])
threshold = median_nn * ALPHA_FACTOR
print(f"Median nearest-neighbor distance: {median_nn:.5f}, alpha threshold: {threshold:.5f}")

keep = max_edge < threshold
kept_tets = tets[keep]
print(f"Kept {len(kept_tets)} / {len(tets)} tetrahedra after alpha filtering")

# --- extract boundary faces of the surviving tetrahedra (faces used by exactly one kept tet) ---
face_count = Counter()
for tet in kept_tets:
    for face in itertools.combinations(tet, 3):
        face_count[tuple(sorted(face))] += 1

boundary_faces = np.array([f for f, c in face_count.items() if c == 1])
print(f"Extracted {len(boundary_faces)} boundary triangles")

mesh = trimesh.Trimesh(vertices=verts, faces=boundary_faces, process=False)
mesh.fix_normals()

out_dir = "data/3d/mesh_3d"
os.makedirs(out_dir, exist_ok=True)
mesh_path = os.path.join(out_dir, "umbrella_alpha_mesh.ply")
mesh.export(mesh_path)
print(f"Saved mesh to {mesh_path}")

# --- visualization: original point cloud vs extracted surface mesh ---
xyz_min, xyz_max = verts.min(axis=0), verts.max(axis=0)


def set_equal_3d_limits(ax):
    center = (xyz_max + xyz_min) / 2
    radius = (xyz_max - xyz_min).max() / 2
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


fig = plt.figure(figsize=(14, 7))

ax_pts = fig.add_subplot(1, 2, 1, projection="3d")
ax_pts.scatter(verts[:, 0], verts[:, 1], verts[:, 2], s=1.5, c="tab:blue", alpha=0.5, depthshade=False)
set_equal_3d_limits(ax_pts)
ax_pts.view_init(elev=15, azim=-60)
ax_pts.set_xlabel("X")
ax_pts.set_ylabel("Y")
ax_pts.set_zlabel("Z")
ax_pts.set_title(f"Input point cloud\n({n_pts} pts)")

ax_mesh = fig.add_subplot(1, 2, 2, projection="3d")
face_verts = verts[boundary_faces]
collection = Poly3DCollection(face_verts, facecolor="tab:orange", edgecolor="k", linewidths=0.05, alpha=0.9)
ax_mesh.add_collection3d(collection)
set_equal_3d_limits(ax_mesh)
ax_mesh.view_init(elev=15, azim=-60)
ax_mesh.set_xlabel("X")
ax_mesh.set_ylabel("Y")
ax_mesh.set_zlabel("Z")
ax_mesh.set_title(f"Alpha-filtered Delaunay surface mesh\n({len(boundary_faces)} triangles)")

fig.tight_layout()
render_path = os.path.join(out_dir, "umbrella_alpha_mesh_render.png")
fig.savefig(render_path, dpi=150)
plt.close(fig)
print(f"Saved render to {render_path}")
