import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

in_path = "data/3d/garden_clean.ply"
out_dir = "data/3d/garden_clean_mesh_3d"
os.makedirs(out_dir, exist_ok=True)

pcd = o3d.io.read_point_cloud(in_path)
n_pts = len(pcd.points)
print(f"Loaded {n_pts} points")

dists = pcd.compute_nearest_neighbor_distance()
avg_dist = np.mean(dists)
print(f"Average nearest-neighbor distance: {avg_dist:.5f}")

pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=avg_dist * 4, max_nn=30))
pcd.orient_normals_consistent_tangent_plane(k=15)

radii = [avg_dist * f for f in (1.5, 2.0, 3.0, 4.0)]
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd, o3d.utility.DoubleVector(radii)
)
mesh.remove_duplicated_vertices()
mesh.remove_degenerate_triangles()
mesh.remove_duplicated_triangles()
mesh.remove_unreferenced_vertices()

n_tris = len(mesh.triangles)
print(f"Ball pivoting produced {n_tris} triangles (radii={[round(r,5) for r in radii]})")

mesh_path = os.path.join(out_dir, "garden_clean_ball_pivot_mesh.ply")
o3d.io.write_triangle_mesh(mesh_path, mesh)
print(f"Saved mesh to {mesh_path}")

verts = np.asarray(pcd.points)
mesh_verts = np.asarray(mesh.vertices)
mesh_tris = np.asarray(mesh.triangles)

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
ax_pts.view_init(elev=20, azim=-50)
ax_pts.set_xlabel("X")
ax_pts.set_ylabel("Y")
ax_pts.set_zlabel("Z")
ax_pts.set_title(f"Input point cloud\n({n_pts} pts)")

ax_mesh = fig.add_subplot(1, 2, 2, projection="3d")
face_verts = mesh_verts[mesh_tris]
collection = Poly3DCollection(face_verts, facecolor="tab:orange", edgecolor="k", linewidths=0.05, alpha=0.95)
ax_mesh.add_collection3d(collection)
set_equal_3d_limits(ax_mesh)
ax_mesh.view_init(elev=20, azim=-50)
ax_mesh.set_xlabel("X")
ax_mesh.set_ylabel("Y")
ax_mesh.set_zlabel("Z")
ax_mesh.set_title(f"Ball-pivoting surface mesh\n({n_tris} triangles)")

fig.tight_layout()
render_path = os.path.join(out_dir, "garden_clean_ball_pivot_mesh_render.png")
fig.savefig(render_path, dpi=150)
plt.close(fig)
print(f"Saved render to {render_path}")
