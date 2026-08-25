import numpy as np
import trimesh

rng = np.random.default_rng(1)

points = []
colors = []


def add(pts, color, jitter=0):
    pts = np.asarray(pts, dtype=float)
    n = len(pts)
    c = np.tile(np.array(color, dtype=float), (n, 1))
    if jitter:
        c = c + rng.normal(0, jitter, c.shape)
    points.append(pts)
    colors.append(np.clip(c, 0, 255))


# ============================================================
# Ground: flat garden lawn, slightly noisy in height and color
# ============================================================
ground_half = 2.2
n_ground = 90
gx, gy = np.meshgrid(
    np.linspace(-ground_half, ground_half, n_ground),
    np.linspace(-ground_half, ground_half, n_ground),
)
gx = gx.ravel() + rng.normal(0, 0.01, n_ground * n_ground)
gy = gy.ravel() + rng.normal(0, 0.01, n_ground * n_ground)
gz = rng.normal(0, 0.006, n_ground * n_ground)  # subtle terrain roughness
ground_pts = np.column_stack([gx, gy, gz])
ground_color = np.array([70, 130, 55])
add(ground_pts, ground_color, jitter=10)


# ============================================================
# Small tree: trunk (cylinder) + foliage (blobby canopy)
# ============================================================
tree_pos = np.array([-1.2, -1.1, 0.0])
trunk_height = 0.55
trunk_radius = 0.045
n_trunk = 260

trunk_theta = rng.uniform(0, 2 * np.pi, n_trunk)
trunk_r = trunk_radius * np.sqrt(rng.uniform(0.85, 1.0, n_trunk))
trunk_z = rng.uniform(0, trunk_height, n_trunk)
trunk_pts = np.column_stack(
    [
        tree_pos[0] + trunk_r * np.cos(trunk_theta),
        tree_pos[1] + trunk_r * np.sin(trunk_theta),
        tree_pos[2] + trunk_z,
    ]
)
trunk_color = np.array([90, 60, 35])
add(trunk_pts, trunk_color, jitter=8)

# canopy: several overlapping jittered spheres for a "blobby" foliage look
canopy_center = tree_pos + np.array([0.0, 0.0, trunk_height + 0.32])
n_blobs = 7
blob_offsets = rng.normal(0, 0.16, (n_blobs, 3))
blob_offsets[:, 2] *= 0.6
blob_radii = rng.uniform(0.22, 0.32, n_blobs)
n_per_blob = 260

canopy_pts_list = []
for center_off, radius in zip(blob_offsets, blob_radii):
    center = canopy_center + center_off
    # sample inside a sphere: random direction * radius^(1/3) for uniform volume density,
    # but bias towards the shell for a leafier silhouette
    dirs = rng.normal(0, 1, (n_per_blob, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    shell_bias = rng.uniform(0.7, 1.0, n_per_blob) ** 0.5
    pts = center + dirs * (radius * shell_bias[:, None])
    canopy_pts_list.append(pts)
canopy_pts = np.concatenate(canopy_pts_list, axis=0)
canopy_color = np.array([55, 110, 40])
add(canopy_pts, canopy_color, jitter=14)


# ============================================================
# Table: flat top + 4 legs
# ============================================================
table_pos = np.array([1.1, 0.9, 0.0])
table_top_h = 0.5
table_w, table_d = 0.9, 0.6
top_thickness = 0.02
leg_radius = 0.025
leg_inset = 0.06

n_top = 900
tx = rng.uniform(-table_w / 2, table_w / 2, n_top)
ty = rng.uniform(-table_d / 2, table_d / 2, n_top)
tz = table_top_h + rng.uniform(-top_thickness / 2, top_thickness / 2, n_top)
table_top_pts = np.column_stack([table_pos[0] + tx, table_pos[1] + ty, table_pos[2] + tz])
table_color = np.array([120, 80, 45])
add(table_top_pts, table_color, jitter=8)

leg_xy_signs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
n_per_leg = 140
for sx, sy in leg_xy_signs:
    leg_center_x = table_pos[0] + sx * (table_w / 2 - leg_inset)
    leg_center_y = table_pos[1] + sy * (table_d / 2 - leg_inset)
    theta = rng.uniform(0, 2 * np.pi, n_per_leg)
    r = leg_radius * np.sqrt(rng.uniform(0.8, 1.0, n_per_leg))
    z = rng.uniform(0, table_top_h - top_thickness / 2, n_per_leg)
    leg_pts = np.column_stack(
        [
            leg_center_x + r * np.cos(theta),
            leg_center_y + r * np.sin(theta),
            table_pos[2] + z,
        ]
    )
    add(leg_pts, table_color * 0.9, jitter=6)


# ============================================================
# assemble + export
# ============================================================
all_points = np.concatenate(points, axis=0)
all_colors = np.clip(np.concatenate(colors, axis=0), 0, 255).astype(np.uint8)

cloud = trimesh.points.PointCloud(vertices=all_points, colors=all_colors)
out_path = "data/3d/garden_scene_pointcloud.ply"
cloud.export(out_path)
print(f"Saved {len(all_points)} points to {out_path}")
print("Bounds:", all_points.min(axis=0), all_points.max(axis=0))
