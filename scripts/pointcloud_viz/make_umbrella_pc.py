import numpy as np
import trimesh

rng = np.random.default_rng(0)

points = []
colors = []

# --- Canopy: dome made of N triangular panels (like a real umbrella) ---
n_ribs = 8
canopy_radius = 1.0
canopy_height = 0.35  # dome sag height
rib_tip_drop = 0.12   # ribs curve down slightly at the edge

theta_edges = np.linspace(0, 2 * np.pi, n_ribs, endpoint=False)

def canopy_point(r, theta):
    # radial profile: dome rising then ribs bending down near edge
    z = canopy_height * (1 - (r / canopy_radius) ** 2)
    z -= rib_tip_drop * (r / canopy_radius) ** 4
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y, z

canopy_color = np.array([200, 30, 30])

for i in range(n_ribs):
    t0 = theta_edges[i]
    t1 = t0 + 2 * np.pi / n_ribs
    # sample a panel between two ribs, slightly bulging outward (scalloped edge)
    n_r = 25
    n_a = 25
    for r in np.linspace(0.05, canopy_radius, n_r):
        for a in np.linspace(0, 1, n_a):
            theta = t0 + (t1 - t0) * a
            # scalloped edge: panel bulges out slightly at mid-angle, dips at rib seams
            scallop = 0.03 * np.sin(a * np.pi) * (r / canopy_radius) ** 2
            x, y, z = canopy_point(r - scallop, theta)
            points.append((x, y, z))
            jitter = rng.normal(0, 4, 3)
            colors.append(np.clip(canopy_color + jitter, 0, 255))

# --- Ribs themselves (slightly darker, along each seam) ---
rib_color = np.array([140, 15, 15])
for i in range(n_ribs):
    theta = theta_edges[i]
    for r in np.linspace(0.0, canopy_radius, 40):
        x, y, z = canopy_point(r, theta)
        points.append((x, y, z))
        colors.append(rib_color)

# --- Ferrule (small tip spike at the very top center) ---
tip_color = np.array([40, 40, 40])
for z in np.linspace(canopy_height, canopy_height + 0.08, 10):
    points.append((0.0, 0.0, z))
    colors.append(tip_color)

# --- Pole (shaft going down from canopy center to handle) ---
pole_color = np.array([50, 50, 55])
pole_top = canopy_height - 0.05
pole_bottom = -1.4
n_pole = 200
for z in np.linspace(pole_bottom, pole_top, n_pole):
    theta = rng.uniform(0, 2 * np.pi)
    r = 0.015
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    points.append((x, y, z))
    colors.append(pole_color + rng.normal(0, 3, 3))

# --- Runner / stretchers (small cone slightly below canopy center where ribs would attach) ---
runner_color = np.array([70, 70, 75])
runner_z = canopy_height - 0.15
for i in range(n_ribs):
    theta = theta_edges[i]
    for a in np.linspace(0, 1, 15):
        # stretcher strut from runner up to mid-rib
        r_target, theta_t = canopy_radius * 0.55, theta
        rx, ry, rz = canopy_point(r_target, theta_t)
        x = a * rx
        y = a * ry
        z = runner_z + a * (rz - runner_z)
        points.append((x, y, z))
        colors.append(runner_color)

# --- Handle (curved hook at bottom, like a J-shape) ---
handle_color = np.array([90, 55, 30])
hook_center = np.array([0.15, 0.0, pole_bottom - 0.15])
hook_radius = 0.15
angles = np.linspace(-np.pi * 0.1, np.pi * 1.3, 120)
for ang in angles:
    x = hook_center[0] - hook_radius * np.cos(ang)
    y = 0.0
    z = hook_center[2] + hook_radius * np.sin(ang)
    points.append((x, y, z))
    colors.append(handle_color + rng.normal(0, 3, 3))

points = np.array(points)
colors = np.clip(np.array(colors), 0, 255).astype(np.uint8)

cloud = trimesh.points.PointCloud(vertices=points, colors=colors)

out_path = "data/3d/umbrella_pointcloud.ply"
cloud.export(out_path)
print(f"Saved {len(points)} points to {out_path}")
