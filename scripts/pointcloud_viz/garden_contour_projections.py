import numpy as np
import trimesh
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

pc = trimesh.load("data/3d/garden_scene_pointcloud.ply")
pts = pc.vertices
x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

grid_res = 300
out_dir = "data/3d/garden_scene_heatmaps"


def contour_projection(u, v, w, u_label, v_label, w_label, out_path, mask_dist=0.08):
    """Project points onto (u, v) plane, contour-color by w."""
    ui = np.linspace(u.min(), u.max(), grid_res)
    vi = np.linspace(v.min(), v.max(), grid_res)
    Ui, Vi = np.meshgrid(ui, vi)

    Wi = griddata((u, v), w, (Ui, Vi), method="linear")
    Wi_nearest = griddata((u, v), w, (Ui, Vi), method="nearest")
    Wi = np.where(np.isnan(Wi), Wi_nearest, Wi)

    tree = cKDTree(np.column_stack([u, v]))
    dist, _ = tree.query(np.column_stack([Ui.ravel(), Vi.ravel()]))
    mask = dist.reshape(Ui.shape) > mask_dist
    Wi_masked = np.ma.array(Wi, mask=mask)

    fig, ax = plt.subplots(figsize=(8, 8))
    cf = ax.contourf(Ui, Vi, Wi_masked, levels=25, cmap="turbo")
    ax.contour(Ui, Vi, Wi_masked, levels=25, colors="k", linewidths=0.2, alpha=0.3)
    ax.scatter(u, v, s=1, c="black", alpha=0.1)
    ax.set_xlabel(u_label)
    ax.set_ylabel(v_label)
    ax.set_title(f"{w_label} heatmap ({u_label}{v_label} plane projection)")
    ax.set_aspect("equal")
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(w_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# 1) XY plane (top-down over the garden), colored by Z (height: ground/tree/table)
contour_projection(x, y, z, "X", "Y", "Z", f"{out_dir}/garden_heatmap_xy_by_z.png")

# 2) XZ plane (front view), colored by Y (depth)
contour_projection(x, z, y, "X", "Z", "Y", f"{out_dir}/garden_heatmap_xz_by_y.png")

# 3) YZ plane (side view), colored by X
contour_projection(y, z, x, "Y", "Z", "X", f"{out_dir}/garden_heatmap_yz_by_x.png")
