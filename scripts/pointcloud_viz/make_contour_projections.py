import numpy as np
import trimesh
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

pc = trimesh.load("data/3d/umbrella_pointcloud.ply")
pts = pc.vertices
x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

grid_res = 300


def contour_projection(u, v, w, u_label, v_label, w_label, out_path):
    """Project points onto (u, v) plane, contour-color by w."""
    ui = np.linspace(u.min(), u.max(), grid_res)
    vi = np.linspace(v.min(), v.max(), grid_res)
    Ui, Vi = np.meshgrid(ui, vi)

    Wi = griddata((u, v), w, (Ui, Vi), method="linear")
    Wi_nearest = griddata((u, v), w, (Ui, Vi), method="nearest")
    Wi = np.where(np.isnan(Wi), Wi_nearest, Wi)

    # mask grid cells far from any actual point (avoid extrapolating over empty space)
    from scipy.spatial import cKDTree
    tree = cKDTree(np.column_stack([u, v]))
    dist, _ = tree.query(np.column_stack([Ui.ravel(), Vi.ravel()]))
    mask = dist.reshape(Ui.shape) > 0.06
    Wi_masked = np.ma.array(Wi, mask=mask)

    fig, ax = plt.subplots(figsize=(7, 7))
    cf = ax.contourf(Ui, Vi, Wi_masked, levels=25, cmap="turbo")
    ax.contour(Ui, Vi, Wi_masked, levels=25, colors="k", linewidths=0.2, alpha=0.3)
    ax.scatter(u, v, s=1, c="black", alpha=0.15)
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


# 1) XY plane, colored by Z
contour_projection(x, y, z, "X", "Y", "Z", "data/3d/umbrella_heatmap_xy_by_z.png")

# 2) XZ plane, colored by Y
contour_projection(x, z, y, "X", "Z", "Y", "data/3d/umbrella_heatmap_xz_by_y.png")

# 3) YZ plane, colored by X
contour_projection(y, z, x, "Y", "Z", "X", "data/3d/umbrella_heatmap_yz_by_x.png")
