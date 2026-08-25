import os
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from scipy.spatial import Delaunay, cKDTree, QhullError

pc = trimesh.load("data/3d/garden_clean.ply")
pts = pc.vertices
x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

n_bins = 10
out_root = "data/3d/garden_clean_axis_slices"
ALPHA_FACTOR = 4.0

axes = {
    "x": dict(coord=x, other1=y, other2=z, other1_label="Y", other2_label="Z", color="tab:red"),
    "y": dict(coord=y, other1=x, other2=z, other1_label="X", other2_label="Z", color="tab:green"),
    "z": dict(coord=z, other1=x, other2=y, other1_label="X", other2_label="Y", color="tab:blue"),
}


def alpha_filtered_triangulation(px, py, alpha_factor=ALPHA_FACTOR):
    """Delaunay triangulate (px, py), then drop triangles whose longest edge
    is much bigger than the local point spacing, so the mesh doesn't bridge
    real gaps in the slice."""
    p = np.column_stack([px, py])
    if len(p) < 4:
        return None, None

    try:
        tri = Delaunay(p)
    except QhullError:
        return None, None

    simplices = tri.simplices
    a, b, c = p[simplices[:, 0]], p[simplices[:, 1]], p[simplices[:, 2]]
    e_ab = np.linalg.norm(a - b, axis=1)
    e_bc = np.linalg.norm(b - c, axis=1)
    e_ca = np.linalg.norm(c - a, axis=1)
    max_edge = np.maximum(np.maximum(e_ab, e_bc), e_ca)

    tree = cKDTree(p)
    nn_dist, _ = tree.query(p, k=2)
    median_nn = np.median(nn_dist[:, 1])
    if median_nn <= 0:
        median_nn = np.median(max_edge) / alpha_factor if len(max_edge) else 1e-6

    threshold = median_nn * alpha_factor
    keep = max_edge < threshold
    if not np.any(keep):
        return None, None

    triang = Triangulation(p[:, 0], p[:, 1], simplices[keep])
    return triang, keep.sum()


for axis_name, info in axes.items():
    coord = info["coord"]
    o1, o2 = info["other1"], info["other2"]
    o1_label, o2_label = info["other1_label"], info["other2_label"]
    color = info["color"]

    out_dir = os.path.join(out_root, f"{axis_name}_slices")
    os.makedirs(out_dir, exist_ok=True)

    lo, hi = coord.min(), coord.max()
    edges = np.linspace(lo, hi, n_bins + 1)

    o1_min, o1_max = o1.min(), o1.max()
    o2_min, o2_max = o2.min(), o2.max()
    pad1 = 0.05 * (o1_max - o1_min if o1_max > o1_min else 1)
    pad2 = 0.05 * (o2_max - o2_min if o2_max > o2_min else 1)

    for i in range(n_bins):
        b_lo, b_hi = edges[i], edges[i + 1]
        mask = (coord >= b_lo) & (coord <= b_hi) if i == n_bins - 1 else (coord >= b_lo) & (coord < b_hi)
        px, py = o1[mask], o2[mask]

        triang, n_tris = alpha_filtered_triangulation(px, py)

        fig, axs = plt.subplots(1, 2, figsize=(13, 6.5))

        axs[0].scatter(px, py, s=3, c=color, alpha=0.8)
        axs[0].set_xlim(o1_min - pad1, o1_max + pad1)
        axs[0].set_ylim(o2_min - pad2, o2_max + pad2)
        axs[0].set_xlabel(o1_label)
        axs[0].set_ylabel(o2_label)
        axs[0].set_aspect("equal")
        axs[0].set_title(
            f"{axis_name.upper()} slice {i+1}/{n_bins} points\n"
            f"{axis_name}∈[{b_lo:.3f}, {b_hi:.3f}]  ({mask.sum()} pts)"
        )

        axs[1].set_xlim(o1_min - pad1, o1_max + pad1)
        axs[1].set_ylim(o2_min - pad2, o2_max + pad2)
        axs[1].set_xlabel(o1_label)
        axs[1].set_ylabel(o2_label)
        axs[1].set_aspect("equal")
        if triang is not None:
            axs[1].triplot(triang, color=color, linewidth=0.4)
            axs[1].scatter(px, py, s=1.5, c="black", alpha=0.3, zorder=3)
            axs[1].set_title(f"Alpha-filtered Delaunay mesh\n({n_tris} triangles)")
        else:
            axs[1].scatter(px, py, s=3, c=color, alpha=0.8)
            axs[1].set_title("Not enough points for a mesh")

        fig.tight_layout(rect=(0, 0, 1, 0.94))
        out_path = os.path.join(out_dir, f"slice_{i:02d}_mesh.png")
        fig.savefig(out_path, dpi=130)
        plt.close(fig)

    print(f"Saved {n_bins} mesh images to {out_dir}")
