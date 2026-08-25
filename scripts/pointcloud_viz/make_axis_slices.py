import os
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

pc = trimesh.load("data/3d/umbrella_pointcloud.ply")
pts = pc.vertices
x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

n_bins = 10
out_root = "data/3d/axis_slices"

axes = {
    "x": dict(coord=x, other1=y, other2=z, other1_label="Y", other2_label="Z", color="tab:red"),
    "y": dict(coord=y, other1=x, other2=z, other1_label="X", other2_label="Z", color="tab:green"),
    "z": dict(coord=z, other1=x, other2=y, other1_label="X", other2_label="Y", color="tab:blue"),
}

# fixed 3D view angles per axis so the "slab" reads clearly in the context panel
view_angles = {
    "x": dict(elev=15, azim=-60),
    "y": dict(elev=15, azim=-60),
    "z": dict(elev=15, azim=-60),
}

xyz_min = pts.min(axis=0)
xyz_max = pts.max(axis=0)


def set_equal_3d_limits(ax):
    center = (xyz_max + xyz_min) / 2
    radius = (xyz_max - xyz_min).max() / 2
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


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

        fig = plt.figure(figsize=(13, 6.5))

        # --- left: 2D projection of just this slab ---
        ax_proj = fig.add_subplot(1, 2, 1)
        ax_proj.scatter(o1[mask], o2[mask], s=3, c=color, alpha=0.8)
        ax_proj.set_xlim(o1_min - pad1, o1_max + pad1)
        ax_proj.set_ylim(o2_min - pad2, o2_max + pad2)
        ax_proj.set_xlabel(o1_label)
        ax_proj.set_ylabel(o2_label)
        ax_proj.set_aspect("equal")
        ax_proj.set_title(
            f"{axis_name.upper()} slice {i+1}/{n_bins} projection\n"
            f"{axis_name}∈[{b_lo:.3f}, {b_hi:.3f}]  ({mask.sum()} pts)"
        )

        # --- right: full 3D point cloud with this slab highlighted ---
        ax_3d = fig.add_subplot(1, 2, 2, projection="3d")
        ax_3d.scatter(
            x[~mask], y[~mask], z[~mask],
            s=2, c="lightgray", alpha=0.25, depthshade=False,
        )
        ax_3d.scatter(
            x[mask], y[mask], z[mask],
            s=6, c=color, alpha=0.9, depthshade=False,
        )
        ax_3d.set_xlabel("X")
        ax_3d.set_ylabel("Y")
        ax_3d.set_zlabel("Z")
        set_equal_3d_limits(ax_3d)
        ax_3d.view_init(**view_angles[axis_name])
        ax_3d.set_title(f"Where this slice sits in the full point cloud\n({axis_name.upper()} slab highlighted)")

        fig.tight_layout(rect=(0, 0, 1, 0.94))
        out_path = os.path.join(out_dir, f"slice_{i:02d}.png")
        fig.savefig(out_path, dpi=130)
        plt.close(fig)

    print(f"Saved {n_bins} slice+context images to {out_dir}")
