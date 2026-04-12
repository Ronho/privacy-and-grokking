import numpy as np
from matplotlib import pyplot as plt

from privacy_and_grokking.utils import Logger
from privacy_and_grokking.visualize.handler import DataHandler
from privacy_and_grokking.visualize.visualizations.shared import handle_missing_data


def training_trajectory(ax: plt.Axes, dh: DataHandler) -> None:
    logger = Logger.get()
    logger.info("Creating training trajectory plot.", extra={"run_id": dh.run_id})

    traj = dh.load_weight_trajectory()

    if len(traj) < 3:
        handle_missing_data(ax, dh.run_id, "training trajectory")
        return

    sorted_steps = sorted(traj.keys())
    weight_matrix = np.stack([traj[s] for s in sorted_steps], axis=0)  # (T, D)

    w_centred = weight_matrix - weight_matrix.mean(axis=0, keepdims=True)

    if np.allclose(w_centred, 0, atol=1e-9):
        handle_missing_data(ax, dh.run_id, "training trajectory")
        return

    u_matrix, singular_values, _ = np.linalg.svd(w_centred, full_matrices=False)
    coords = u_matrix[:, :2] * singular_values[:2]  # (T, 2)

    steps_arr = np.array(sorted_steps, dtype=float)
    norm_steps = (steps_arr - steps_arr.min()) / max(steps_arr.max() - steps_arr.min(), 1.0)

    for i in range(len(coords) - 1):
        dx = coords[i + 1, 0] - coords[i, 0]
        dy = coords[i + 1, 1] - coords[i, 1]
        if dx * dx + dy * dy < 1e-18:
            continue
        ax.annotate(
            "",
            xy=(coords[i + 1, 0], coords[i + 1, 1]),
            xytext=(coords[i, 0], coords[i, 1]),
            arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=0.8, alpha=0.5),
            zorder=2,
        )

    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=norm_steps,
        cmap="viridis",
        s=25,
        zorder=3,
        edgecolors="none",
    )

    ax.scatter(
        [coords[0, 0]],
        [coords[0, 1]],
        color="#22c55e",
        s=100,
        zorder=4,
        marker="o",
        label=f"Start (step {sorted_steps[0]})",
    )
    ax.scatter(
        [coords[-1, 0]],
        [coords[-1, 1]],
        color="#ef4444",
        s=100,
        zorder=4,
        marker="*",
        label=f"End (step {sorted_steps[-1]})",
    )

    var_total = float((singular_values**2).sum())
    var1 = float(singular_values[0] ** 2) / var_total * 100
    var2 = float(singular_values[1] ** 2) / var_total * 100
    ax.set_xlabel(f"PC1 ({var1:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var2:.1f}% var)")
    ax.legend(loc="best", fontsize=7)

    cbar = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label("Training progress")
    cbar.ax.tick_params(labelsize=7)

    logger.info("Created training trajectory plot.", extra={"run_id": dh.run_id})
