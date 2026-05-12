from matplotlib import pyplot as plt

from privacy_and_grokking.utils import Logger
from privacy_and_grokking.visualize.handler import DataHandler
from privacy_and_grokking.visualize.visualizations.shared import (
    LAYER_COLORS,
    STEP_LABEL,
    TOTAL_COLOR,
    handle_missing_data,
    plot_with_band,
)


def loss_components_over_steps(ax: plt.Axes, dh: DataHandler):
    """Plot every component contributing to the training loss.

    Shows the task loss, each registered regularizer, and the total loss on
    the same axes so it's easy to see which component dominates and when.
    """
    logger = Logger.get()
    logger.info("Creating loss components over steps plot.", extra={"run_id": dh.run_id})

    task_loss = dh.get_metric_history("train/task_loss")
    total_loss = dh.get_metric_history("train/total_loss")
    regularizer_keys = dh.discover_keys("train/regularizer/")

    if not task_loss["steps"] and not total_loss["steps"] and not regularizer_keys:
        handle_missing_data(ax, dh.run_id, "loss components over steps")
        return

    ax.set_xlabel(STEP_LABEL)
    ax.set_ylabel("Loss Component (per-batch, training)")
    ax.set_yscale("symlog", linthresh=1e-6)

    if task_loss["steps"]:
        plot_with_band(
            ax,
            task_loss,
            color="tab:blue",
            label="Task loss",
            linewidth=1.2,
            alpha=0.9,
        )

    for idx, key in enumerate(regularizer_keys):
        name = key[len("train/regularizer/") :]
        data = dh.get_metric_history(key)
        if not data["steps"]:
            continue
        color = LAYER_COLORS[(idx + 2) % len(LAYER_COLORS)]
        plot_with_band(
            ax,
            data,
            color=color,
            label=f"Reg: {name}",
            linewidth=1.2,
            linestyle="--",
            alpha=0.9,
        )

    if total_loss["steps"]:
        plot_with_band(
            ax,
            total_loss,
            color=TOTAL_COLOR,
            label="Total loss",
            linewidth=1.8,
        )

    ax.legend(loc="best", fontsize=8)

    logger.info("Created loss components over steps plot.", extra={"run_id": dh.run_id})
