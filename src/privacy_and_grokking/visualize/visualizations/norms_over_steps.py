from matplotlib import pyplot as plt

from privacy_and_grokking.utils import Logger
from privacy_and_grokking.visualize.handler import DataHandler
from privacy_and_grokking.visualize.visualizations.shared import (
    LAYER_COLORS,
    STEP_LABEL,
    TOTAL_COLOR,
    plot_with_band,
)


def _plot_norms_over_steps(ax: plt.Axes, dh: DataHandler, prefix: str, ylabel: str):
    all_keys = dh.discover_keys(prefix)
    total_key = f"{prefix}total"
    layer_keys = [k for k in all_keys if k != total_key]

    seen: dict[str, int] = {}
    for key in layer_keys:
        name = key[len(prefix) :]
        base = name.removesuffix(".weight").removesuffix(".bias")
        if base not in seen:
            seen[base] = len(seen)

    for key in layer_keys:
        name = key[len(prefix) :]
        base = name.removesuffix(".weight").removesuffix(".bias")
        color = LAYER_COLORS[seen[base] % len(LAYER_COLORS)]
        linestyle = "--" if name.endswith(".bias") else "-"
        data = dh.get_metric_history(key)
        plot_with_band(ax, data, color=color, label=name, linestyle=linestyle, linewidth=1, alpha=0.8)

    if total_key in all_keys:
        data = dh.get_metric_history(total_key)
        plot_with_band(ax, data, color=TOTAL_COLOR, label="total", linewidth=2)

    ax.set_xlabel(STEP_LABEL)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")


def weight_norms_over_steps(ax: plt.Axes, dh: DataHandler):
    logger = Logger.get()
    logger.info("Creating weight norms over steps plot.", extra={"run_id": dh.run_id})
    _plot_norms_over_steps(ax, dh, prefix="eval/weight_norm/", ylabel="Weight Norm")
    logger.info("Created weight norms over steps plot.", extra={"run_id": dh.run_id})


def gradient_norms_over_steps(ax: plt.Axes, dh: DataHandler):
    logger = Logger.get()
    logger.info("Creating gradient norms over steps plot.", extra={"run_id": dh.run_id})
    _plot_norms_over_steps(ax, dh, prefix="eval/grad_norm/", ylabel="Gradient Norm")
    logger.info("Created gradient norms over steps plot.", extra={"run_id": dh.run_id})
