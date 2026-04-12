from matplotlib import pyplot as plt

from privacy_and_grokking.utils import Logger
from privacy_and_grokking.visualize.handler import DataHandler
from privacy_and_grokking.visualize.visualizations.shared import (
    MIA_BASE_NICE_NAMES,
    MIA_COLORS,
    STEP_LABEL,
    handle_missing_data,
)


def mia_auc_over_steps(ax: plt.Axes, dh: DataHandler):
    logger = Logger.get()
    logger.info("Creating MIA AUC over steps plot.", extra={"run_id": dh.run_id})

    prefix = "eval/attack/"
    suffix = "/auc"
    auc_keys = [k for k in dh.discover_keys(prefix) if k.endswith(suffix)]

    if not auc_keys:
        handle_missing_data(ax, dh.run_id, "MIA AUC over steps")
        return

    for key in auc_keys:
        data = dh.get_metric_history(key)
        if not data["steps"]:
            continue
        base = key[len(prefix): -len(suffix)]
        label = MIA_BASE_NICE_NAMES.get(base, base)
        color = MIA_COLORS.get(base, "tab:gray")
        ax.plot(data["steps"], data["values"], label=label, color=color, linewidth=1.5)

    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.5, label="Random (0.5)")
    ax.set_xlabel(STEP_LABEL)
    ax.set_ylabel("AUC")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="best")

    logger.info("Created MIA AUC over steps plot.", extra={"run_id": dh.run_id})
