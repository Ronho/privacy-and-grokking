from matplotlib import pyplot as plt

from privacy_and_grokking.utils import Logger
from privacy_and_grokking.visualize.handler import DataHandler
from privacy_and_grokking.visualize.visualizations.shared import (
    MIA_BASE_NICE_NAMES,
    MIA_COLORS,
    STEP_LABEL,
    handle_missing_data,
    plot_with_band,
)


def mia_recall_over_steps(ax: plt.Axes, dh: DataHandler):
    """Plot recall (TPR) at the attacker's best operating point over steps.

    The threshold is selected per-step by maximising Youden's J = TPR - FPR,
    which is the standard "best operating point" on the ROC curve. Higher
    values mean the attack identifies more true members at its preferred
    threshold.
    """
    logger = Logger.get()
    logger.info("Creating MIA recall over steps plot.", extra={"run_id": dh.run_id})

    prefix = "eval/attack/"
    suffix = "/recall"
    recall_keys = [
        k for k in dh.discover_keys(prefix) if k.endswith(suffix) and not k.endswith("/recall_fpr")
    ]

    if not recall_keys:
        handle_missing_data(ax, dh.run_id, "MIA recall over steps")
        return

    for key in recall_keys:
        data = dh.get_metric_history(key)
        if not data["steps"]:
            continue
        base = key[len(prefix) : -len(suffix)]
        label = MIA_BASE_NICE_NAMES.get(base, base)
        color = MIA_COLORS.get(base, "tab:gray")
        plot_with_band(ax, data, color=color, label=label, linewidth=1.5)

    ax.set_xlabel(STEP_LABEL)
    ax.set_ylabel("Recall @ best Youden's J")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="best")

    logger.info("Created MIA recall over steps plot.", extra={"run_id": dh.run_id})
