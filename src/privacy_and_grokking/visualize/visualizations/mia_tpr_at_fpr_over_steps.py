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


def mia_tpr_at_fpr_over_steps(ax: plt.Axes, dh: DataHandler, fpr_pct: int = 5):
    logger = Logger.get()
    logger.info(
        f"Creating MIA TPR@FPR={fpr_pct}% over steps plot.",
        extra={"run_id": dh.run_id},
    )

    prefix = "eval/attack/"
    suffix = f"/tpr-at-fpr/{fpr_pct}"
    tpr_keys = [k for k in dh.discover_keys(prefix) if k.endswith(suffix)]

    if not tpr_keys:
        handle_missing_data(ax, dh.run_id, f"MIA TPR@FPR={fpr_pct}% over steps")
        return

    for key in tpr_keys:
        data = dh.get_metric_history(key)
        if not data["steps"]:
            continue
        base = key[len(prefix) : -len(suffix)]
        label = MIA_BASE_NICE_NAMES.get(base, base)
        color = MIA_COLORS.get(base, "tab:gray")
        plot_with_band(ax, data, color=color, label=label, linewidth=1.5)

    random_baseline = fpr_pct / 100
    ax.axhline(
        random_baseline,
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label=f"Random ({random_baseline:.2f})",
    )
    ax.set_xlabel(STEP_LABEL)
    ax.set_ylabel(f"TPR @ FPR={fpr_pct}%")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="best")

    logger.info(
        f"Created MIA TPR@FPR={fpr_pct}% over steps plot.",
        extra={"run_id": dh.run_id},
    )
