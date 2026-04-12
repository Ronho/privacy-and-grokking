from matplotlib import pyplot as plt

from privacy_and_grokking.utils import Logger
from privacy_and_grokking.visualize.handler import DataHandler
from privacy_and_grokking.visualize.visualizations.shared import STEP_LABEL, handle_missing_data

_STAT_COLORS = {
    "norm": "tab:blue",
    "mean": "tab:orange",
    "abs_mean": "tab:green",
}

_STAT_LABELS = {
    "norm": "L2 Norm",
    "mean": "Mean",
    "abs_mean": "Abs Mean",
}

_STATS_ORDERED = ["norm", "mean", "abs_mean"]


def optimizer_internals(ax: plt.Axes, dh: DataHandler, state_key: str) -> None:
    """Plot norm, mean, and abs_mean for one optimizer state key."""
    logger = Logger.get()
    logger.info(
        "Creating optimizer internals plot.",
        extra={"run_id": dh.run_id, "state_key": state_key},
    )

    has_data = False
    for stat in _STATS_ORDERED:
        data = dh.get_metric_history(f"optimizer/{state_key}/{stat}")
        if data["steps"]:
            has_data = True
            ax.plot(
                data["steps"],
                data["values"],
                color=_STAT_COLORS.get(stat, "tab:blue"),
                linewidth=1.5,
                label=_STAT_LABELS.get(stat, stat),
            )

    if not has_data:
        handle_missing_data(ax, dh.run_id, f"optimizer internals ({state_key})")
        return

    ax.set_xlabel(STEP_LABEL)
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=7)

    logger.info(
        "Created optimizer internals plot.",
        extra={"run_id": dh.run_id, "state_key": state_key},
    )
