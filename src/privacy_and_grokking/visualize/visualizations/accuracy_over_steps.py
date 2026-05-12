from matplotlib import pyplot as plt

from privacy_and_grokking.utils import Logger
from privacy_and_grokking.visualize.handler import DataHandler
from privacy_and_grokking.visualize.visualizations.shared import (
    STEP_LABEL,
    handle_missing_data,
    plot_with_band,
)


def accuracy_over_steps(ax: plt.Axes, dh: DataHandler):
    logger = Logger.get()
    logger.info("Creating accuracy over steps plot.", extra={"run_id": dh.run_id})

    train = dh.get_metric_history("eval/train/accuracy")
    test = dh.get_metric_history("eval/test/accuracy")
    gap = dh.get_metric_history("eval/generalization_gap")

    if not train["steps"] or not test["steps"]:
        handle_missing_data(ax, dh.run_id, "accuracy over steps")
        return

    ax.set_xlabel(STEP_LABEL)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    plot_with_band(ax, train, color="tab:blue", label="Train")
    plot_with_band(ax, test, color="tab:red", label="Test")

    if gap["steps"]:
        ax2 = ax.twinx()
        ax2.set_ylabel("Generalization Gap (Train − Test)")
        plot_with_band(
            ax2,
            gap,
            color="tab:green",
            label="Gen. Gap",
            linestyle="--",
            linewidth=1.5,
        )
        ax2.axhline(0, color="tab:green", linewidth=0.5, alpha=0.4)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="best")
    else:
        ax.legend(loc="best")

    logger.info("Created accuracy over steps plot.", extra={"run_id": dh.run_id})
