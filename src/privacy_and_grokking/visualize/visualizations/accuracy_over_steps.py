from matplotlib import pyplot as plt

from privacy_and_grokking.utils import Logger
from privacy_and_grokking.visualize.handler import DataHandler
from privacy_and_grokking.visualize.visualizations.shared import STEP_LABEL, handle_missing_data, plot_with_band


def accuracy_over_steps(ax: plt.Axes, dh: DataHandler):
    logger = Logger.get()
    logger.info("Creating accuracy over steps plot.", extra={"run_id": dh.run_id})

    train = dh.get_metric_history("eval/train/accuracy")
    test = dh.get_metric_history("eval/test/accuracy")

    if not train["steps"] or not test["steps"]:
        handle_missing_data(ax, dh.run_id, "accuracy over steps")
        return

    ax.set_xlabel(STEP_LABEL)
    ax.set_ylabel("Accuracy")
    plot_with_band(ax, train, color="tab:blue", label="Train")
    plot_with_band(ax, test, color="tab:red", label="Test")
    ax.legend(loc="best")

    logger.info("Created accuracy over steps plot.", extra={"run_id": dh.run_id})
