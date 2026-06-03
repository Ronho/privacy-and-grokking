from matplotlib import pyplot as plt

from privacy_and_grokking.utils import Logger
from privacy_and_grokking.visualize.handler import DataHandler
from privacy_and_grokking.visualize.visualizations.shared import (
    STEP_LABEL,
    handle_missing_data,
    plot_with_band,
)

NC_METRICS = [
    ("nc/nc0/train", "nc/nc0/test", "NC0"),
    ("nc/rnc1/train", "nc/rnc1/test", "RNC1"),
    ("nc/nc1/train", "nc/nc1/test", "NC1"),
    ("nc/nc2/train", "nc/nc2/test", "NC2"),
    ("nc/nc3/train", "nc/nc3/test", "NC3"),
    ("nc/nc4/train", "nc/nc4/test", "NC4"),
    ("nc/between_class_variance/train", "nc/between_class_variance/test", "Between-class Var"),
    ("nc/within_class_variance/train", "nc/within_class_variance/test", "Within-class Var"),
]


def neural_collapse_over_steps(ax: plt.Axes, dh: DataHandler):
    """Plot all neural collapse metrics over training steps.

    Creates a multi-panel view with:
    - NC1, RNC1 (within-class collapse measures) on the left y-axis
    - NC2 (ETF condition number), NC3 (alignment), NC4 (NCC agreement) on subplots
    - Between-class and within-class variance
    """
    logger = Logger.get()
    logger.info("Creating neural collapse over steps plot.", extra={"run_id": dh.run_id})

    # Try to load at least the train RNC1 to verify data exists
    rnc1_train = dh.get_metric_history("eval/nc/rnc1/train")
    if not rnc1_train["steps"]:
        handle_missing_data(ax, dh.run_id, "neural collapse over steps")
        return

    colors_train = {
        "NC0": "tab:cyan",
        "RNC1": "tab:blue",
        "NC1": "tab:orange",
        "NC2": "tab:green",
        "NC3": "tab:red",
        "NC4": "tab:purple",
        "Between-class Var": "tab:brown",
        "Within-class Var": "tab:pink",
    }

    ax.set_xlabel(STEP_LABEL)
    ax.set_ylabel("Value")
    ax.set_title("Neural Collapse Metrics (Train)")

    for train_key, _test_key, label in NC_METRICS:
        data = dh.get_metric_history(f"eval/{train_key}")
        if data["steps"]:
            plot_with_band(ax, data, color=colors_train[label], label=label)

    ax.legend(loc="best", fontsize="small")
    ax.set_yscale("log")
    logger.info("Created neural collapse over steps plot.", extra={"run_id": dh.run_id})


def neural_collapse_nc1_rnc1(ax: plt.Axes, dh: DataHandler):
    """Plot NC1 over training steps.

    Kept under the existing function name for backward compatibility in visualization keys.
    """
    logger = Logger.get()
    logger.info("Creating NC1 plot.", extra={"run_id": dh.run_id})

    nc1_train = dh.get_metric_history("eval/nc/nc1/train")
    nc1_test = dh.get_metric_history("eval/nc/nc1/test")

    if not nc1_train["steps"] and not nc1_test["steps"]:
        handle_missing_data(ax, dh.run_id, "NC1")
        return

    ax.set_xlabel(STEP_LABEL)
    ax.set_ylabel("NC1")

    if nc1_train["steps"]:
        plot_with_band(ax, nc1_train, color="tab:blue", label="NC1 (train)")
    if nc1_test["steps"]:
        plot_with_band(ax, nc1_test, color="tab:blue", label="NC1 (test)", linestyle="--")

    ax.set_yscale("log")
    ax.legend(loc="best", fontsize="small")
    logger.info("Created NC1 plot.", extra={"run_id": dh.run_id})


def neural_collapse_rnc1(ax: plt.Axes, dh: DataHandler):
    """Plot RNC1 separately with a log x-axis and fixed y-axis scale from 0 to 0.5."""
    logger = Logger.get()
    logger.info("Creating RNC1 plot.", extra={"run_id": dh.run_id})

    rnc1_train = dh.get_metric_history("eval/nc/rnc1/train")
    rnc1_test = dh.get_metric_history("eval/nc/rnc1/test")

    if not rnc1_train["steps"] and not rnc1_test["steps"]:
        handle_missing_data(ax, dh.run_id, "RNC1")
        return

    ax.set_xlabel(STEP_LABEL)
    ax.set_ylabel("RNC1")
    ax.set_xscale("log")
    ax.set_ylim(0, 0.5)

    if rnc1_train["steps"]:
        plot_with_band(ax, rnc1_train, color="tab:orange", label="RNC1 (train)")
    if rnc1_test["steps"]:
        plot_with_band(ax, rnc1_test, color="tab:orange", label="RNC1 (test)", linestyle="--")

    ax.legend(loc="best", fontsize="small")
    logger.info("Created RNC1 plot.", extra={"run_id": dh.run_id})


def neural_collapse_nc2(ax: plt.Axes, dh: DataHandler):
    """Plot NC2 (condition number of class means — ETF structure)."""
    logger = Logger.get()
    logger.info("Creating NC2 plot.", extra={"run_id": dh.run_id})

    nc2_train = dh.get_metric_history("eval/nc/nc2/train")
    nc2_test = dh.get_metric_history("eval/nc/nc2/test")

    if not nc2_train["steps"]:
        handle_missing_data(ax, dh.run_id, "NC2")
        return

    ax.set_xlabel(STEP_LABEL)
    ax.set_ylabel("NC2 (Condition Number)")
    plot_with_band(ax, nc2_train, color="tab:green", label="Train")
    if nc2_test["steps"]:
        plot_with_band(ax, nc2_test, color="tab:green", label="Test", linestyle="--")
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5, label="Ideal (ETF)")
    ax.legend(loc="best", fontsize="small")
    logger.info("Created NC2 plot.", extra={"run_id": dh.run_id})


def neural_collapse_nc3_nc4(ax: plt.Axes, dh: DataHandler):
    """Plot NC3 (weight-mean alignment) and NC4 (NCC agreement)."""
    logger = Logger.get()
    logger.info("Creating NC3/NC4 plot.", extra={"run_id": dh.run_id})

    nc3_train = dh.get_metric_history("eval/nc/nc3/train")
    nc4_train = dh.get_metric_history("eval/nc/nc4/train")
    nc3_test = dh.get_metric_history("eval/nc/nc3/test")
    nc4_test = dh.get_metric_history("eval/nc/nc4/test")

    if not nc3_train["steps"] and not nc4_train["steps"]:
        handle_missing_data(ax, dh.run_id, "NC3/NC4")
        return

    ax.set_xlabel(STEP_LABEL)
    ax.set_ylabel("Value")
    ax.set_ylim(0, 1.05)

    if nc3_train["steps"]:
        plot_with_band(ax, nc3_train, color="tab:red", label="NC3 (train)")
    if nc3_test["steps"]:
        plot_with_band(ax, nc3_test, color="tab:red", label="NC3 (test)", linestyle="--")
    if nc4_train["steps"]:
        plot_with_band(ax, nc4_train, color="tab:purple", label="NC4 (train)")
    if nc4_test["steps"]:
        plot_with_band(ax, nc4_test, color="tab:purple", label="NC4 (test)", linestyle="--")

    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5, label="Perfect collapse")
    ax.legend(loc="best", fontsize="small")
    logger.info("Created NC3/NC4 plot.", extra={"run_id": dh.run_id})


def neural_collapse_variance(ax: plt.Axes, dh: DataHandler):
    """Plot between-class and within-class variance over training."""
    logger = Logger.get()
    logger.info("Creating variance plot.", extra={"run_id": dh.run_id})

    bw_train = dh.get_metric_history("eval/nc/between_class_variance/train")
    ww_train = dh.get_metric_history("eval/nc/within_class_variance/train")
    bw_test = dh.get_metric_history("eval/nc/between_class_variance/test")
    ww_test = dh.get_metric_history("eval/nc/within_class_variance/test")

    if not bw_train["steps"] and not ww_train["steps"]:
        handle_missing_data(ax, dh.run_id, "class variance")
        return

    ax.set_xlabel(STEP_LABEL)
    ax.set_ylabel("Variance (Trace)")

    if bw_train["steps"]:
        plot_with_band(ax, bw_train, color="tab:brown", label="Between-class (train)")
    if bw_test["steps"]:
        plot_with_band(ax, bw_test, color="tab:brown", label="Between-class (test)", linestyle="--")
    if ww_train["steps"]:
        plot_with_band(ax, ww_train, color="tab:pink", label="Within-class (train)")
    if ww_test["steps"]:
        plot_with_band(ax, ww_test, color="tab:pink", label="Within-class (test)", linestyle="--")

    ax.set_yscale("log")
    ax.legend(loc="best", fontsize="small")
    logger.info("Created variance plot.", extra={"run_id": dh.run_id})
