from matplotlib import pyplot as plt

from privacy_and_grokking.utils import Logger
from privacy_and_grokking.visualize.handler import DataHandler
from privacy_and_grokking.visualize.visualizations.shared import STEP_LABEL, handle_missing_data, plot_with_band


def curvature_over_steps(ax: plt.Axes, dh: DataHandler):
    logger = Logger.get()
    logger.info("Creating curvature over steps plot.", extra={"run_id": dh.run_id})

    trace = dh.get_metric_history("eval/curvature/hessian_trace")
    top_eig = dh.get_metric_history("eval/curvature/top_eigenvalue")

    if not trace["steps"] or not top_eig["steps"]:
        handle_missing_data(ax, dh.run_id, "curvature over steps")
        return

    ax2 = ax.twinx()

    if trace["steps"]:
        plot_with_band(ax, trace, color="tab:blue", label="tr(H) (Hutchinson)", linewidth=1.5)
    if top_eig["steps"]:
        plot_with_band(
            ax2,
            top_eig,
            color="tab:orange",
            label="λ_max(H) (power iter.)",
            linewidth=1.5,
            linestyle="--",
        )

    ax.set_xlabel(STEP_LABEL)
    ax.set_ylabel("Hessian Trace  tr(H)", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax2.set_ylabel("Top Eigenvalue  λ_max(H)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="best")

    logger.info("Created curvature over steps plot.", extra={"run_id": dh.run_id})
