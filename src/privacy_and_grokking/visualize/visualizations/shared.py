from matplotlib import pyplot as plt

from privacy_and_grokking.utils import Logger

STEP_LABEL = "Optimization Step"

LAYER_COLORS = [
    "tab:blue",
    "tab:red",
    "tab:green",
    "tab:orange",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:cyan",
]
TOTAL_COLOR = "black"

MIA_BASE_NICE_NAMES = {
    "true_class_prob": "Prob",
    "true_class_logit": "Logit",
    "ce_loss": "CE Loss",
    "mse_loss": "MSE Loss",
    "correctness": "Correct",
    "mm_ce": "MM CE",
    "mm_mse": "MM MSE",
}

MIA_COLORS = {
    "true_class_prob": "tab:blue",
    "true_class_logit": "tab:orange",
    "ce_loss": "tab:green",
    "mse_loss": "tab:red",
    "correctness": "tab:purple",
    "mm_ce": "tab:brown",
    "mm_mse": "tab:pink",
}


def handle_missing_data(ax: plt.Axes, run_id: str, plot_name: str):
    logger = Logger.get()
    logger.warning(
        f"Some data missing. Skipping {plot_name} plot.",
        extra={"run_id": run_id},
    )
    ax.text(
        0.5,
        0.5,
        "Some data missing.",
        ha="center",
        va="center",
        transform=ax.transAxes,
    )
