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
    "mia_prob": "Prob",
    "mia_logit": "Logit",
    "mia_ce_loss": "CE Loss",
    "mia_mse_loss": "MSE Loss",
    "mia_correctness": "Correct",
    "mia_merlin_morgan_ce": "MM CE",
    "mia_merlin_morgan_mse": "MM MSE",
}

MIA_COLORS = {
    "mia_prob": "tab:blue",
    "mia_logit": "tab:orange",
    "mia_ce_loss": "tab:green",
    "mia_mse_loss": "tab:red",
    "mia_correctness": "tab:purple",
    "mia_merlin_morgan_ce": "tab:brown",
    "mia_merlin_morgan_mse": "tab:pink",
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
