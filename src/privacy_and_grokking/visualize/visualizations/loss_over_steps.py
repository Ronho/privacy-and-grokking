import mlflow
from matplotlib import pyplot as plt

from privacy_and_grokking.config.model import TrainConfig
from privacy_and_grokking.utils import Logger
from privacy_and_grokking.visualize.handler import DataHandler
from privacy_and_grokking.visualize.visualizations.shared import STEP_LABEL, handle_missing_data


def loss_over_steps(ax: plt.Axes, dh: DataHandler):
    logger = Logger.get()
    logger.info("Creating loss over steps plot.", extra={"run_id": dh.run_id})

    try:
        cfg = TrainConfig.model_validate(
            mlflow.artifacts.load_dict(f"runs:/{dh.run_id}/training_config.json")
        )
    except Exception as exc:
        logger.warning(
            "Could not load training config.",
            extra={"run_id": dh.run_id, "error": str(exc)},
        )
        handle_missing_data(ax, dh.run_id, "loss over steps")
        return

    loss_name = cfg.loss.name
    train_mean = dh.get_metric_history(f"eval/test/loss/{loss_name}/mean")
    train_std = dh.get_metric_history(f"eval/test/loss/{loss_name}/std")
    test_mean = dh.get_metric_history(f"eval/test/loss/{loss_name}/mean")
    test_std = dh.get_metric_history(f"eval/test/loss/{loss_name}/std")
    overlap = dh.get_metric_history(f"eval/loss/{loss_name}/overlap")

    if not all([x["steps"] for x in (train_mean, train_std, test_mean, test_std, overlap)]):
        handle_missing_data(ax, dh.run_id, "loss over steps")
        return

    train_steps = train_mean["steps"]
    train_values = train_mean["values"]
    train_std_values = train_std["values"]
    test_steps = test_mean["steps"]
    test_values = test_mean["values"]
    test_std_values = test_std["values"]

    ax.set_xlabel(STEP_LABEL)
    ax.set_ylabel("Loss")

    ax.plot(train_steps, train_values, label="Train (mean)", color="tab:blue")
    ax.fill_between(
        train_steps,
        [m - s for m, s in zip(train_values, train_std_values)],
        [m + s for m, s in zip(train_values, train_std_values)],
        alpha=0.2,
        color="tab:blue",
        label="Train (±std)",
    )

    ax.plot(test_steps, test_values, label="Test (mean)", color="tab:red")
    ax.fill_between(
        test_steps,
        [m - s for m, s in zip(test_values, test_std_values)],
        [m + s for m, s in zip(test_values, test_std_values)],
        alpha=0.2,
        color="tab:red",
        label="Test (±std)",
    )

    ax2 = ax.twinx()
    ax2.set_ylabel("Overlap", color="tab:orange")
    ax2.set_ylim(0, 1)
    ax2.plot(
        overlap["steps"], overlap["values"], label="Overlap", color="tab:orange", linestyle="--"
    )
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="best")

    logger.info("Created loss over steps plot.", extra={"run_id": dh.run_id})
