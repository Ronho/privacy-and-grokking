import tempfile
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mlflow

from privacy_and_grokking.utils import Logger, setup_mlflow
from privacy_and_grokking.visualize.mlflow_data import RunData, load_run_data
from privacy_and_grokking.visualize.superplot import (
    plot_per_run_roc_auc,
    plot_per_run_training,
    plot_superplot,
)
from privacy_and_grokking.visualize.tsne import make_tsne_video, plot_tsne, plot_tsne_classes

matplotlib.use("Agg")


def _save_figure_to_mlflow(
    fig: plt.Figure,
    filename: str,
    *,
    run_id: str,
    dpi: int = 150,
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / filename
        fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifact(str(path), artifact_path="visualizations")


def _generate_per_run_plots(rd: RunData, *, tsne_video: bool = False) -> None:
    fig = plot_per_run_training(rd)
    _save_figure_to_mlflow(fig, "training_curves.png", run_id=rd.run_id)

    fig = plot_per_run_roc_auc(rd)
    _save_figure_to_mlflow(fig, "mia_roc_auc_evolution.png", run_id=rd.run_id)

    if rd.train_activations is not None and rd.test_activations is not None:
        fig = plot_tsne(
            rd.train_activations,
            rd.test_activations,
            title=f"t-SNE – {rd.config.full_name}",
        )
        _save_figure_to_mlflow(fig, "tsne_activations.png", run_id=rd.run_id)

        if rd.train_labels is not None and rd.test_labels is not None:
            fig = plot_tsne_classes(
                rd.train_activations,
                rd.test_activations,
                rd.train_labels,
                rd.test_labels,
                title=f"t-SNE by Class – {rd.config.full_name}",
            )
            _save_figure_to_mlflow(fig, "tsne_classes.png", run_id=rd.run_id)

    if tsne_video and rd.all_step_activations and len(rd.all_step_activations) > 1:
        with tempfile.TemporaryDirectory() as tmpdir:
            mp4_path = Path(tmpdir) / "tsne_evolution.mp4"
            make_tsne_video(
                rd.all_step_activations,
                mp4_path,
                title_prefix=f"t-SNE – {rd.config.full_name}",
            )
            with mlflow.start_run(run_id=rd.run_id):
                mlflow.log_artifact(str(mp4_path), artifact_path="visualizations")


def _generate_superplot(runs: list[RunData]) -> None:
    variants = [
        ("superplot_log.png", dict(log_scale=True, x_log_scale=True)),
        ("superplot_linear.png", dict(log_scale=False, x_log_scale=True)),
        ("superplot_log_y_linear_x.png", dict(log_scale=True, x_log_scale=False)),
    ]
    for filename, kwargs in variants:
        fig = plot_superplot(runs, **kwargs)
        for rd in runs:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / filename
                fig.savefig(str(path), dpi=150, bbox_inches="tight")
                with mlflow.start_run(run_id=rd.run_id):
                    mlflow.log_artifact(str(path), artifact_path="visualizations")
        plt.close(fig)


def visualization_handler(exp_name: str, run_ids: list[str], *, tsne_video: bool = False) -> None:
    import os

    os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"
    setup_mlflow(exp_name)

    with Logger() as logger:
        logger.info(
            "Starting visualization.",
            run_ids=run_ids,
            tsne_video=tsne_video,
        )

        runs: list[RunData] = []
        for run_id in run_ids:
            logger.info("Loading run data.", run_id=run_id)
            rd = load_run_data(run_id, load_all_activations=tsne_video)
            runs.append(rd)

        for rd in runs:
            logger.info("Generating per-run plots.", run_id=rd.run_id)
            _generate_per_run_plots(rd, tsne_video=tsne_video)

        if runs:
            logger.info("Generating superplot.", n_runs=len(runs))
            _generate_superplot(runs)

        logger.info("Visualization complete.")
