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

SINGLE_VIZ_NAMES: frozenset[str] = frozenset(
    {
        "training_curves",
        "mia_roc_auc_evolution",
        "tsne_activations",
        "tsne_classes",
        "tsne_evolution",
    }
)

MULTI_VIZ_NAMES: frozenset[str] = frozenset(
    {
        "superplot_log",
        "superplot_linear",
        "superplot_log_y_linear_x",
    }
)

_SUPERPLOT_VARIANTS: list[tuple[str, str, dict]] = [
    ("superplot_log", "superplot_log.png", {"log_scale": True, "x_log_scale": True}),
    ("superplot_linear", "superplot_linear.png", {"log_scale": False, "x_log_scale": True}),
    (
        "superplot_log_y_linear_x",
        "superplot_log_y_linear_x.png",
        {"log_scale": True, "x_log_scale": False},
    ),
]


def _resolve_active(
    valid: frozenset[str],
    include: list[str] | None,
    exclude: list[str] | None,
) -> set[str]:
    active = set(include) if include is not None else set(valid)
    if exclude:
        active -= set(exclude)
    invalid = active - valid
    if invalid:
        raise ValueError(
            f"Unknown visualization names: {sorted(invalid)!r}. Valid names: {sorted(valid)!r}"
        )
    return active


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


def _generate_per_run_plots(rd: RunData, active: set[str]) -> None:
    if "training_curves" in active:
        fig = plot_per_run_training(rd)
        _save_figure_to_mlflow(fig, "training_curves.png", run_id=rd.run_id)

    if "mia_roc_auc_evolution" in active:
        fig = plot_per_run_roc_auc(rd)
        _save_figure_to_mlflow(fig, "mia_roc_auc_evolution.png", run_id=rd.run_id)

    has_activations = rd.train_activations is not None and rd.test_activations is not None

    if "tsne_activations" in active and has_activations:
        fig = plot_tsne(
            rd.train_activations,
            rd.test_activations,
            title=f"t-SNE – {rd.config.full_name}",
        )
        _save_figure_to_mlflow(fig, "tsne_activations.png", run_id=rd.run_id)

    if (
        "tsne_classes" in active
        and has_activations
        and rd.train_labels is not None
        and rd.test_labels is not None
    ):
        fig = plot_tsne_classes(
            rd.train_activations,
            rd.test_activations,
            rd.train_labels,
            rd.test_labels,
            title=f"t-SNE by Class – {rd.config.full_name}",
        )
        _save_figure_to_mlflow(fig, "tsne_classes.png", run_id=rd.run_id)

    if "tsne_evolution" in active and rd.all_step_activations and len(rd.all_step_activations) > 1:
        with tempfile.TemporaryDirectory() as tmpdir:
            mp4_path = Path(tmpdir) / "tsne_evolution.mp4"
            make_tsne_video(
                rd.all_step_activations,
                mp4_path,
                title_prefix=f"t-SNE – {rd.config.full_name}",
            )
            with mlflow.start_run(run_id=rd.run_id):
                mlflow.log_artifact(str(mp4_path), artifact_path="visualizations")


def _generate_superplot(runs: list[RunData], active: set[str]) -> None:
    for name, filename, kwargs in _SUPERPLOT_VARIANTS:
        if name not in active:
            continue
        fig = plot_superplot(runs, **kwargs)
        for rd in runs:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / filename
                fig.savefig(str(path), dpi=150, bbox_inches="tight")
                with mlflow.start_run(run_id=rd.run_id):
                    mlflow.log_artifact(str(path), artifact_path="visualizations")
        plt.close(fig)


def visualization_single_handler(
    exp_name: str,
    run_id: str,
    *,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> None:
    setup_mlflow(exp_name)

    active = _resolve_active(SINGLE_VIZ_NAMES, include, exclude)

    with Logger() as logger:
        logger.info(
            "Starting single-run visualization.",
            run_id=run_id,
            active=sorted(active),
        )
        rd = load_run_data(run_id, load_all_activations="tsne_evolution" in active)
        logger.info("Generating per-run plots.", run_id=rd.run_id)
        _generate_per_run_plots(rd, active)
        logger.info("Visualization complete.", run_id=rd.run_id)


def visualization_multi_handler(
    exp_name: str,
    run_ids: list[str],
    *,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> None:
    setup_mlflow(exp_name)

    active = _resolve_active(MULTI_VIZ_NAMES, include, exclude)

    with Logger() as logger:
        logger.info(
            "Starting multi-run visualization.",
            run_ids=run_ids,
            active=sorted(active),
        )
        runs: list[RunData] = []
        for rid in run_ids:
            logger.info("Loading run data.", run_id=rid)
            runs.append(load_run_data(rid))

        if runs:
            logger.info("Generating superplot.", n_runs=len(runs))
            _generate_superplot(runs, active)

        logger.info("Visualization complete.")
