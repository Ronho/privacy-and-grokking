import tempfile
from functools import partial
from pathlib import Path

import mlflow
import numpy as np
import torch
from matplotlib import pyplot as plt
from mlflow import MlflowClient

from privacy_and_grokking.utils import Logger


class DataHandler:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.mlflow_client = MlflowClient()

    def get_metric_history(self, metric_name: str):
        history = self.mlflow_client.get_metric_history(self.run_id, metric_name)
        steps = [m.step for m in history]
        values = [m.value for m in history]
        return {"steps": steps, "values": values}

    def discover_keys(self, prefix: str) -> list[str]:
        run = self.mlflow_client.get_run(self.run_id)
        return sorted(k for k in run.data.metrics if k.startswith(prefix))

    def load_weight_trajectory(self) -> dict[int, np.ndarray]:
        artifacts = self.mlflow_client.list_artifacts(self.run_id, path="checkpoints")
        steps = sorted(
            int(a.path.split("/")[-1]) for a in artifacts if a.path.split("/")[-1].isdigit()
        )
        result: dict[int, np.ndarray] = {}
        for step in steps:
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    mlflow.artifacts.download_artifacts(
                        artifact_uri=f"runs:/{self.run_id}/checkpoints/{step}/model.pth",
                        dst_path=tmpdir,
                    )
                    state_dict = torch.load(
                        Path(tmpdir) / "model.pth",
                        map_location="cpu",
                        weights_only=True,
                    )
                    result[step] = np.concatenate(
                        [p.float().numpy().ravel() for p in state_dict.values()]
                    )
            except Exception:
                continue
        return result

    def load_activation_data(self) -> dict | None:
        artifacts = self.mlflow_client.list_artifacts(self.run_id, path="activations")
        steps = sorted(int(Path(a.path).stem) for a in artifacts if Path(a.path).stem.isdigit())
        if not steps:
            return None
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                mlflow.artifacts.download_artifacts(
                    artifact_uri=f"runs:/{self.run_id}/activations/{steps[-1]}.pt",
                    dst_path=tmpdir,
                )
                return torch.load(
                    Path(tmpdir) / f"{steps[-1]}.pt",
                    map_location="cpu",
                    weights_only=True,
                )
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Dispatch helpers
# ---------------------------------------------------------------------------

# MULTI_AXES_VISUALIZATIONS keys that expand per activation layer
_LAYER_BASED_VIZ = frozenset({"class_layer_activation_grid", "rdm_per_layer", "tsne_per_layer"})
# MULTI_AXES_VISUALIZATIONS key that expands per optimizer state key
_OPTIMIZER_BASED_VIZ = frozenset({"optimizer_internals_over_steps"})


def _discover_dynamic_rows(dh: DataHandler, viz_names: list[str]) -> list[tuple[str, object]]:
    """Expand viz_names into (row_label, ax_fn) pairs.

    SINGLE_AXIS_VISUALIZATIONS entries produce one row each.
    MULTI_AXES_VISUALIZATIONS entries are expanded into one row per activation
    layer (layer-based vizzes) or per optimizer state key, probed from *dh*.
    """
    # Lazy import to avoid circular dependency:
    # visualizations/__init__.py imports DataHandler from this module.
    from privacy_and_grokking.visualize.visualizations import (
        MULTI_AXES_VISUALIZATIONS,
        SINGLE_AXIS_VISUALIZATIONS,
    )

    rows: list[tuple[str, object]] = []

    activation_layers: list[str] | None = None
    activation_loaded = False

    def _get_layers() -> list[str]:
        nonlocal activation_layers, activation_loaded
        if not activation_loaded:
            activation_loaded = True
            data = dh.load_activation_data()
            if data is not None:
                acts = data.get("train_activations", {})
                if acts:
                    activation_layers = sorted(acts.keys())
                else:
                    layer_acts = data.get("train_layer_activations")
                    if layer_acts:
                        activation_layers = list(layer_acts.keys())
        return activation_layers or []

    for viz_name in viz_names:
        if viz_name in SINGLE_AXIS_VISUALIZATIONS:
            rows.append((viz_name, SINGLE_AXIS_VISUALIZATIONS[viz_name]))
        elif viz_name in MULTI_AXES_VISUALIZATIONS:
            fn = MULTI_AXES_VISUALIZATIONS[viz_name]
            if viz_name in _LAYER_BASED_VIZ:
                for layer in _get_layers():
                    rows.append((f"{viz_name}/{layer}", partial(fn, layer=layer)))
            elif viz_name in _OPTIMIZER_BASED_VIZ:
                all_keys = dh.discover_keys("optimizer/")
                seen_state_keys: list[str] = []
                for key in all_keys:
                    parts = key.split("/")
                    if len(parts) == 3 and parts[1] not in seen_state_keys:
                        seen_state_keys.append(parts[1])
                for sk in sorted(seen_state_keys):
                    rows.append((f"optimizer/{sk}", partial(fn, state_key=sk)))
    return rows


def _save_figure_to_mlflow(fig, filename: str, run_id: str, filetype: str = "pdf"):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / f"{filename}.{filetype}"
        fig.savefig(str(path), bbox_inches="tight")
        mlflow.log_artifact(str(path), artifact_path="visualizations", run_id=run_id)


def _single_image_handler(dh: DataHandler, plot_func, filename: str):
    logger = Logger.get()
    logger.info(f"Creating {filename} plot.", extra={"run_id": dh.run_id, "filename": filename})

    run = mlflow.get_run(dh.run_id)
    run_name = run.data.tags.get("mlflow.runName")

    try:
        fig, ax = plt.subplots(figsize=(6, 5))
        plot_func(ax, dh)
        ax.set_yscale("linear")
        ax.set_xscale("linear")
        ax.grid(True, alpha=0.3, which="major", axis="both")
        fig.suptitle(run_name)
        fig.tight_layout()

        _save_figure_to_mlflow(fig, filename, run_id=dh.run_id)
    except Exception as exc:
        logger.warning(
            f"Failed to create {filename} plot: {exc}",
            extra={"run_id": dh.run_id, "filename": filename},
            exc_info=True,
        )
    finally:
        plt.close("all")

    logger.info(f"Created {filename} plot.", extra={"run_id": dh.run_id, "filename": filename})


def visualization_single(
    run_id: str,
    visualizations: list[str],
) -> None:
    # Lazy import to avoid circular dependency with visualizations/__init__.py
    from privacy_and_grokking.visualize.visualizations import (
        MULTI_AXES_VISUALIZATIONS,
        SINGLE_AXIS_VISUALIZATIONS,
    )

    logger = Logger.get()
    dh = DataHandler(run_id)
    for viz_name in visualizations:
        if viz_name in SINGLE_AXIS_VISUALIZATIONS:
            _single_image_handler(dh, SINGLE_AXIS_VISUALIZATIONS[viz_name], viz_name)
        elif viz_name in MULTI_AXES_VISUALIZATIONS:
            rows = _discover_dynamic_rows(dh, [viz_name])
            if not rows:
                continue
            run = mlflow.get_run(run_id)
            run_name = run.data.tags.get("mlflow.runName", run_id)
            n_rows = len(rows)
            try:
                fig, axes = plt.subplots(n_rows, 1, figsize=(8, 5 * n_rows), squeeze=False)
                for row_idx, (row_label, plot_func) in enumerate(rows):
                    ax = axes[row_idx][0]
                    plot_func(ax, dh)
                    ax.set_yscale("linear")
                    ax.set_xscale("linear")
                    ax.grid(True, alpha=0.3, which="major", axis="both")
                    ax.set_title(row_label, fontsize=8)
                fig.suptitle(run_name)
                fig.tight_layout()
                _save_figure_to_mlflow(fig, viz_name, run_id=run_id)
            except Exception as exc:
                logger.warning(
                    f"Failed to create {viz_name} figure: {exc}",
                    extra={"run_id": run_id, "viz_name": viz_name},
                    exc_info=True,
                )
            finally:
                plt.close("all")


def visualization_multi(
    run_ids: list[str],
    visualizations: list[str],
    postfix: str | None = None,
):
    logger = Logger.get()
    run_names: list[str] = []
    for rid in run_ids:
        run = mlflow.get_run(rid)
        run_names.append(run.data.tags.get("mlflow.runName", rid))

    data_handlers = [DataHandler(rid) for rid in run_ids]

    # Expand figure visualizations into per-layer/per-state-key rows,
    # probing the first run to discover layer names / optimizer state keys.
    row_specs = _discover_dynamic_rows(data_handlers[0], visualizations)
    if not row_specs:
        logger.warning("No visualization rows resolved.", extra={"run_ids": run_ids})
        return

    n_rows = len(row_specs)
    n_cols = len(run_ids)
    col_w, row_h = 6.0, 5.0
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(col_w * n_cols, row_h * n_rows),
        squeeze=False,
    )

    for row, (row_label, plot_func) in enumerate(row_specs):
        for col, dh in enumerate(data_handlers):
            ax = axes[row][col]
            try:
                plot_func(ax, dh)
            except Exception as exc:
                ax.text(
                    0.5,
                    0.5,
                    f"Error:\n{exc}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=7,
                    color="tab:red",
                    wrap=True,
                )
            ax.set_yscale("linear")
            ax.set_xscale("linear")
            ax.grid(True, alpha=0.3, which="major", axis="both")

            # Column header on the top row: bold run name + faded run ID
            if row == 0:
                ax.set_title(
                    run_names[col],
                    fontsize=9,
                    fontweight="bold",
                    loc="center",
                )
                ax.text(
                    0.5,
                    1.02,
                    run_ids[col],
                    transform=ax.transAxes,
                    fontsize=6,
                    color="#888888",
                    ha="center",
                    va="bottom",
                )

            # Row label on the left column
            if col == 0:
                ax.set_ylabel(
                    f"{row_label}\n{ax.get_ylabel()}",
                    fontsize=7,
                )

    fig.tight_layout()

    filename = "multi_run_comparison"
    if postfix:
        filename = f"{filename}_{postfix}"

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / f"{filename}.pdf"
        fig.savefig(str(path), bbox_inches="tight")
        for rid in run_ids:
            mlflow.log_artifact(str(path), artifact_path="visualizations", run_id=rid)

    plt.close(fig)
