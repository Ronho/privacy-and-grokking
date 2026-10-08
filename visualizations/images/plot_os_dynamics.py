#!/usr/bin/env python3
"""plot_os_dynamics.py.

Visualizes optimization and training dynamics across canonical default runs from the
optimizer sweep.

Generates multi-panel figures with 4 designated columns:
  - Column (a): Trajectory in the 2D subspace spanned by the top two principal components
  - Column (b): Train & Test Accuracy (left axis) with Train & Test Loss (right axis, same colors)
  - Column (c): Weight Norm (left axis) with Distance and Step Size (right axis, log scale)
  - Column (d): Optimizer Metrics (First Momentum, Velocity, Update Size)

Outputs:
  - cache/plots/optimizer_sweep_dynamics/os_dynamics_GROK_MNIST_CE_MLP.pdf (single-row)
  - cache/plots/optimizer_sweep_dynamics/os_dynamics_other_models.pdf (4-row combined)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import typer
from matplotlib.collections import LineCollection
from matplotlib.ticker import FuncFormatter

from privacy_and_grokking.artifacts.cache import DEFAULT_TRACKING_URI
from privacy_and_grokking.artifacts.local import (
    get_cache_dir,
    mlflow_export_cache_path,
    plots_optimizer_sweep_dynamics_dir,
    plots_optimizer_sweep_dynamics_path,
    trajectory_projection_path,
)

app = typer.Typer(pretty_exceptions_enable=False)

# Styling tokens matching publication standards
COLOR_TRAIN = "#d55e00"  # Vermillion (Train Acc solid, Train Loss dashed)
COLOR_TEST = "#0072b2"  # Blue (Test Acc solid, Test Loss dashed)

# Column (c) - Distinct colors per independent metric
COLOR_WEIGHT_NORM = "#312e81"  # Deep Indigo / Navy (Left axis, solid)
COLOR_DIST_START = "#7c3aed"  # Purple / Violet (Right axis, solid)
COLOR_STEP_SIZE = "#0891b2"  # Teal / Cyan (Right axis, dashed)

# Column (d) - Distinct colors per independent metric
COLOR_UPDATE = "#dc2626"  # Crimson Red (Foreground solid, zorder=5)
COLOR_MOMENTUM = "#059669"  # Emerald Green (First Momentum, solid, zorder=3)
COLOR_SECOND_MOMENTUM = "#d97706"  # Amber (Second Momentum, dashed, zorder=2)
COLOR_GRADIENT = "#2563eb"  # Royal Blue (Gradient, dashed, zorder=4)

# Canonical default run specifications
DEFAULT_CE_MLP = {
    "model_tag": "GROK_MNIST_CE_MLP",
    "row_label": "(I) MLP (CE)\nGrokking",
    "filename": "os_dynamics_GROK_MNIST_CE_MLP",
    "pca_run_id": "23d2c3bbfd5d4c3eb373ec98564ba77f",
}

OTHER_DEFAULT_MODELS = [
    {
        "model_tag": "GROK_MNIST_MSE_MLP",
        "row_label": "(I) MLP (MSE)\nGrokking",
        "pca_run_id": "580fc01a67d04a6697626730a88e7bb7",
    },
    {
        "model_tag": "NO_GROK_MNIST_CE_MLP",
        "row_label": "(II) MLP (CE)\nNo Grokking",
        "pca_run_id": "51c177903c1d4f29a8f4dae05c35587f",
    },
    {
        "model_tag": "GROK_MNIST_CE_VIT",
        "row_label": "(III) ViT (CE)\nGrokking",
        "pca_run_id": "a46642282384408481241143a6b4bdc0",
    },
    {
        "model_tag": "NO_GROK_MNIST_CE_VIT",
        "row_label": "(IV) ViT (CE)\nNo Grokking",
        "pca_run_id": "0ad0d9f8dbc64e6fa407479a613726d9",
    },
]

# Mapping of model tag to canonical default PCA run IDs
CANONICAL_PCA_MAP = {
    "GROK_MNIST_CE_MLP": "23d2c3bbfd5d4c3eb373ec98564ba77f",
    "GROK_MNIST_MSE_MLP": "580fc01a67d04a6697626730a88e7bb7",
    "NO_GROK_MNIST_CE_MLP": "51c177903c1d4f29a8f4dae05c35587f",
    "GROK_MNIST_CE_VIT": "a46642282384408481241143a6b4bdc0",
    "NO_GROK_MNIST_CE_VIT": "0ad0d9f8dbc64e6fa407479a613726d9",
}

# Known default AdamW run IDs in optimizer-sweep-v1
DEFAULT_RUN_IDS = {
    "GROK_MNIST_CE_MLP": "cf14929914b04fc2a0f08342f00650fb",
    "NO_GROK_MNIST_CE_MLP": "c5ad77cf4ecd451e9a8f61a2dc531b0a",
    "GROK_MNIST_CE_VIT": "8121a0190e68494eb9fb19e00db34411",
    "NO_GROK_MNIST_CE_VIT": "d45b7adf7d314737af7fb986993778eb",
    "GROK_MNIST_MSE_MLP": "58a1b3638b3e4756a937dd7f75a95fa5",
}


def format_step(x: float, pos: int | None = None) -> str:
    """Format step numbers cleanly for linear axes (e.g. 0, 50k, 100k, 1M)."""
    if x <= 0:
        return "0"
    if x >= 1_000_000:
        return f"{x / 1_000_000:.1f}M" if (x % 1_000_000) != 0 else f"{int(x / 1_000_000)}M"
    if x >= 1_000:
        return f"{int(x / 1_000)}k" if x % 1_000 == 0 else f"{x / 1_000:.1f}k"
    return f"{int(x)}"


def load_pca_projection(run_id: str, model_tag: str | None = None) -> dict[str, Any] | None:
    """Load cached PCA projection data and compute Euclidean distances."""
    candidate_ids = [run_id]
    if model_tag and model_tag in CANONICAL_PCA_MAP:
        candidate_ids.append(CANONICAL_PCA_MAP[model_tag])

    for cid in candidate_ids:
        proj_path = trajectory_projection_path(cid)
        if proj_path.exists():
            try:
                data = json.loads(proj_path.read_text())
                if "trajectories" in data:
                    traj = data["trajectories"]
                    coords = np.array(traj.get("pca_coords", []))
                    if len(coords) > 0:
                        # Compute Euclidean distance from initial weights (origin in trajectory)
                        # and step-to-step euclidean distance
                        dist_start = np.linalg.norm(coords - coords[0], axis=1).tolist()
                        step_diff = np.linalg.norm(coords[1:] - coords[:-1], axis=1)
                        step_dist = np.concatenate([[0.0], step_diff]).tolist()
                        traj["distance_from_start"] = dist_start
                        traj["step_distance"] = step_dist
                return data
            except Exception:
                pass

        alt_pca_parquet = get_cache_dir() / f"{cid}_pca.parquet"
        if alt_pca_parquet.exists():
            try:
                df_pca = pl.read_parquet(alt_pca_parquet)
                meta_json = get_cache_dir() / f"{cid}_pca_meta.json"
                evr = [0.0, 0.0]
                if meta_json.exists():
                    meta = json.loads(meta_json.read_text())
                    evr = meta.get("explained_variance_ratio", [0.0, 0.0])

                steps = df_pca["step"].to_list()
                pc1 = df_pca["pc1"].to_list()
                pc2 = df_pca["pc2"].to_list()

                pca_coords = [[p1, p2] for p1, p2 in zip(pc1, pc2, strict=False)]
                coords = np.array(pca_coords)
                dist_start = np.linalg.norm(coords - coords[0], axis=1).tolist()
                step_diff = np.linalg.norm(coords[1:] - coords[:-1], axis=1)
                step_dist = np.concatenate([[0.0], step_diff]).tolist()

                return {
                    "trajectories": {
                        "steps": steps,
                        "pca_coords": pca_coords,
                        "distance_from_start": dist_start,
                        "step_distance": step_dist,
                    },
                    "explained_variance_ratio": evr,
                }
            except Exception:
                pass

    return None


def render_trajectory_panel(
    ax: plt.Axes,
    pca_data: dict[str, Any] | None,
) -> None:
    """Render Column (a): 2D trajectory in top 2 PC subspace with right-side colorbar."""
    if pca_data is None or "trajectories" not in pca_data:
        ax.text(
            0.5,
            0.5,
            "PCA projection not available",
            ha="center",
            va="center",
            fontsize=9.5,
            color="#6b7280",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f3f4f6", edgecolor="#d1d5db"),
        )
        ax.set_xlabel("PC 1", fontsize=10, fontweight="bold", color="black")
        ax.set_ylabel("PC 2", fontsize=10, fontweight="bold", color="black")
        ax.grid(True, linestyle=":", alpha=0.3)
        return

    traj = pca_data["trajectories"]
    coords = np.array(traj.get("pca_coords", []))
    steps = np.array(traj.get("steps", []))

    if len(coords) < 2:
        ax.text(
            0.5,
            0.5,
            "Insufficient checkpoints",
            ha="center",
            va="center",
            fontsize=10,
            color="#6b7280",
            transform=ax.transAxes,
        )
        return

    pc1 = coords[:, 0]
    pc2 = coords[:, 1]

    evr = pca_data.get("explained_variance_ratio", [0.0, 0.0])
    evr1 = evr[0] * 100 if len(evr) > 0 else 0.0
    evr2 = evr[1] * 100 if len(evr) > 1 else 0.0

    points = np.array([pc1, pc2]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    norm = plt.Normalize(vmin=steps.min(), vmax=steps.max())
    lc = LineCollection(segments, cmap="plasma", norm=norm, linewidth=2.0, alpha=0.85, zorder=2)
    lc.set_array(steps[:-1])
    ax.add_collection(lc)

    # Mark discrete trajectory jump locations
    step_diff = np.linalg.norm(coords[1:] - coords[:-1], axis=1)
    jump_threshold = max(float(np.quantile(step_diff, 0.985)), 2.5)
    jump_mask = step_diff >= jump_threshold
    jump_indices = np.where(jump_mask)[0]

    if len(jump_indices) > 0:
        ax.scatter(
            pc1[jump_indices],
            pc2[jump_indices],
            color="#dc2626",
            s=32,
            marker="D",
            edgecolors="black",
            linewidth=0.8,
            zorder=4,
            label="Jump Step",
        )

    # Start (green circle) and End (red star) markers
    ax.scatter(
        [pc1[0]],
        [pc2[0]],
        color="#10b981",
        s=70,
        marker="o",
        edgecolors="black",
        linewidth=1.2,
        zorder=5,
        label="Start",
    )
    ax.scatter(
        [pc1[-1]],
        [pc2[-1]],
        color="#ef4444",
        s=110,
        marker="*",
        edgecolors="black",
        linewidth=1.2,
        zorder=5,
        label="End",
    )

    pad_x = (pc1.max() - pc1.min()) * 0.12 if pc1.max() > pc1.min() else 1.0
    pad_y = (pc2.max() - pc2.min()) * 0.12 if pc2.max() > pc2.min() else 1.0
    ax.set_xlim(pc1.min() - pad_x, pc1.max() + pad_x)
    ax.set_ylim(pc2.min() - pad_y, pc2.max() + pad_y)

    pc1_label = f"PC 1 ({evr1:.1f}%)" if evr1 > 0 else "PC 1"
    pc2_label = f"PC 2 ({evr2:.1f}%)" if evr2 > 0 else "PC 2"
    ax.set_xlabel(pc1_label, fontsize=10, fontweight="bold", color="black")
    ax.set_ylabel(pc2_label, fontsize=10, fontweight="bold", color="black")
    ax.grid(True, linestyle=":", alpha=0.4)

    # Place training step color bar on the right side of the trajectory panel
    cbar = plt.colorbar(lc, ax=ax, orientation="vertical", pad=0.03, fraction=0.046)
    cbar.set_label("Training Step", fontsize=8.5, color="black")
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(format_step))
    cbar.ax.tick_params(labelsize=7.5)


def render_accuracy_loss_panel(
    ax_acc: plt.Axes,
    df_run: pl.DataFrame,
    is_ce: bool = True,
) -> None:
    """Render Column (b): Train & Test Accuracy on left axis, Train & Test Loss on right axis."""
    ax_loss = ax_acc.twinx()

    train_acc = df_run.filter(pl.col("metric_name") == "eval/train/accuracy").sort("step")
    test_acc = df_run.filter(pl.col("metric_name") == "eval/test/accuracy").sort("step")

    train_loss_key = (
        "eval/train/loss/cross_entropy/mean" if is_ce else "eval/train/loss/mse/mean"
    )
    test_loss_key = (
        "eval/test/loss/cross_entropy/mean" if is_ce else "eval/test/loss/mse/mean"
    )

    train_loss = df_run.filter(pl.col("metric_name") == train_loss_key).sort("step")
    test_loss = df_run.filter(pl.col("metric_name") == test_loss_key).sort("step")

    # Accuracy: solid lines
    if not train_acc.is_empty():
        ax_acc.plot(
            train_acc["step"].to_numpy(),
            train_acc["value"].to_numpy(),
            color=COLOR_TRAIN,
            linestyle="-",
            linewidth=1.8,
            label="Train Acc",
            zorder=3,
        )
    if not test_acc.is_empty():
        ax_acc.plot(
            test_acc["step"].to_numpy(),
            test_acc["value"].to_numpy(),
            color=COLOR_TEST,
            linestyle="-",
            linewidth=1.8,
            label="Test Acc",
            zorder=3,
        )

    # Loss: dashed lines (matching train/test colors)
    if not train_loss.is_empty():
        ax_loss.plot(
            train_loss["step"].to_numpy(),
            train_loss["value"].to_numpy(),
            color=COLOR_TRAIN,
            linestyle="--",
            linewidth=1.6,
            alpha=0.85,
            label="Train Loss",
            zorder=2,
        )
    if not test_loss.is_empty():
        ax_loss.plot(
            test_loss["step"].to_numpy(),
            test_loss["value"].to_numpy(),
            color=COLOR_TEST,
            linestyle="--",
            linewidth=1.6,
            alpha=0.85,
            label="Test Loss",
            zorder=2,
        )

    ax_acc.set_ylim(-0.02, 1.05)
    ax_acc.set_ylabel("Accuracy", fontsize=10, fontweight="bold", color="black")
    ax_acc.set_xlabel("Training Step", fontsize=10, fontweight="bold", color="black")
    ax_acc.xaxis.set_major_formatter(FuncFormatter(format_step))
    ax_acc.grid(True, linestyle=":", alpha=0.4)

    ax_loss.set_yscale("log")
    ax_loss.set_ylabel("Loss", fontsize=10, fontweight="bold", color="black")

    lines_acc, labels_acc = ax_acc.get_legend_handles_labels()
    lines_loss, labels_loss = ax_loss.get_legend_handles_labels()
    ax_acc.legend(
        lines_acc + lines_loss,
        labels_acc + labels_loss,
        loc="center right",
        fontsize=8,
        framealpha=0.85,
    )


def render_weight_norm_distance_panel(
    ax_wn: plt.Axes,
    df_run: pl.DataFrame,
    pca_data: dict[str, Any] | None,
) -> None:
    """Render Column (c): Weight Norm with Step Size and Distance from Start (log scale)."""
    ax_dist = ax_wn.twinx()

    wn_df = df_run.filter(
        pl.col("metric_name").is_in(["eval/weight_norm/total", "optim/global_weight_norm"])
    ).sort("step")

    if not wn_df.is_empty():
        eval_wn = wn_df.filter(pl.col("metric_name") == "eval/weight_norm/total")
        chosen_wn = eval_wn if not eval_wn.is_empty() else wn_df
        ax_wn.plot(
            chosen_wn["step"].to_numpy(),
            chosen_wn["value"].to_numpy(),
            color=COLOR_WEIGHT_NORM,
            linestyle="-",
            linewidth=1.8,
            label="Weight Norm",
            zorder=3,
        )

    has_dist_from_pca = False
    if pca_data is not None and "trajectories" in pca_data:
        traj = pca_data["trajectories"]
        steps = traj.get("steps", [])
        dist_start = traj.get("distance_from_start")
        step_dist = traj.get("step_distance")

        if dist_start is not None and len(dist_start) == len(steps):
            steps_arr = np.array(steps)
            dist_arr = np.array(dist_start)
            # Filter zeros for valid log scale rendering
            valid_dist = dist_arr > 0
            if np.any(valid_dist):
                ax_dist.plot(
                    steps_arr[valid_dist],
                    dist_arr[valid_dist],
                    color=COLOR_DIST_START,
                    linestyle="-",
                    linewidth=1.6,
                    label="Distance to Start",
                    zorder=2,
                )
                has_dist_from_pca = True

        if step_dist is not None and len(step_dist) == len(steps):
            steps_arr = np.array(steps)
            step_arr = np.array(step_dist)
            valid_step = step_arr > 0
            if np.any(valid_step):
                ax_dist.plot(
                    steps_arr[valid_step],
                    step_arr[valid_step],
                    color=COLOR_STEP_SIZE,
                    linestyle="--",
                    linewidth=1.4,
                    alpha=0.85,
                    label="Step Size",
                    zorder=2,
                )
                # Highlight jump peaks in step size
                jump_threshold = max(float(np.quantile(step_arr[valid_step], 0.985)), 2.5)
                jump_idx = np.where(step_arr >= jump_threshold)[0]
                if len(jump_idx) > 0:
                    ax_dist.scatter(
                        steps_arr[jump_idx],
                        step_arr[jump_idx],
                        color="#dc2626",
                        s=26,
                        marker="D",
                        edgecolors="black",
                        linewidth=0.7,
                        zorder=4,
                    )

    if not has_dist_from_pca:
        step_norm_df = df_run.filter(
            pl.col("metric_name").is_in(["optim/global_update_norm", "optim/global_step_norm"])
        ).sort("step")
        if not step_norm_df.is_empty():
            upd_df = step_norm_df.filter(pl.col("metric_name") == "optim/global_update_norm")
            chosen_upd = upd_df if not upd_df.is_empty() else step_norm_df
            ax_dist.plot(
                chosen_upd["step"].to_numpy(),
                chosen_upd["value"].to_numpy(),
                color=COLOR_STEP_SIZE,
                linestyle="--",
                linewidth=1.4,
                alpha=0.85,
                label="Update Size",
                zorder=2,
            )

    ax_wn.set_ylabel("Weight Norm", fontsize=10, fontweight="bold", color="black")
    ax_wn.set_xlabel("Training Step", fontsize=10, fontweight="bold", color="black")
    ax_wn.xaxis.set_major_formatter(FuncFormatter(format_step))
    ax_wn.grid(True, linestyle=":", alpha=0.4)

    # Log scale for distances
    ax_dist.set_yscale("log")
    ax_dist.set_ylabel("Distance", fontsize=10, fontweight="bold", color="black")

    lines_wn, labels_wn = ax_wn.get_legend_handles_labels()
    lines_dist, labels_dist = ax_dist.get_legend_handles_labels()
    ax_wn.legend(
        lines_wn + lines_dist,
        labels_wn + labels_dist,
        loc="upper left",
        fontsize=8,
        framealpha=0.85,
    )


def render_optimizer_metrics_panel(
    ax: plt.Axes,
    df_run: pl.DataFrame,
) -> None:
    """Render Column (d): Optimizer Metrics with distinct colors for independent metrics."""
    # 1. First momentum norm (m_t)
    mt_df = df_run.filter(pl.col("metric_name") == "optim/global_mt_norm").sort("step")
    if not mt_df.is_empty():
        ax.plot(
            mt_df["step"].to_numpy(),
            mt_df["value"].to_numpy(),
            color=COLOR_MOMENTUM,
            linestyle="-",
            linewidth=1.6,
            label="First Momentum",
            zorder=3,
        )

    # 2. Second momentum (sqrt(v_t) or sqrt(square_avg))
    vt_df = df_run.filter(
        pl.col("metric_name").is_in(
            ["optim/global_sqrt_vt_norm", "optim/global_sqrt_square_avg_norm"]
        )
    ).sort("step")
    if not vt_df.is_empty():
        ax.plot(
            vt_df["step"].to_numpy(),
            vt_df["value"].to_numpy(),
            color=COLOR_SECOND_MOMENTUM,
            linestyle="--",
            linewidth=1.5,
            label="Second Momentum",
            zorder=2,
        )

    # 3. Gradients norm
    grad_df = df_run.filter(pl.col("metric_name") == "optim/global_grad_norm").sort("step")
    if not grad_df.is_empty():
        ax.plot(
            grad_df["step"].to_numpy(),
            grad_df["value"].to_numpy(),
            color=COLOR_GRADIENT,
            linestyle="--",
            linewidth=1.4,
            alpha=0.85,
            label="Gradient",
            zorder=4,
        )

    # 4. Global update size (Foreground)
    upd_df = df_run.filter(pl.col("metric_name") == "optim/global_update_norm").sort("step")
    if not upd_df.is_empty():
        ax.plot(
            upd_df["step"].to_numpy(),
            upd_df["value"].to_numpy(),
            color=COLOR_UPDATE,
            linestyle="-",
            linewidth=2.0,
            alpha=0.95,
            label="Update Size",
            zorder=5,
        )

    ax.set_yscale("log")
    ax.set_ylabel("Norm", fontsize=10, fontweight="bold", color="black")
    ax.set_xlabel("Training Step", fontsize=10, fontweight="bold", color="black")
    ax.xaxis.set_major_formatter(FuncFormatter(format_step))
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)


def render_dynamics_grid(
    rows_spec: list[dict[str, Any]],
    df_metrics: pl.DataFrame,
    output_pdf: Path,
    output_png: Path | None = None,
) -> None:
    """Renders a grid of rows x 4 columns with row indicators and column headers."""
    num_rows = len(rows_spec)
    if num_rows == 0:
        return

    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        nrows=num_rows,
        ncols=4,
        figsize=(22.0, 4.4 * num_rows),
        dpi=300,
        squeeze=False,
    )

    column_headers = [
        "(a) Trajectory",
        "(b) Accuracy and Loss",
        "(c) Weight Norm and Distance",
        "(d) Optimizer Dynamics",
    ]

    for row_idx, spec in enumerate(rows_spec):
        run_id = spec["run_id"]
        model_tag = spec.get("model_tag", "")
        row_label = spec.get("row_label")
        is_ce = "MSE" not in model_tag

        df_run = df_metrics.filter(pl.col("run_id") == run_id)
        pca_data = load_pca_projection(run_id, model_tag=model_tag)

        ax_a = axes[row_idx, 0]
        ax_b = axes[row_idx, 1]
        ax_c = axes[row_idx, 2]
        ax_d = axes[row_idx, 3]

        render_trajectory_panel(ax_a, pca_data)
        render_accuracy_loss_panel(ax_b, df_run, is_ce=is_ce)
        render_weight_norm_distance_panel(ax_c, df_run, pca_data)
        render_optimizer_metrics_panel(ax_d, df_run)

        # Add row indicator on the leftmost side (matching reproduction_nc_grokking)
        if row_label:
            ax_a.annotate(
                row_label,
                xy=(-0.28, 0.5),
                xycoords="axes fraction",
                fontsize=11.5,
                fontweight="bold",
                ha="center",
                va="center",
                rotation=90,
                color="black",
            )

        # Set column headers on top row only
        if row_idx == 0:
            for col_idx, header_text in enumerate(column_headers):
                axes[0, col_idx].set_title(
                    header_text, fontsize=12, fontweight="bold", pad=12, color="black"
                )

    plt.tight_layout()

    fig.savefig(output_pdf, format="pdf", bbox_inches="tight")
    print(f"Saved vector PDF: {output_pdf}")

    if output_png is not None:
        fig.savefig(output_png, format="png", dpi=300, bbox_inches="tight")
        print(f"Saved 300 DPI raster preview: {output_png}")

    plt.close(fig)


@app.command()
def main(
    tracking_uri: Annotated[
        str,
        typer.Option("--tracking-uri", "-u", help="MLflow server URI."),
    ] = DEFAULT_TRACKING_URI,
    export_png: Annotated[
        bool,
        typer.Option(
            "--export-png/--no-export-png", help="Whether to export 300 DPI preview PNGs."
        ),
    ] = True,
) -> None:
    """Entrypoint: Generates publication-ready optimizer dynamics figures."""
    parquet_path = mlflow_export_cache_path("optimizer-sweep-v1")
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Required dataset '{parquet_path}' not found. Run export_mlflow_metrics.py first."
        )

    print(f"Loading metrics from {parquet_path}...")
    df = pl.read_parquet(parquet_path)

    plots_dir = plots_optimizer_sweep_dynamics_dir()
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate single-row plot for default MLP CE Grokking model
    print(f"\n{'=' * 70}\n[Plotting] Default MLP CE Grokking Model\n{'=' * 70}")
    ce_mlp_rid = DEFAULT_RUN_IDS["GROK_MNIST_CE_MLP"]
    ce_mlp_spec = [
        {
            "run_id": ce_mlp_rid,
            "model_tag": DEFAULT_CE_MLP["model_tag"],
            "row_label": None,
        }
    ]
    out_ce_pdf = plots_optimizer_sweep_dynamics_path(model_tag="GROK_MNIST_CE_MLP", fmt="pdf")
    out_ce_png = (
        plots_optimizer_sweep_dynamics_path(model_tag="GROK_MNIST_CE_MLP", fmt="png")
        if export_png
        else None
    )
    render_dynamics_grid(ce_mlp_spec, df, out_ce_pdf, out_ce_png)

    # 2. Generate 4-row combined plot for other default models
    print(f"\n{'=' * 70}\n[Plotting] Other Default Models (Combined)\n{'=' * 70}")
    other_specs = []
    for m_info in OTHER_DEFAULT_MODELS:
        m_tag = m_info["model_tag"]
        rid = DEFAULT_RUN_IDS[m_tag]
        other_specs.append(
            {
                "run_id": rid,
                "model_tag": m_tag,
                "row_label": m_info["row_label"],
            }
        )

    out_other_pdf = plots_optimizer_sweep_dynamics_path(model_tag="other_models", fmt="pdf")
    out_other_png = (
        plots_optimizer_sweep_dynamics_path(model_tag="other_models", fmt="png")
        if export_png
        else None
    )
    render_dynamics_grid(other_specs, df, out_other_pdf, out_other_png)

    print("\n[Done] Successfully generated all optimizer dynamics plots!")


if __name__ == "__main__":
    app()
