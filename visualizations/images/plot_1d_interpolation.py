#!/usr/bin/env python3
"""plot_1d_interpolation.py.

Visualizes 1D linear parameter interpolation profiles between checkpoints
along an optimization trajectory (Li et al., NeurIPS 2018; Goodfellow et al., ICLR 2015).

Supports:
  1. Pair mode: Single 1D slice between two checkpoints theta_A -> theta_B for alpha in [alpha_min, alpha_max]
  2. Jumps grid: Multi-panel grid of 1D slices across all detected optimization slingshot jumps / instabilities.

Outputs:
  - cache/plots/interpolation_1d/interpolation_1d_{run_id}_{start_step}_{end_step}.pdf / .png
  - cache/plots/interpolation_1d/interpolation_1d_{run_id}_jumps_grid.pdf / .png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
import typer

from privacy_and_grokking.artifacts.cache import DEFAULT_TRACKING_URI
from privacy_and_grokking.artifacts.local import (
    interpolation_1d_pair_path,
    interpolation_1d_path,
    plots_1d_interpolation_dir,
    plots_1d_interpolation_path,
    plots_1d_jumps_grid_path,
    plots_1d_pair_path,
)
from privacy_and_grokking.utils import Logger
from visualizations.preprocessing.extract_1d_interpolation import (
    extract_1d_interpolation,
    extract_1d_pair_interpolation,
    get_available_checkpoints,
)

app = typer.Typer(pretty_exceptions_enable=False)


def detect_trajectory_jumps(
    run_id: str,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    min_loss_threshold: float = 1e-6,
    jump_ratio: float = 10.0,
) -> list[tuple[int, int]]:
    """Detects checkpoint steps where train loss jumps significantly from an ultra-low basin."""
    data_file = interpolation_1d_path(run_id)
    if not data_file.exists():
        extract_1d_interpolation(run_id=run_id, tracking_uri=tracking_uri)
    data = json.loads(data_file.read_bytes())

    ckpt_evals = data.get("checkpoint_evals", {})
    ckpts = sorted([int(k) for k in ckpt_evals.keys()])
    jumps = []

    for i in range(len(ckpts) - 1):
        c1, c2 = ckpts[i], ckpts[i + 1]
        l1 = ckpt_evals[str(c1)]["train_loss"]
        l2 = ckpt_evals[str(c2)]["train_loss"]
        if l1 < min_loss_threshold and (l2 / max(l1, 1e-15)) >= jump_ratio:
            jumps.append((c1, c2))

    return jumps


def plot_1d_pair_interpolation(
    run_id: str,
    start_step: int | None = None,
    end_step: int | None = None,
    alpha_min: float = -0.5,
    alpha_max: float = 1.5,
    num_points: int = 100,
    output_dir: Path | None = None,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    force_extract: bool = False,
) -> tuple[Path, Path]:
    """Generates a publication-grade 1D parameter linear interpolation plot between two checkpoints."""
    available = get_available_checkpoints(run_id=run_id, tracking_uri=tracking_uri)
    s_start = start_step if start_step is not None else available[0]
    s_end = end_step if end_step is not None else available[-1]

    data_file = interpolation_1d_pair_path(run_id, s_start, s_end)

    if not data_file.exists() or force_extract:
        print(f"Extraction file missing. Extracting 1D pair interpolation for {s_start} -> {s_end}...")
        data = extract_1d_pair_interpolation(
            run_id=run_id,
            start_step=s_start,
            end_step=s_end,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            num_points=num_points,
            tracking_uri=tracking_uri,
            force=True,
        )
    else:
        data = json.loads(data_file.read_bytes())

    samples = data["samples"]
    alphas = np.array([s["alpha"] for s in samples])
    train_losses = np.array([s["train_loss"] for s in samples])
    test_losses = np.array([s["test_loss"] for s in samples])
    train_accs = np.array([s["train_acc"] for s in samples])
    test_accs = np.array([s["test_acc"] for s in samples])
    weight_norms = np.array([s["weight_norm"] for s in samples])

    idx_0 = int(np.argmin(np.abs(alphas - 0.0)))
    idx_1 = int(np.argmin(np.abs(alphas - 1.0)))

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
    })

    fig, (ax_loss, ax_norm) = plt.subplots(
        2, 1, figsize=(8.5, 6.5), sharex=True, height_ratios=[1.2, 1.0], dpi=300
    )

    c_train = "#e66101"  # Orange/Vermillion
    c_test = "#1f78b4"   # Blue
    c_norm = "#6a3d9a"   # Purple
    c_acc_tr = "#33a02c" # Green
    c_acc_te = "#ff7f00" # Light orange

    # --- Top Subplot: Train & Test Loss (Log Scale) ---
    ax_loss.plot(
        alphas, train_losses, label="Train Loss", color=c_train, linewidth=2.0, alpha=0.95
    )
    ax_loss.plot(
        alphas, test_losses, label="Test Loss", color=c_test, linewidth=2.0, alpha=0.95
    )

    ax_loss.scatter(
        [alphas[idx_0]], [train_losses[idx_0]], color=c_train, edgecolors="black", s=60, zorder=6
    )
    ax_loss.scatter(
        [alphas[idx_1]], [train_losses[idx_1]], color=c_train, edgecolors="black", s=60, zorder=6
    )
    ax_loss.scatter(
        [alphas[idx_0]], [test_losses[idx_0]], color=c_test, edgecolors="black", s=60, zorder=6
    )
    ax_loss.scatter(
        [alphas[idx_1]], [test_losses[idx_1]], color=c_test, edgecolors="black", s=60, zorder=6
    )

    ax_loss.axvline(0.0, color="gray", linestyle=":", linewidth=1.2, alpha=0.7)
    ax_loss.axvline(1.0, color="gray", linestyle=":", linewidth=1.2, alpha=0.7)

    ax_loss.text(
        0.0,
        train_losses[idx_0] * 2.0,
        r"$\theta_0$ (Step " f"{s_start:,})",
        verticalalignment="bottom",
        horizontalalignment="center",
        fontsize=10,
        fontweight="bold",
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none"),
    )
    ax_loss.text(
        1.0,
        test_losses[idx_1] * 2.5,
        r"$\theta^*$ (Step " f"{s_end:,})",
        verticalalignment="bottom",
        horizontalalignment="center",
        fontsize=10,
        fontweight="bold",
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none"),
    )

    ax_loss.set_yscale("log")
    ax_loss.set_ylabel(r"Loss $\mathcal{L}((1-\alpha)\theta_A + \alpha\theta_B)$", fontweight="bold")
    ax_loss.set_title(
        f"1D Linear Parameter Interpolation: Step {s_start:,} $\\to$ Step {s_end:,}\nRun ID: {run_id}",
        fontweight="bold",
        pad=10,
    )
    ax_loss.grid(True, which="both", linestyle="--", alpha=0.3, color="gray")
    ax_loss.legend(loc="lower left", framealpha=0.9, edgecolor="none")

    # --- Bottom Subplot: Weight Norm & Accuracy ---
    ax_norm.plot(
        alphas, weight_norms, label=r"Weight Norm $\|\theta(\alpha)\|_2$", color=c_norm, linewidth=2.0
    )
    ax_norm.scatter(
        [alphas[idx_0], alphas[idx_1]],
        [weight_norms[idx_0], weight_norms[idx_1]],
        color=c_norm,
        edgecolors="black",
        s=60,
        zorder=6,
    )

    ax_norm.axvline(0.0, color="gray", linestyle=":", linewidth=1.2, alpha=0.7)
    ax_norm.axvline(1.0, color="gray", linestyle=":", linewidth=1.2, alpha=0.7)

    ax_norm.set_ylabel(r"Weight Norm $\|\theta\|_2$", color=c_norm, fontweight="bold")
    ax_norm.tick_params(axis="y", labelcolor=c_norm)
    ax_norm.set_xlabel(r"Interpolation Parameter $\alpha$", fontweight="bold")
    ax_norm.grid(True, linestyle="--", alpha=0.3, color="gray")

    ax_acc = ax_norm.twinx()
    ax_acc.plot(
        alphas, train_accs * 100, label="Train Acc (%)", color=c_acc_tr, linestyle="--", linewidth=1.5
    )
    ax_acc.plot(
        alphas, test_accs * 100, label="Test Acc (%)", color=c_acc_te, linestyle=":", linewidth=1.5
    )
    ax_acc.set_ylabel("Accuracy (%)", color="#333333", fontweight="bold")
    ax_acc.set_ylim(-5, 105)

    lines_1, labels_1 = ax_norm.get_legend_handles_labels()
    lines_2, labels_2 = ax_acc.get_legend_handles_labels()
    ax_norm.legend(lines_1 + lines_2, labels_1 + labels_2, loc="lower left", framealpha=0.9)

    plt.tight_layout()

    if output_dir is None:
        target_dir = plots_1d_interpolation_dir()
    else:
        target_dir = output_dir

    target_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = plots_1d_pair_path(run_id, s_start, s_end, "pdf")
    png_path = plots_1d_pair_path(run_id, s_start, s_end, "png")

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"Saved 1D Pair Interpolation PDF plot to: {pdf_path}")
    print(f"Saved 1D Pair Interpolation PNG plot to: {png_path}")
    return pdf_path, png_path


def plot_1d_jumps_grid(
    run_id: str,
    alpha_min: float = -0.5,
    alpha_max: float = 1.5,
    num_points: int = 60,
    output_dir: Path | None = None,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    force_extract: bool = False,
) -> tuple[Path, Path]:
    """Generates a publication-grade multi-panel grid of 1D interpolations across all detected jumps."""
    jumps = detect_trajectory_jumps(run_id=run_id, tracking_uri=tracking_uri)
    if not jumps:
        raise RuntimeError(f"No trajectory jumps detected for run '{run_id}'.")

    print(f"Detected {len(jumps)} trajectory jumps for Run ID '{run_id}': {jumps}")

    n_jumps = len(jumps)
    cols = 4 if n_jumps >= 4 else n_jumps
    rows = int(np.ceil(n_jumps / cols))

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 13,
    })

    fig, axes = plt.subplots(
        rows, cols, figsize=(3.8 * cols, 3.2 * rows), squeeze=False, dpi=300, sharex=True, sharey=True
    )

    c_train = "#e66101"  # Vermillion
    c_test = "#1f78b4"   # Blue
    c_norm = "#6a3d9a"   # Purple

    for idx, (s_pre, s_post) in enumerate(jumps):
        r, c = divmod(idx, cols)
        ax = axes[r][c]

        data = extract_1d_pair_interpolation(
            run_id=run_id,
            start_step=s_pre,
            end_step=s_post,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            num_points=num_points,
            tracking_uri=tracking_uri,
            force=force_extract,
        )

        samples = data["samples"]
        alphas = np.array([s["alpha"] for s in samples])
        train_losses = np.array([s["train_loss"] for s in samples])
        test_losses = np.array([s["test_loss"] for s in samples])
        weight_norms = np.array([s["weight_norm"] for s in samples])

        idx_0 = int(np.argmin(np.abs(alphas - 0.0)))
        idx_1 = int(np.argmin(np.abs(alphas - 1.0)))

        # Left y-axis: Loss (log scale)
        ax.plot(alphas, train_losses, label="Train Loss", color=c_train, linewidth=1.6)
        ax.plot(alphas, test_losses, label="Test Loss", color=c_test, linewidth=1.6)

        ax.scatter([alphas[idx_0]], [train_losses[idx_0]], color=c_train, edgecolors="black", s=35, zorder=5)
        ax.scatter([alphas[idx_1]], [train_losses[idx_1]], color=c_train, edgecolors="black", s=35, zorder=5)

        ax.axvline(0.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.6)
        ax.axvline(1.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.6)

        ax.set_yscale("log")
        ax.set_title(f"Jump {idx+1}: {s_pre:,} $\\to$ {s_post:,}", fontweight="bold")
        ax.grid(True, which="both", linestyle="--", alpha=0.25, color="gray")

        # Right y-axis: Weight Norm
        ax_norm = ax.twinx()
        ax_norm.plot(
            alphas, weight_norms, label=r"$\|\theta\|_2$", color=c_norm, linewidth=1.4, linestyle="-."
        )
        ax_norm.scatter(
            [alphas[idx_0], alphas[idx_1]],
            [weight_norms[idx_0], weight_norms[idx_1]],
            color=c_norm,
            edgecolors="black",
            s=25,
            zorder=5,
        )
        ax_norm.tick_params(axis="y", labelcolor=c_norm, labelsize=8)

        if r == rows - 1:
            ax.set_xlabel(r"$\alpha$", fontweight="bold")
        if c == 0:
            ax.set_ylabel(r"Loss $\mathcal{L}(\theta)$ (log)", fontweight="bold")
        if c == cols - 1 or idx == n_jumps - 1:
            ax_norm.set_ylabel(r"$\|\theta\|_2$", color=c_norm, fontweight="bold")

        if idx == 0:
            lines_l, labels_l = ax.get_legend_handles_labels()
            lines_r, labels_r = ax_norm.get_legend_handles_labels()
            ax.legend(lines_l + lines_r, labels_l + labels_r, loc="lower left", framealpha=0.85)

    # Hide unused subplots
    for idx in range(n_jumps, rows * cols):
        r, c = divmod(idx, cols)
        fig.delaxes(axes[r][c])

    fig.suptitle(
        f"1D Parameter Linear Interpolation Across Trajectory Jumps (Slingshots)\nRun ID: {run_id}",
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()

    if output_dir is None:
        target_dir = plots_1d_interpolation_dir()
    else:
        target_dir = output_dir

    target_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = plots_1d_jumps_grid_path(run_id, "pdf")
    png_path = plots_1d_jumps_grid_path(run_id, "png")

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"Saved 1D Jumps Grid PDF plot to: {pdf_path}")
    print(f"Saved 1D Jumps Grid PNG plot to: {png_path}")
    return pdf_path, png_path


@app.command()
def main(
    run_id: Annotated[str, typer.Option("--run-id", help="Target MLflow Run ID.")],
    start_step: Annotated[
        int | None, typer.Option("--start-step", help="Initial checkpoint step.")
    ] = None,
    end_step: Annotated[
        int | None, typer.Option("--end-step", help="Final checkpoint step.")
    ] = None,
    alpha_min: Annotated[
        float, typer.Option("--alpha-min", help="Minimum alpha for interpolation.")
    ] = -0.5,
    alpha_max: Annotated[
        float, typer.Option("--alpha-max", help="Maximum alpha for interpolation.")
    ] = 1.5,
    num_points: Annotated[
        int, typer.Option("--num-points", help="Number of alpha points.")
    ] = 100,
    jumps_grid: Annotated[
        bool, typer.Option("--jumps-grid", help="Plot multi-panel grid of all detected jumps.")
    ] = False,
    output_dir: Annotated[
        Path | None, typer.Option("--output-dir", help="Optional custom output directory.")
    ] = None,
    tracking_uri: Annotated[
        str, typer.Option("--tracking-uri", help="MLflow server URI.")
    ] = DEFAULT_TRACKING_URI,
    force_extract: Annotated[
        bool, typer.Option("--force-extract", help="Force re-extraction of interpolation data.")
    ] = False,
) -> None:
    """CLI endpoint to plot 1D linear parameter interpolation."""
    Logger().setup()
    if jumps_grid:
        plot_1d_jumps_grid(
            run_id=run_id,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            num_points=num_points,
            output_dir=output_dir,
            tracking_uri=tracking_uri,
            force_extract=force_extract,
        )
    else:
        plot_1d_pair_interpolation(
            run_id=run_id,
            start_step=start_step,
            end_step=end_step,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            num_points=num_points,
            output_dir=output_dir,
            tracking_uri=tracking_uri,
            force_extract=force_extract,
        )


if __name__ == "__main__":
    app()
