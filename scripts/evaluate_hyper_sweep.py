"""Evaluation script for Hyper Sweep Runs (hyper-sweep-v1).

Generates:
1. 2D Scatter Grids (Dataset Size vs. Initialization Scale) reproducing the
   layout and style of the provided reference plot, plus cross-variable relations.
2. Ablation Studies comparing various attacks (Standard CE/MSE, Canary CE/MSE,
   Train+Canary vs. Test) as a function of Dataset Size, Initialization Scale,
   and Weight Decay based on the GROK baseline configuration.
"""

import argparse
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import polars as pl
from matplotlib.backends.backend_pdf import PdfPages

# ==============================================================================
# BASELINE CONFIGURATION CONSTANTS (GROK Models)
# ==============================================================================
DEFAULT_BASE_INIT_SCALE = 9.0
DEFAULT_BASE_DATASET_SIZE = 2000.0
DEFAULT_BASE_WEIGHT_DECAY_MSE = 0.1
DEFAULT_BASE_WEIGHT_DECAY_CE = 0.01
DEFAULT_TARGET_STEP = 100000

# ==============================================================================
# ATTACKS & METRICS DEFINITIONS
# ==============================================================================
ATTACK_SPECS = [
    {
        "id": "ce_loss",
        "label": "MIA CE Loss",
        "auc_metric": "eval/attack/ce_loss/auc",
        "tpr1_metric": "eval/attack/ce_loss/tpr-at-fpr/1",
        "color": "#1f77b4",  # Blue
        "linestyle": "-",
        "marker": "o",
    },
    {
        "id": "mse_loss",
        "label": "MIA MSE Loss",
        "auc_metric": "eval/attack/mse_loss/auc",
        "tpr1_metric": "eval/attack/mse_loss/tpr-at-fpr/1",
        "color": "#17becf",  # Cyan
        "linestyle": "--",
        "marker": "s",
    },
    {
        "id": "canary_ce_loss",
        "label": "Canary CE Loss",
        "auc_metric": "eval/attack/canary_ce_loss/auc",
        "tpr1_metric": "eval/attack/canary_ce_loss/tpr-at-fpr/1",
        "color": "#d62728",  # Red
        "linestyle": "-",
        "marker": "^",
    },
    {
        "id": "canary_mse_loss",
        "label": "Canary MSE Loss",
        "auc_metric": "eval/attack/canary_mse_loss/auc",
        "tpr1_metric": "eval/attack/canary_mse_loss/tpr-at-fpr/1",
        "color": "#ff7f0e",  # Orange
        "linestyle": "--",
        "marker": "v",
    },
    {
        "id": "train_plus_canary_ce_loss",
        "label": "Train+Canary CE Loss",
        "auc_metric": "eval/attack/train_plus_canary_vs_test/ce_loss/auc",
        "tpr1_metric": "eval/attack/train_plus_canary_vs_test/ce_loss/tpr-at-fpr/1",
        "color": "#2ca02c",  # Green
        "linestyle": "-",
        "marker": "D",
    },
    {
        "id": "train_plus_canary_mse_loss",
        "label": "Train+Canary MSE Loss",
        "auc_metric": "eval/attack/train_plus_canary_vs_test/mse_loss/auc",
        "tpr1_metric": "eval/attack/train_plus_canary_vs_test/mse_loss/tpr-at-fpr/1",
        "color": "#8c564b",  # Brown
        "linestyle": "--",
        "marker": "p",
    },
]

GENERAL_METRIC_SPECS = {
    "RNC1 Train": {
        "metric": "eval/nc/rnc1/train",
        "vmin": 0.0,
        "vmax": 0.10,
        "cmap": "Blues_r",
    },
    "RNC1 Test": {
        "metric": "eval/nc/rnc1/test",
        "vmin": 0.0,
        "vmax": 0.10,
        "cmap": "Blues_r",
    },
    "RNC1 Train Mean Test Var": {
        "metric": "eval/nc/rnc1/train_mean_test_variance",
        "vmin": 0.0,
        "vmax": 0.10,
        "cmap": "Blues_r",
    },
    "Train Accuracy": {
        "metric": "eval/train/accuracy",
        "vmin": 0.0,
        "vmax": 1.0,
        "cmap": "Blues",
    },
    "Test Accuracy": {
        "metric": "eval/test/accuracy",
        "vmin": 0.0,
        "vmax": 1.0,
        "cmap": "Blues",
    },
}


def get_all_needed_metric_names() -> List[str]:
    """Collects all metric names used by the script."""
    metrics = set()
    for spec in GENERAL_METRIC_SPECS.values():
        metrics.add(spec["metric"])
    for spec in ATTACK_SPECS:
        metrics.add(spec["auc_metric"])
        metrics.add(spec["tpr1_metric"])
    return sorted(list(metrics))


def format_log2_tick(y, _):
    """Format powers of 2 as 2^k."""
    if y > 0:
        log_val = math.log2(y)
        if abs(log_val - round(log_val)) < 1e-4:
            return f"$2^{{{int(round(log_val))}}}$"
    return f"{y:g}"


def format_log10_tick(x, _):
    """Format powers of 10 as 10^k."""
    if x > 0:
        log_val = math.log10(x)
        if abs(log_val - round(log_val)) < 1e-4:
            return f"$10^{{{int(round(log_val))}}}$"
    return f"{x:g}"


# ==============================================================================
# DATA LOADING VIA POLARS
# ==============================================================================
def load_hyper_sweep_data(parquet_path: str) -> pl.DataFrame:
    """Loads and standardizes hyper sweep data from parquet using Polars."""
    needed_metrics = get_all_needed_metric_names()
    print(f"Loading data from {parquet_path}...")

    lf = pl.scan_parquet(parquet_path)
    filtered = (
        lf.filter(pl.col("metric_name").is_in(needed_metrics))
        .with_columns(
            [
                pl.col("params.initialization_scale").cast(pl.Float64).alias("init_scale"),
                pl.col("params.weight_decay").cast(pl.Float64).alias("weight_decay"),
                (pl.col("params.train_size").cast(pl.Float64) * 2.0).alias("dataset_size"),
                pl.col("params.loss_function").alias("loss"),
            ]
        )
        .collect()
    )

    print(f"Loaded {len(filtered)} filtered metric rows.")
    return filtered


# ==============================================================================
# PLOT TYPE 1: 2D SCATTER GRID (LAYOUT MATCHING REFERENCE PLOT)
# ==============================================================================
def plot_scatter_grid(
    df: pl.DataFrame,
    loss: str,
    target_step: int,
    base_weight_decay: float,
    output_path_base: str,
):
    """Generates a multi-panel 2D scatter plot (Dataset Size vs.

    Init Scale).

    Specifically includes RNC1 Train, RNC1 Test, Attack AUC, and Attack TPR@1%
    in the 2x2 or 3x2 layout shown in the user's reference image.
    """
    # Subset to loss and base weight decay
    sub = df.filter(
        (pl.col("loss") == loss)
        & (pl.col("weight_decay") == base_weight_decay)
        & (pl.col("step") == target_step)
    )

    if sub.is_empty():
        print(
            f"No data for loss={loss}, weight_decay={base_weight_decay}, target_step={target_step}"
        )
        return

    # Metrics to display in the main grid matching the reference plot
    # Subplot layout: 2 columns, 3 rows
    # Row 0: RNC1 Train, RNC1 Test
    # Row 1: MIA CE Loss AUC, MIA CE Loss TPR@1%
    # Row 2: Canary CE Loss AUC, Canary CE Loss TPR@1% (or MSE if loss==mse)
    primary_attack = "ce_loss" if loss == "cross_entropy" else "mse_loss"
    primary_label = "MIA " + ("CE" if loss == "cross_entropy" else "MSE") + " Loss"
    canary_label = "Canary " + ("CE" if loss == "cross_entropy" else "MSE") + " Loss"
    canary_attack = "canary_" + primary_attack

    panels = [
        ("RNC1 Train", "eval/nc/rnc1/train", 0.0, 0.10, "Blues_r"),
        ("RNC1 Test", "eval/nc/rnc1/test", 0.0, 0.10, "Blues_r"),
        (f"{primary_label} AUC", f"eval/attack/{primary_attack}/auc", 0.4, 1.0, "Blues"),
        (
            f"{primary_label} TPR@1%",
            f"eval/attack/{primary_attack}/tpr-at-fpr/1",
            0.0,
            1.0,
            "Blues",
        ),
        (f"{canary_label} AUC", f"eval/attack/{canary_attack}/auc", 0.4, 1.0, "Blues"),
        (
            f"{canary_label} TPR@1%",
            f"eval/attack/{canary_attack}/tpr-at-fpr/1",
            0.0,
            1.0,
            "Blues",
        ),
    ]

    # Aggregate across seeds
    agg = sub.group_by(["metric_name", "dataset_size", "init_scale"]).agg(
        [pl.col("value").mean().alias("mean_val")]
    )

    n_panels = len(panels)
    n_cols = 2
    n_rows = (n_panels + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4.8 * n_rows))
    fig.suptitle(
        f"Hyper Sweep 2D Parameter Grid | Loss: {loss.upper()} | Step: {target_step} | Weight Decay: {base_weight_decay}",
        fontsize=15,
        y=0.995,
    )

    for idx, (title, metric_key, vmin, vmax, cmap) in enumerate(panels):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        metric_df = agg.filter(pl.col("metric_name") == metric_key)
        if metric_df.is_empty():
            ax.set_title(f"{title} (No data)")
            ax.axis("off")
            continue

        xs = metric_df["dataset_size"].to_numpy()
        ys = metric_df["init_scale"].to_numpy()
        cs = metric_df["mean_val"].to_numpy()

        sc = ax.scatter(
            xs,
            ys,
            c=cs,
            cmap=cmap,
            s=150,
            edgecolors="k",
            linewidths=1.0,
            alpha=0.9,
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_xscale("log")
        ax.set_yscale("log", base=2)
        ax.set_xlabel("Dataset Size", fontsize=11)
        ax.set_ylabel("Init Scale", fontsize=11)
        ax.set_title(title, fontsize=12, pad=8)

        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_log10_tick))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_log2_tick))

        # Set axis limits matching reference plot style
        ax.set_xlim(10, 60000)
        ax.set_ylim(0.5, 64.0)

        # Style ticks & colorbar
        ax.tick_params(axis="both", which="major", labelsize=10)
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(title, fontsize=10)
        cbar.ax.tick_params(labelsize=9)

    plt.tight_layout()
    pdf_path = f"{output_path_base}_scatter_grid.pdf"
    png_path = f"{output_path_base}_scatter_grid.png"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved 2D Scatter Grid to {pdf_path} and {png_path}")


# ==============================================================================
# PLOT TYPE 2: ALL ATTACKS 2D SCATTER GRID
# ==============================================================================
def plot_all_attacks_scatter_grid(
    df: pl.DataFrame,
    loss: str,
    target_step: int,
    base_weight_decay: float,
    output_path_base: str,
):
    """Generates a 6x2 scatter plot grid for all 6 core attacks (AUC &

    TPR@1%).
    """
    sub = df.filter(
        (pl.col("loss") == loss)
        & (pl.col("weight_decay") == base_weight_decay)
        & (pl.col("step") == target_step)
    )

    if sub.is_empty():
        return

    agg = sub.group_by(["metric_name", "dataset_size", "init_scale"]).agg(
        [pl.col("value").mean().alias("mean_val")]
    )

    n_attacks = len(ATTACK_SPECS)
    fig, axes = plt.subplots(n_attacks, 2, figsize=(14, 4.2 * n_attacks))
    fig.suptitle(
        f"Attack Comparison 2D Grid | Loss: {loss.upper()} | Step: {target_step} | WD: {base_weight_decay}",
        fontsize=15,
        y=0.995,
    )

    for i, spec in enumerate(ATTACK_SPECS):
        # Column 0: AUC
        ax_auc = axes[i, 0]
        m_auc = agg.filter(pl.col("metric_name") == spec["auc_metric"])
        if not m_auc.is_empty():
            sc1 = ax_auc.scatter(
                m_auc["dataset_size"].to_numpy(),
                m_auc["init_scale"].to_numpy(),
                c=m_auc["mean_val"].to_numpy(),
                cmap="Blues",
                s=130,
                edgecolors="k",
                vmin=0.4,
                vmax=1.0,
                alpha=0.9,
            )
            ax_auc.set_xscale("log")
            ax_auc.set_yscale("log", base=2)
            ax_auc.set_xlabel("Dataset Size")
            ax_auc.set_ylabel("Init Scale")
            ax_auc.set_title(f"{spec['label']} AUC")
            ax_auc.xaxis.set_major_formatter(ticker.FuncFormatter(format_log10_tick))
            ax_auc.yaxis.set_major_formatter(ticker.FuncFormatter(format_log2_tick))
            ax_auc.set_xlim(10, 60000)
            ax_auc.set_ylim(0.5, 64.0)
            cbar1 = plt.colorbar(sc1, ax=ax_auc)
            cbar1.set_label("AUC")
        else:
            ax_auc.set_title(f"{spec['label']} AUC (No data)")

        # Column 1: TPR@1%
        ax_tpr = axes[i, 1]
        m_tpr = agg.filter(pl.col("metric_name") == spec["tpr1_metric"])
        if not m_tpr.is_empty():
            sc2 = ax_tpr.scatter(
                m_tpr["dataset_size"].to_numpy(),
                m_tpr["init_scale"].to_numpy(),
                c=m_tpr["mean_val"].to_numpy(),
                cmap="Blues",
                s=130,
                edgecolors="k",
                vmin=0.0,
                vmax=1.0,
                alpha=0.9,
            )
            ax_tpr.set_xscale("log")
            ax_tpr.set_yscale("log", base=2)
            ax_tpr.set_xlabel("Dataset Size")
            ax_tpr.set_ylabel("Init Scale")
            ax_tpr.set_title(f"{spec['label']} TPR@1%")
            ax_tpr.xaxis.set_major_formatter(ticker.FuncFormatter(format_log10_tick))
            ax_tpr.yaxis.set_major_formatter(ticker.FuncFormatter(format_log2_tick))
            ax_tpr.set_xlim(10, 60000)
            ax_tpr.set_ylim(0.5, 64.0)
            cbar2 = plt.colorbar(sc2, ax=ax_tpr)
            cbar2.set_label("TPR@1%")
        else:
            ax_tpr.set_title(f"{spec['label']} TPR@1% (No data)")

    plt.tight_layout()
    pdf_path = f"{output_path_base}_attacks_scatter_grid.pdf"
    png_path = f"{output_path_base}_attacks_scatter_grid.png"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved All Attacks Scatter Grid to {pdf_path} and {png_path}")


# ==============================================================================
# PLOT TYPE 3: ABLATION STUDY - ATTACKS VS. PARAMETERS (TARGET STEP)
# ==============================================================================
def plot_ablation_attacks_target_step(
    df: pl.DataFrame,
    loss: str,
    target_step: int,
    base_init_scale: float,
    base_dataset_size: float,
    base_weight_decay: float,
    output_path_base: str,
):
    """Ablation Study:

    Plots attack metrics (AUC and TPR@1%) at target step as each of the 3
    parameters is varied individually, holding the other two fixed at the base
    GROK config.
    """
    sub = df.filter((pl.col("loss") == loss) & (pl.col("step") == target_step))
    if sub.is_empty():
        return

    # Aggregate across seeds
    agg = sub.group_by(
        ["metric_name", "dataset_size", "init_scale", "weight_decay"]
    ).agg([pl.col("value").mean().alias("mean_val"), pl.col("value").std().alias("std_val")])

    for metric_type, y_label, vmin, vmax in [
        ("auc_metric", "Attack AUC", 0.45, 1.02),
        ("tpr1_metric", "Attack TPR @ 1% FPR", -0.02, 1.02),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        fig.suptitle(
            f"Ablation Study: Attacks vs. Hyperparameters | Loss: {loss.upper()} | Step: {target_step}\n"
            f"Baseline: Dataset Size={int(base_dataset_size)}, Init Scale={base_init_scale}, Weight Decay={base_weight_decay}",
            fontsize=13,
            y=1.02,
        )

        # -------------------------------------------------------------
        # Subplot 1: Vary Dataset Size (fixed IS=base, WD=base)
        # -------------------------------------------------------------
        ax1 = axes[0]
        ax1.set_title(
            f"Vary Dataset Size\n(IS={base_init_scale}, WD={base_weight_decay})", fontsize=11
        )
        data_ds = agg.filter(
            (pl.col("init_scale") == base_init_scale)
            & (pl.col("weight_decay") == base_weight_decay)
        )

        for spec in ATTACK_SPECS:
            m_data = data_ds.filter(pl.col("metric_name") == spec[metric_type]).sort(
                "dataset_size"
            )
            if not m_data.is_empty():
                xs = m_data["dataset_size"].to_numpy()
                ys = m_data["mean_val"].to_numpy()
                stds = m_data["std_val"].fill_null(0.0).to_numpy()

                ax1.plot(
                    xs,
                    ys,
                    label=spec["label"],
                    color=spec["color"],
                    linestyle=spec["linestyle"],
                    marker=spec["marker"],
                    markersize=6,
                    linewidth=1.8,
                )
                ax1.fill_between(xs, ys - stds, ys + stds, color=spec["color"], alpha=0.15)

        ax1.set_xscale("log")
        ax1.set_xlabel("Dataset Size", fontsize=11)
        ax1.set_ylabel(y_label, fontsize=11)
        ax1.set_ylim(vmin, vmax)
        ax1.xaxis.set_major_formatter(ticker.FuncFormatter(format_log10_tick))
        ax1.grid(True, linestyle=":", alpha=0.6)

        # -------------------------------------------------------------
        # Subplot 2: Vary Init Scale (fixed DS=base, WD=base)
        # -------------------------------------------------------------
        ax2 = axes[1]
        ax2.set_title(
            f"Vary Initialization Scale\n(DS={int(base_dataset_size)}, WD={base_weight_decay})",
            fontsize=11,
        )
        data_is = agg.filter(
            (pl.col("dataset_size") == base_dataset_size)
            & (pl.col("weight_decay") == base_weight_decay)
        )

        for spec in ATTACK_SPECS:
            m_data = data_is.filter(pl.col("metric_name") == spec[metric_type]).sort(
                "init_scale"
            )
            if not m_data.is_empty():
                xs = m_data["init_scale"].to_numpy()
                ys = m_data["mean_val"].to_numpy()
                stds = m_data["std_val"].fill_null(0.0).to_numpy()

                ax2.plot(
                    xs,
                    ys,
                    label=spec["label"],
                    color=spec["color"],
                    linestyle=spec["linestyle"],
                    marker=spec["marker"],
                    markersize=6,
                    linewidth=1.8,
                )
                ax2.fill_between(xs, ys - stds, ys + stds, color=spec["color"], alpha=0.15)

        ax2.set_xscale("log", base=2)
        ax2.set_xlabel("Initialization Scale", fontsize=11)
        ax2.set_ylabel(y_label, fontsize=11)
        ax2.set_ylim(vmin, vmax)
        ax2.xaxis.set_major_formatter(ticker.FuncFormatter(format_log2_tick))
        ax2.grid(True, linestyle=":", alpha=0.6)

        # -------------------------------------------------------------
        # Subplot 3: Vary Weight Decay (fixed DS=base, IS=base)
        # -------------------------------------------------------------
        ax3 = axes[2]
        ax3.set_title(
            f"Vary Weight Decay\n(DS={int(base_dataset_size)}, IS={base_init_scale})",
            fontsize=11,
        )
        data_wd = agg.filter(
            (pl.col("dataset_size") == base_dataset_size)
            & (pl.col("init_scale") == base_init_scale)
        )

        for spec in ATTACK_SPECS:
            m_data = data_wd.filter(pl.col("metric_name") == spec[metric_type]).sort(
                "weight_decay"
            )
            if not m_data.is_empty():
                xs = m_data["weight_decay"].to_numpy()
                ys = m_data["mean_val"].to_numpy()
                stds = m_data["std_val"].fill_null(0.0).to_numpy()

                ax3.plot(
                    xs,
                    ys,
                    label=spec["label"],
                    color=spec["color"],
                    linestyle=spec["linestyle"],
                    marker=spec["marker"],
                    markersize=6,
                    linewidth=1.8,
                )
                ax3.fill_between(xs, ys - stds, ys + stds, color=spec["color"], alpha=0.15)

        ax3.set_xscale("log")
        ax3.set_xlabel("Weight Decay", fontsize=11)
        ax3.set_ylabel(y_label, fontsize=11)
        ax3.set_ylim(vmin, vmax)
        ax3.xaxis.set_major_formatter(ticker.FuncFormatter(format_log10_tick))
        ax3.grid(True, linestyle=":", alpha=0.6)
        handles, labels = ax3.get_legend_handles_labels()
        if labels:
            ax3.legend(fontsize=9, loc="best", framealpha=0.9)

        plt.tight_layout()
        tag = "auc" if "auc" in metric_type else "tpr1"
        pdf_path = f"{output_path_base}_ablation_attacks_{tag}.pdf"
        png_path = f"{output_path_base}_ablation_attacks_{tag}.png"
        plt.savefig(pdf_path, bbox_inches="tight")
        plt.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved Ablation Parameter Curves ({tag}) to {pdf_path} and {png_path}")


# ==============================================================================
# PLOT TYPE 4: ABLATION STUDY - DYNAMICS OVER TRAINING STEPS
# ==============================================================================
def plot_ablation_attacks_dynamics(
    df: pl.DataFrame,
    loss: str,
    base_init_scale: float,
    base_dataset_size: float,
    base_weight_decay: float,
    output_path_base: str,
):
    """Ablation Study:

    Plots attack dynamics (AUC and TPR@1%) across training steps as Dataset
    Size, Init Scale, and Weight Decay are varied.
    """
    sub = df.filter(pl.col("loss") == loss)
    if sub.is_empty():
        return

    # Select representative attacks for the dynamics grid to keep it compact and readable
    selected_attacks = [
        ("MIA CE Loss AUC", "eval/attack/ce_loss/auc", 0.45, 1.02),
        ("MIA CE Loss TPR@1%", "eval/attack/ce_loss/tpr-at-fpr/1", -0.02, 1.02),
        ("Canary CE Loss AUC", "eval/attack/canary_ce_loss/auc", 0.45, 1.02),
        ("Canary CE Loss TPR@1%", "eval/attack/canary_ce_loss/tpr-at-fpr/1", -0.02, 1.02),
        ("Train+Canary CE AUC", "eval/attack/train_plus_canary_vs_test/ce_loss/auc", 0.45, 1.02),
        (
            "Train+Canary CE TPR@1%",
            "eval/attack/train_plus_canary_vs_test/ce_loss/tpr-at-fpr/1",
            -0.02,
            1.02,
        ),
    ]

    # If loss is MSE, adjust to MSE metrics
    if loss == "mse":
        selected_attacks = [
            ("MIA MSE Loss AUC", "eval/attack/mse_loss/auc", 0.45, 1.02),
            ("MIA MSE Loss TPR@1%", "eval/attack/mse_loss/tpr-at-fpr/1", -0.02, 1.02),
            ("Canary MSE Loss AUC", "eval/attack/canary_mse_loss/auc", 0.45, 1.02),
            ("Canary MSE Loss TPR@1%", "eval/attack/canary_mse_loss/tpr-at-fpr/1", -0.02, 1.02),
            (
                "Train+Canary MSE AUC",
                "eval/attack/train_plus_canary_vs_test/mse_loss/auc",
                0.45,
                1.02,
            ),
            (
                "Train+Canary MSE TPR@1%",
                "eval/attack/train_plus_canary_vs_test/mse_loss/tpr-at-fpr/1",
                -0.02,
                1.02,
            ),
        ]

    n_rows = len(selected_attacks)
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 3.8 * n_rows))
    fig.suptitle(
        f"Ablation Study: Attack Dynamics over Steps | Loss: {loss.upper()}\n"
        f"Baseline: DS={int(base_dataset_size)}, IS={base_init_scale}, WD={base_weight_decay}",
        fontsize=14,
        y=0.998,
    )

    # 1. Dataset Size group
    group_ds = sub.filter(
        (pl.col("init_scale") == base_init_scale) & (pl.col("weight_decay") == base_weight_decay)
    )
    unique_ds = sorted(group_ds["dataset_size"].unique().to_list())
    colors_ds = plt.cm.viridis(np.linspace(0, 0.9, max(len(unique_ds), 1)))

    # 2. Init Scale group
    group_is = sub.filter(
        (pl.col("dataset_size") == base_dataset_size)
        & (pl.col("weight_decay") == base_weight_decay)
    )
    unique_is = sorted(group_is["init_scale"].unique().to_list())
    colors_is = plt.cm.plasma(np.linspace(0, 0.9, max(len(unique_is), 1)))

    # 3. Weight Decay group
    group_wd = sub.filter(
        (pl.col("dataset_size") == base_dataset_size) & (pl.col("init_scale") == base_init_scale)
    )
    unique_wd = sorted(group_wd["weight_decay"].unique().to_list())
    colors_wd = plt.cm.coolwarm(np.linspace(0, 1.0, max(len(unique_wd), 1)))

    for r_idx, (m_label, m_key, vmin, vmax) in enumerate(selected_attacks):
        # Col 0: Vary Dataset Size
        ax0 = axes[r_idx, 0]
        if r_idx == 0:
            ax0.set_title(
                f"Vary Dataset Size\n(IS={base_init_scale}, WD={base_weight_decay})", fontsize=12
            )
        for ds_val, c in zip(unique_ds, colors_ds):
            m_data = (
                group_ds.filter((pl.col("dataset_size") == ds_val) & (pl.col("metric_name") == m_key))
                .group_by("step")
                .agg(pl.col("value").mean().alias("mean_val"))
                .sort("step")
            )
            if not m_data.is_empty():
                ax0.plot(
                    m_data["step"].to_numpy(),
                    m_data["mean_val"].to_numpy(),
                    label=f"DS={int(ds_val)}",
                    color=c,
                    linewidth=1.6,
                )

        ax0.set_xscale("log")
        ax0.set_ylabel(m_label, fontsize=10)
        ax0.set_xlabel("Step" if r_idx == n_rows - 1 else "")
        ax0.set_ylim(vmin, vmax)
        ax0.grid(True, linestyle=":", alpha=0.5)
        if r_idx == 0:
            handles0, labels0 = ax0.get_legend_handles_labels()
            if labels0:
                ax0.legend(fontsize=8, loc="best")

        # Col 1: Vary Init Scale
        ax1 = axes[r_idx, 1]
        if r_idx == 0:
            ax1.set_title(
                f"Vary Init Scale\n(DS={int(base_dataset_size)}, WD={base_weight_decay})",
                fontsize=12,
            )
        for is_val, c in zip(unique_is, colors_is):
            m_data = (
                group_is.filter((pl.col("init_scale") == is_val) & (pl.col("metric_name") == m_key))
                .group_by("step")
                .agg(pl.col("value").mean().alias("mean_val"))
                .sort("step")
            )
            if not m_data.is_empty():
                ax1.plot(
                    m_data["step"].to_numpy(),
                    m_data["mean_val"].to_numpy(),
                    label=f"IS={is_val}",
                    color=c,
                    linewidth=1.6,
                )

        ax1.set_xscale("log")
        ax1.set_ylabel(m_label, fontsize=10)
        ax1.set_xlabel("Step" if r_idx == n_rows - 1 else "")
        ax1.set_ylim(vmin, vmax)
        ax1.grid(True, linestyle=":", alpha=0.5)
        if r_idx == 0:
            handles1, labels1 = ax1.get_legend_handles_labels()
            if labels1:
                ax1.legend(fontsize=8, loc="best")

        # Col 2: Vary Weight Decay
        ax2 = axes[r_idx, 2]
        if r_idx == 0:
            ax2.set_title(
                f"Vary Weight Decay\n(DS={int(base_dataset_size)}, IS={base_init_scale})",
                fontsize=12,
            )
        for wd_val, c in zip(unique_wd, colors_wd):
            m_data = (
                group_wd.filter((pl.col("weight_decay") == wd_val) & (pl.col("metric_name") == m_key))
                .group_by("step")
                .agg(pl.col("value").mean().alias("mean_val"))
                .sort("step")
            )
            if not m_data.is_empty():
                ax2.plot(
                    m_data["step"].to_numpy(),
                    m_data["mean_val"].to_numpy(),
                    label=f"WD={wd_val}",
                    color=c,
                    linewidth=1.6,
                )

        ax2.set_xscale("log")
        ax2.set_ylabel(m_label, fontsize=10)
        ax2.set_xlabel("Step" if r_idx == n_rows - 1 else "")
        ax2.set_ylim(vmin, vmax)
        ax2.grid(True, linestyle=":", alpha=0.5)
        if r_idx == 0:
            handles2, labels2 = ax2.get_legend_handles_labels()
            if labels2:
                ax2.legend(fontsize=8, loc="best")

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    pdf_path = f"{output_path_base}_ablation_dynamics.pdf"
    png_path = f"{output_path_base}_ablation_dynamics.png"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Ablation Dynamics Grid to {pdf_path} and {png_path}")


# ==============================================================================
# PLOT TYPE 5: 3-VARIABLE RELATIONS (CROSS SECTIONS)
# ==============================================================================
def plot_cross_variable_relations(
    df: pl.DataFrame,
    loss: str,
    target_step: int,
    base_init_scale: float,
    base_dataset_size: float,
    base_weight_decay: float,
    output_path_base: str,
):
    """Generates cross-sectional scatter grids comparing all pairs of the 3

    variables:

    1. Dataset Size vs. Init Scale (fixed WD)
    2. Dataset Size vs. Weight Decay (fixed IS)
    3. Init Scale vs. Weight Decay (fixed DS)
    """
    sub = df.filter((pl.col("loss") == loss) & (pl.col("step") == target_step))
    if sub.is_empty():
        return

    primary_attack = "ce_loss" if loss == "cross_entropy" else "mse_loss"
    key_metrics = [
        (
            f"MIA {primary_attack.upper()} AUC",
            f"eval/attack/{primary_attack}/auc",
            0.4,
            1.0,
            "Blues",
        ),
        (
            f"Canary {primary_attack.upper()} AUC",
            f"eval/attack/canary_{primary_attack}/auc",
            0.4,
            1.0,
            "Blues",
        ),
        ("Train Accuracy", "eval/train/accuracy", 0.0, 1.0, "Blues"),
        ("Test Accuracy", "eval/test/accuracy", 0.0, 1.0, "Blues"),
    ]

    agg = sub.group_by(
        ["metric_name", "dataset_size", "init_scale", "weight_decay"]
    ).agg([pl.col("value").mean().alias("mean_val")])

    fig, axes = plt.subplots(len(key_metrics), 3, figsize=(18, 4.5 * len(key_metrics)))
    fig.suptitle(
        f"Cross-Variable 2D Relations | Loss: {loss.upper()} | Step: {target_step}",
        fontsize=15,
        y=0.995,
    )

    for r_idx, (m_label, m_key, vmin, vmax, cmap) in enumerate(key_metrics):
        m_agg = agg.filter(pl.col("metric_name") == m_key)

        # Col 0: DS vs IS (fixed WD=base)
        ax0 = axes[r_idx, 0]
        data0 = m_agg.filter(pl.col("weight_decay") == base_weight_decay)
        if not data0.is_empty():
            sc0 = ax0.scatter(
                data0["dataset_size"].to_numpy(),
                data0["init_scale"].to_numpy(),
                c=data0["mean_val"].to_numpy(),
                cmap=cmap,
                s=130,
                edgecolors="k",
                vmin=vmin,
                vmax=vmax,
                alpha=0.9,
            )
            ax0.set_xscale("log")
            ax0.set_yscale("log", base=2)
            ax0.set_xlabel("Dataset Size")
            ax0.set_ylabel("Init Scale")
            ax0.xaxis.set_major_formatter(ticker.FuncFormatter(format_log10_tick))
            ax0.yaxis.set_major_formatter(ticker.FuncFormatter(format_log2_tick))
            ax0.set_title(
                f"Dataset Size vs Init Scale\n(WD={base_weight_decay})" if r_idx == 0 else ""
            )
            cbar0 = plt.colorbar(sc0, ax=ax0)
            cbar0.set_label(m_label)

        # Col 1: DS vs WD (fixed IS=base)
        ax1 = axes[r_idx, 1]
        data1 = m_agg.filter(pl.col("init_scale") == base_init_scale)
        if not data1.is_empty():
            sc1 = ax1.scatter(
                data1["dataset_size"].to_numpy(),
                data1["weight_decay"].to_numpy(),
                c=data1["mean_val"].to_numpy(),
                cmap=cmap,
                s=130,
                edgecolors="k",
                vmin=vmin,
                vmax=vmax,
                alpha=0.9,
            )
            ax1.set_xscale("log")
            ax1.set_yscale("log")
            ax1.set_xlabel("Dataset Size")
            ax1.set_ylabel("Weight Decay")
            ax1.xaxis.set_major_formatter(ticker.FuncFormatter(format_log10_tick))
            ax1.yaxis.set_major_formatter(ticker.FuncFormatter(format_log10_tick))
            ax1.set_title(
                f"Dataset Size vs Weight Decay\n(IS={base_init_scale})" if r_idx == 0 else ""
            )
            cbar1 = plt.colorbar(sc1, ax=ax1)
            cbar1.set_label(m_label)

        # Col 2: IS vs WD (fixed DS=base)
        ax2 = axes[r_idx, 2]
        data2 = m_agg.filter(pl.col("dataset_size") == base_dataset_size)
        if not data2.is_empty():
            sc2 = ax2.scatter(
                data2["init_scale"].to_numpy(),
                data2["weight_decay"].to_numpy(),
                c=data2["mean_val"].to_numpy(),
                cmap=cmap,
                s=130,
                edgecolors="k",
                vmin=vmin,
                vmax=vmax,
                alpha=0.9,
            )
            ax2.set_xscale("log", base=2)
            ax2.set_yscale("log")
            ax2.set_xlabel("Init Scale")
            ax2.set_ylabel("Weight Decay")
            ax2.xaxis.set_major_formatter(ticker.FuncFormatter(format_log2_tick))
            ax2.yaxis.set_major_formatter(ticker.FuncFormatter(format_log10_tick))
            ax2.set_title(
                f"Init Scale vs Weight Decay\n(DS={int(base_dataset_size)})" if r_idx == 0 else ""
            )
            cbar2 = plt.colorbar(sc2, ax=ax2)
            cbar2.set_label(m_label)

    plt.tight_layout()
    pdf_path = f"{output_path_base}_cross_sections.pdf"
    png_path = f"{output_path_base}_cross_sections.png"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Cross-Variable Relations to {pdf_path} and {png_path}")


# ==============================================================================
# MAIN DRIVER FUNCTION
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and plot Hyper Sweep Runs (hyper-sweep-v1)."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="cache/hyper-sweep-v1_mlflow_export.parquet",
        help="Path to hyper sweep parquet file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots/hyper_sweep",
        help="Directory to save generated plots.",
    )
    parser.add_argument(
        "--loss",
        choices=["all", "mse", "cross_entropy"],
        default="all",
        help="Loss function(s) to evaluate.",
    )
    parser.add_argument(
        "--target-step",
        type=int,
        default=DEFAULT_TARGET_STEP,
        help="Target training step for convergence scatter plots and ablation parameter curves.",
    )
    parser.add_argument(
        "--base-init-scale",
        type=float,
        default=DEFAULT_BASE_INIT_SCALE,
        help="Baseline initialization scale for GROK ablation study.",
    )
    parser.add_argument(
        "--base-dataset-size",
        type=float,
        default=DEFAULT_BASE_DATASET_SIZE,
        help="Baseline dataset size for GROK ablation study.",
    )
    parser.add_argument(
        "--base-weight-decay",
        type=float,
        default=None,
        help="Baseline weight decay (defaults to 0.1 for MSE, 0.01 for CE).",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist.")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_hyper_sweep_data(args.input)

    losses = ["mse", "cross_entropy"] if args.loss == "all" else [args.loss]

    for loss in losses:
        loss_df = df.filter(pl.col("loss") == loss)
        if loss_df.is_empty():
            print(f"Warning: No data found for loss {loss}. Skipping.")
            continue

        # Check and resolve baseline init scale
        available_scales = sorted(loss_df["init_scale"].unique().to_list())
        base_is = args.base_init_scale
        if base_is not in available_scales:
            closest_is = min(available_scales, key=lambda x: abs(x - base_is))
            print(f"Note: Base init scale {base_is} not in {loss} data. Using closest: {closest_is}")
            base_is = closest_is

        # Check and resolve baseline dataset size
        available_ds = sorted(loss_df["dataset_size"].unique().to_list())
        base_ds = args.base_dataset_size
        if base_ds not in available_ds:
            closest_ds = min(available_ds, key=lambda x: abs(x - base_ds))
            print(f"Note: Base dataset size {base_ds} not in {loss} data. Using closest: {closest_ds}")
            base_ds = closest_ds

        # Determine baseline weight decay for this loss if not explicitly provided
        available_wds = sorted(loss_df["weight_decay"].unique().to_list())
        if args.base_weight_decay is not None:
            base_wd = args.base_weight_decay
        else:
            base_wd = (
                DEFAULT_BASE_WEIGHT_DECAY_MSE
                if loss == "mse"
                else DEFAULT_BASE_WEIGHT_DECAY_CE
            )
        if base_wd not in available_wds:
            closest_wd = min(available_wds, key=lambda x: abs(x - base_wd))
            print(f"Note: Base weight decay {base_wd} not in {loss} data. Using closest: {closest_wd}")
            base_wd = closest_wd

        print(f"\n==========================================")
        print(f"Processing Loss: {loss.upper()}")
        print(
            f"Baselines -> Init Scale: {base_is}, Dataset Size: {base_ds}, Weight Decay: {base_wd}"
        )
        print(f"Target Step: {args.target_step}")
        print(f"==========================================")

        prefix = f"{output_dir}/{loss}"

        # 1. 2D Scatter Grid matching user reference plot layout
        plot_scatter_grid(
            df=df,
            loss=loss,
            target_step=args.target_step,
            base_weight_decay=base_wd,
            output_path_base=prefix,
        )

        # 2. All attacks 2D scatter grid
        plot_all_attacks_scatter_grid(
            df=df,
            loss=loss,
            target_step=args.target_step,
            base_weight_decay=base_wd,
            output_path_base=prefix,
        )

        # 3. Ablation Study: Attacks vs Parameters at target step (AUC & TPR@1%)
        plot_ablation_attacks_target_step(
            df=df,
            loss=loss,
            target_step=args.target_step,
            base_init_scale=base_is,
            base_dataset_size=base_ds,
            base_weight_decay=base_wd,
            output_path_base=prefix,
        )

        # 4. Ablation Study: Dynamics over training steps
        plot_ablation_attacks_dynamics(
            df=df,
            loss=loss,
            base_init_scale=base_is,
            base_dataset_size=base_ds,
            base_weight_decay=base_wd,
            output_path_base=prefix,
        )

        # 5. Cross-Variable Relations (DS vs IS, DS vs WD, IS vs WD)
        plot_cross_variable_relations(
            df=df,
            loss=loss,
            target_step=args.target_step,
            base_init_scale=base_is,
            base_dataset_size=base_ds,
            base_weight_decay=base_wd,
            output_path_base=prefix,
        )

    print("\nAll plots generated successfully in:", output_dir)


if __name__ == "__main__":
    main()
