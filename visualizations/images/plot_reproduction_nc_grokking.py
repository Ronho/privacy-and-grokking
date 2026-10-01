#!/usr/bin/env python3
"""plot_reproduction_nc_grokking.py.

Generates publication-quality multi-panel overview figures reproducing neural collapse
and grokking across 8 canonical model configurations:
  1. General Metrics (8 rows x 3 columns):
     - Column (a): Train & Test Accuracy over step [0, 1]
     - Column (b): Train & Test Loss (log-scale) over step
     - Column (c): Total & Last-Layer Weight Norms over step
  2. Neural Collapse Metrics (8 rows x 4 columns):
     - Column (a): NC1 (left axis) & RNC1 (right axis) over step
     - Column (b): NC2 Features (equinorm, equiangularity, maximal angle equiangularity)
     - Column (c): NC2 Classifier Weights (equinorm, equiangularity, maximal angle equiangularity)
     - Column (d): NC3 (left axis) & NC4 (right axis) over step

Rows correspond to the 8 canonical model architectures and training setups:
  - Row 1: (I) MLP (MSE) - Grokking
  - Row 2: (II) MLP (MSE) - No Grokking
  - Row 3: (III) MLP (CE) - Grokking
  - Row 4: (IV) MLP (CE) - No Grokking
  - Row 5: (V) ViT (CE) - Grokking
  - Row 6: (VI) ViT (CE) - No Grokking
  - Row 7: (VII) MADD (CE) - Grokking
  - Row 8: (VIII) MADD (MSE) - Grokking

Individual run plots and isolated group plots are omitted; only consolidated all-in-one
general_metrics and nc_metrics files are exported in vector PDF and high-res PNG formats.

Required Inputs:
  - cache/reproduction-nc-grokking-v1_mlflow_export.parquet: Parquet file containing exported
    metrics history across runs and steps for the reproduction experiment.

Generated Outputs:
  - cache/plots/reproduction_nc_grokking-v1/general_metrics.pdf: Vector PDF general metrics.
  - cache/plots/reproduction_nc_grokking-v1/general_metrics.png: 300 DPI raster general metrics.
  - cache/plots/reproduction_nc_grokking-v1/nc_metrics.pdf: Vector PDF neural collapse metrics.
  - cache/plots/reproduction_nc_grokking-v1/nc_metrics.png: 300 DPI raster neural collapse metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from matplotlib.ticker import FuncFormatter

from privacy_and_grokking.artifacts.local import (
    mlflow_export_cache_path,
    plots_reproduction_general_metrics_path,
    plots_reproduction_nc_metrics_path,
)

app = typer.Typer(pretty_exceptions_enable=False)

# Standardized model definitions and titles
MODEL_CONFIGS = [
    {"run_name": "GROK_MNIST_MSE_MLP", "title": "(I) MLP (MSE)\nGrokking"},
    {"run_name": "NO_GROK_MNIST_MSE_MLP", "title": "(II) MLP (MSE)\nNo Grokking"},
    {"run_name": "GROK_MNIST_CE_MLP", "title": "(III) MLP (CE)\nGrokking"},
    {"run_name": "NO_GROK_MNIST_CE_MLP", "title": "(IV) MLP (CE)\nNo Grokking"},
    {"run_name": "GROK_MNIST_CE_VIT", "title": "(V) ViT (CE)\nGrokking"},
    {"run_name": "NO_GROK_MNIST_CE_VIT", "title": "(VI) ViT (CE)\nNo Grokking"},
    {"run_name": "GROK_MADD_CE_TRANSFORMER", "title": "(VII) MADD (CE)\nGrokking"},
    # {"run_name": "GROK_MADD_MSE_TRANSFORMER", "title": "(VIII) MADD (MSE)\nGrokking"},
]


def format_step(x: float, pos: int | None = None) -> str:
    """Format step numbers cleanly for linear axes (e.g. 0, 50k, 100k, 1M)."""
    if x <= 0:
        return "0"
    if x >= 1_000_000:
        return f"{x / 1_000_000:.1f}M" if (x % 1_000_000) != 0 else f"{int(x / 1_000_000)}M"
    if x >= 1_000:
        return f"{int(x / 1_000)}k" if x % 1_000 == 0 else f"{x / 1_000:.1f}k"
    return f"{int(x)}"


def parse_frequency(val: int | str) -> int:
    """Parse human-readable frequency strings (e.g., '1k', '1M', 1000) into integers."""
    if isinstance(val, int):
        return val
    val_s = str(val).strip().lower()
    if val_s.endswith("k"):
        return int(float(val_s[:-1]) * 1000)
    if val_s.endswith("m"):
        return int(float(val_s[:-1]) * 1000000)
    return int(val_s)


def get_polished_name(metric_name: str) -> str:
    """Map raw metric paths to publication-ready figure labels."""
    if metric_name == "eval/train/accuracy":
        return "Train"
    if metric_name == "eval/test/accuracy":
        return "Test"
    if "loss" in metric_name and "train" in metric_name:
        return "Train"
    if "loss" in metric_name and "test" in metric_name:
        return "Test"
    if metric_name == "eval/weight_norm/total":
        return "Total"
    if metric_name.startswith("eval/weight_norm/"):
        return "Last Layer"
    if metric_name == "eval/nc/nc1":
        return "NC1"
    if metric_name == "eval/nc/rnc1/train":
        return "RNC1"
    if metric_name == "eval/nc/rnc1/test":
        return "RNC1 Test"
    if metric_name.startswith("eval/nc/nc2"):
        name = metric_name.replace("eval/nc/", "").replace("_", " ").title()
        return name.replace("Nc2", "NC2")
    if metric_name.startswith("eval/nc/nc3"):
        return "NC3"
    if metric_name.startswith("eval/nc/nc4"):
        return "NC4"
    return metric_name


def backfill_initial_nans(df: pd.DataFrame) -> pd.DataFrame:
    """Detect and backfill initial NaNs (e.g. step 0 for zero-initialized weights in ViT).

    Zero-initialized classifier weights in ViT yield 0/0 (NaN) at step 0 for equiangularity
    and NC3 metrics. Backfilling from the next valid evaluation step ensures continuity.
    """
    nan_mask = df["value"].isna()
    if not nan_mask.any():
        return df

    model_col = (
        "params.model_name"
        if "params.model_name" in df.columns
        else ("run_name" if "run_name" in df.columns else None)
    )
    group_keys = [c for c in ["metric_name", model_col, "step"] if c and c in df.columns]
    nan_summary = df[nan_mask].groupby(group_keys).size().reset_index(name="count")

    print("\n" + "=" * 80)
    print("[NOTICE] NaN values detected and backfilled with next valid evaluation step:")
    for _, row in nan_summary.iterrows():
        m_name = row["metric_name"]
        m_model = row[model_col] if model_col and model_col in row else "all"
        m_step = row.get("step", "?")
        cnt = row["count"]
        print(
            f"  * Metric: '{m_name}' | Model: '{m_model}' | Step: {m_step} ({cnt} rows backfilled)"
        )
    print("  Reason: Zero-initialized classifier weights in ViT yield 0/0 (NaN) at Step 0.")
    print("=" * 80 + "\n")

    sort_cols = [c for c in ["run_id", "run_name", "metric_name", "step"] if c in df.columns]
    bfill_groups = [c for c in ["run_id", "run_name", "metric_name"] if c in df.columns]
    if not bfill_groups:
        bfill_groups = ["metric_name"]
    df = df.sort_values(sort_cols)
    df["value"] = df.groupby(bfill_groups)["value"].bfill()
    return df


def extract_model_df(df: pd.DataFrame, run_name: str) -> pd.DataFrame:
    """Filter metric dataframe for a specific canonical run configuration."""
    if "run_name" in df.columns:
        match = df[df["run_name"] == run_name]
        if not match.empty:
            return match

    # Fallback to parameter matching if run_name column is missing
    r_lower = run_name.lower()
    sub = df.copy()
    if "mse" in r_lower and "params.loss_function" in sub.columns:
        sub = sub[sub["params.loss_function"] == "mse"]
    elif "ce" in r_lower and "params.loss_function" in sub.columns:
        sub = sub[sub["params.loss_function"] == "cross_entropy"]

    if "mlp" in r_lower and "params.model_name" in sub.columns:
        sub = sub[sub["params.model_name"] == "mlp"]
    elif "vit" in r_lower and "params.model_name" in sub.columns:
        sub = sub[sub["params.model_name"] == "vit_torchvision"]
    elif "transformer" in r_lower and "params.model_name" in sub.columns:
        sub = sub[sub["params.model_name"] == "modular_transformer"]

    if ("no_grok" in r_lower or "nogrok" in r_lower) and "params.train_size" in sub.columns:
        sub = sub[sub["params.train_size"] == "25000"]
    elif "grok" in r_lower and "mnist" in r_lower and "params.train_size" in sub.columns:
        sub = sub[sub["params.train_size"] == "1000"]

    return sub


def plot_aggregated(
    ax: plt.Axes,
    df_plot: pd.DataFrame,
    metric_names: list[str],
    title: str | None,
    ylabel: str,
    twin_ax: plt.Axes | None = None,
    twin_metrics: list[str] | None = None,
    show_legend: bool = True,
    vline_x: float | None = None,
    vlines_per_seed: dict[str, float] | None = None,
    show_xlabel: bool = True,
    log_xscale: bool = False,
    separate_seeds: bool = False,
) -> None:
    """Plot metric trajectories (either individual seeds with varying intensity or mean ± std)."""
    if df_plot.empty:
        ax.text(0.5, 0.5, "No data", horizontalalignment="center", verticalalignment="center")
        if title:
            ax.set_title(title)
        return

    colors = plt.cm.tab10.colors
    c_idx = 0
    plotted = False

    sorted_runs = sorted(df_plot["run_id"].unique()) if "run_id" in df_plot.columns else ["default"]
    num_runs = len(sorted_runs)
    # Vary intensity from 0.35 to 0.95 across different random seeds
    alphas = [0.9] if num_runs <= 1 else list(np.linspace(0.35, 0.95, num_runs))

    for m in metric_names:
        m_data = df_plot[df_plot["metric_name"] == m]
        if m_data.empty:
            continue
        plotted = True
        label_name = get_polished_name(m)
        base_color = colors[c_idx % len(colors)]

        if separate_seeds and num_runs > 1:
            for r_idx, r_id in enumerate(sorted_runs):
                r_df = m_data[m_data["run_id"] == r_id].sort_values("step")
                if r_df.empty:
                    continue
                # Label only the most prominent line to avoid duplicate legend entries
                lbl = label_name if r_idx == num_runs - 1 else None
                ax.plot(
                    r_df["step"],
                    r_df["value"],
                    label=lbl,
                    color=base_color,
                    alpha=alphas[r_idx],
                    linewidth=1.2,
                )
        else:
            agg_data = m_data.groupby("step")["value"].agg(["mean", "std"]).reset_index()
            p = ax.plot(
                agg_data["step"],
                agg_data["mean"],
                label=label_name,
                color=base_color,
            )
            color = p[0].get_color()
            if not agg_data["std"].isna().all():
                ax.fill_between(
                    agg_data["step"],
                    agg_data["mean"] - agg_data["std"],
                    agg_data["mean"] + agg_data["std"],
                    color=color,
                    alpha=0.2,
                )
        c_idx += 1

    if not plotted:
        ax.text(0.5, 0.5, "No data", horizontalalignment="center", verticalalignment="center")

    # Mark 100% train accuracy steps
    if separate_seeds and vlines_per_seed and num_runs > 1:
        for r_idx, r_id in enumerate(sorted_runs):
            v = vlines_per_seed.get(str(r_id))
            if v is not None:
                ax.axvline(x=v, color="red", linewidth=1.1, linestyle="--", alpha=alphas[r_idx])
    elif vline_x is not None:
        ax.axvline(x=vline_x, color="red", linewidth=2, linestyle="--")

    if title:
        ax.set_title(title)
    if show_xlabel:
        ax.set_xlabel("Step")

    if log_xscale:
        ax.set_xscale("log")
    else:
        ax.set_xscale("linear")
        ax.xaxis.set_major_formatter(FuncFormatter(format_step))

    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    if twin_ax is not None and twin_metrics is not None:
        for m in twin_metrics:
            m_data = df_plot[df_plot["metric_name"] == m]
            if m_data.empty:
                continue
            label_name = get_polished_name(m)
            base_color = colors[c_idx % len(colors)]

            if separate_seeds and num_runs > 1:
                for r_idx, r_id in enumerate(sorted_runs):
                    r_df = m_data[m_data["run_id"] == r_id].sort_values("step")
                    if r_df.empty:
                        continue
                    lbl = label_name if r_idx == num_runs - 1 else None
                    twin_ax.plot(
                        r_df["step"],
                        r_df["value"],
                        label=lbl,
                        color=base_color,
                        alpha=alphas[r_idx],
                        linestyle="--",
                        linewidth=1.2,
                    )
            else:
                agg_data = m_data.groupby("step")["value"].agg(["mean", "std"]).reset_index()
                p = twin_ax.plot(
                    agg_data["step"],
                    agg_data["mean"],
                    label=label_name,
                    color=base_color,
                    linestyle="--",
                )
                color = p[0].get_color()
                if not agg_data["std"].isna().all():
                    twin_ax.fill_between(
                        agg_data["step"],
                        agg_data["mean"] - agg_data["std"],
                        agg_data["mean"] + agg_data["std"],
                        color=color,
                        alpha=0.2,
                    )
            c_idx += 1
        twin_ax.set_ylabel("RNC1" if "rnc1" in "".join(twin_metrics).lower() else "NC4")

def plot_general_metrics_row(
    axes_row: list[plt.Axes],
    model_df: pd.DataFrame,
    all_metrics: list[str],
    show_col_titles: bool = True,
    show_xlabel: bool = True,
    row_label: str | None = None,
    log_xscale: bool = False,
    separate_seeds: bool = False,
) -> None:
    """Render a single row (3 subplots) of general metrics for a given model."""
    train_acc_data = model_df[model_df["metric_name"] == "eval/train/accuracy"]
    first_step_100 = None
    vlines_per_seed: dict[str, float] = {}
    if not train_acc_data.empty:
        steps_100 = train_acc_data[train_acc_data["value"] >= 0.999]["step"]
        if not steps_100.empty:
            first_step_100 = float(steps_100.min())
        if "run_id" in train_acc_data.columns:
            for r_id, r_df in train_acc_data.groupby("run_id"):
                r_100 = r_df[r_df["value"] >= 0.999]["step"]
                if not r_100.empty:
                    vlines_per_seed[str(r_id)] = float(r_100.min())

    # Subplot (a) Train and Test Accuracy
    acc_metrics = [
        m
        for m in all_metrics
        if "accuracy" in m.lower() and ("train" in m.lower() or "test" in m.lower())
    ]
    if not acc_metrics:
        acc_metrics = [m for m in all_metrics if "acc" in m.lower()]

    title_a = "(a) Train and Test Accuracy" if show_col_titles else None
    plot_aggregated(
        axes_row[0],
        model_df,
        acc_metrics,
        title_a,
        "Accuracy",
        vline_x=first_step_100,
        vlines_per_seed=vlines_per_seed,
        show_xlabel=show_xlabel,
        log_xscale=log_xscale,
        separate_seeds=separate_seeds,
    )
    axes_row[0].set_ylim(0, 1)

    if row_label:
        axes_row[0].annotate(
            row_label,
            xy=(-0.30, 0.5),
            xycoords="axes fraction",
            fontsize=13,
            fontweight="bold",
            ha="center",
            va="center",
            rotation=90,
        )

    # Subplot (b) Train and Test Loss
    loss_function = "cross_entropy"
    if "params.loss_function" in model_df.columns:
        loss_function = str(model_df["params.loss_function"].iloc[0]).lower()
    elif "run_name" in model_df.columns:
        rname = str(model_df["run_name"].iloc[0]).lower()
        if "mse" in rname:
            loss_function = "mse"

    if "mse" in loss_function:
        loss_metrics = [
            m for m in all_metrics if "loss/mse/mean" in m and ("train" in m or "test" in m)
        ]
    else:
        loss_metrics = [
            m
            for m in all_metrics
            if "loss/cross_entropy/mean" in m and ("train" in m or "test" in m)
        ]

    title_b = "(b) Train and Test Loss" if show_col_titles else None
    plot_aggregated(
        axes_row[1],
        model_df,
        loss_metrics,
        title_b,
        "Loss",
        vline_x=first_step_100,
        vlines_per_seed=vlines_per_seed,
        show_xlabel=show_xlabel,
        log_xscale=log_xscale,
        separate_seeds=separate_seeds,
    )
    axes_row[1].set_yscale("log")

    # Subplot (c) Weight Norm (Total & Last Layer)
    wn_metrics = []
    if "eval/weight_norm/total" in all_metrics:
        wn_metrics.append("eval/weight_norm/total")

    group_metrics = model_df["metric_name"].unique()
    model_name = "unknown"
    if "params.model_name" in model_df.columns:
        model_name = str(model_df["params.model_name"].iloc[0]).lower()
    elif "run_name" in model_df.columns:
        rname = str(model_df["run_name"].iloc[0]).lower()
        if "mlp" in rname:
            model_name = "mlp"
        elif "vit" in rname:
            model_name = "vit_torchvision"
        elif "transformer" in rname:
            model_name = "modular_transformer"

    if model_name == "mlp":
        target_suffix = "/fc3.weight"
    elif model_name == "resnet_torchvision":
        target_suffix = ".fc.weight"
    elif model_name == "vit_torchvision":
        target_suffix = "/vit.heads.head.weight"
    elif model_name == "modular_transformer":
        target_suffix = "/head.weight"
    else:
        target_suffix = None

    last_layer = None
    if target_suffix:
        for m in group_metrics:
            if m.startswith("eval/weight_norm/") and m.endswith(target_suffix):
                last_layer = m
                break

    if last_layer:
        wn_metrics.append(last_layer)

    title_c = "(c) Weight Norm" if show_col_titles else None
    plot_aggregated(
        axes_row[2],
        model_df,
        wn_metrics,
        title_c,
        "Weight Norm",
        vline_x=first_step_100,
        vlines_per_seed=vlines_per_seed,
        show_xlabel=show_xlabel,
        log_xscale=log_xscale,
        separate_seeds=separate_seeds,
    )


def plot_nc_metrics_row(
    axes_row: list[plt.Axes],
    model_df: pd.DataFrame,
    all_metrics: list[str],
    show_col_titles: bool = True,
    show_xlabel: bool = True,
    row_label: str | None = None,
    log_xscale: bool = False,
    separate_seeds: bool = False,
) -> None:
    """Render a single row (4 subplots) of neural collapse metrics for a given model."""
    train_acc_data = model_df[model_df["metric_name"] == "eval/train/accuracy"]
    first_step_100 = None
    vlines_per_seed: dict[str, float] = {}
    if not train_acc_data.empty:
        steps_100 = train_acc_data[train_acc_data["value"] >= 0.999]["step"]
        if not steps_100.empty:
            first_step_100 = float(steps_100.min())
        if "run_id" in train_acc_data.columns:
            for r_id, r_df in train_acc_data.groupby("run_id"):
                r_100 = r_df[r_df["value"] >= 0.999]["step"]
                if not r_100.empty:
                    vlines_per_seed[str(r_id)] = float(r_100.min())

    # Subplot (a) NC1 on the left axis and RNC1 on the right axis
    nc1_metrics = [m for m in all_metrics if m == "eval/nc/nc1"]
    rnc1_metrics = [m for m in all_metrics if m == "eval/nc/rnc1/train"]

    ax_a = axes_row[0]
    ax_a_twin = ax_a.twinx() if rnc1_metrics else None
    title_a = "(a) NC1 & RNC1" if show_col_titles else None
    plot_aggregated(
        ax_a,
        model_df,
        nc1_metrics,
        title_a,
        "NC1",
        twin_ax=ax_a_twin,
        twin_metrics=rnc1_metrics,
        vline_x=first_step_100,
        vlines_per_seed=vlines_per_seed,
        show_xlabel=show_xlabel,
        log_xscale=log_xscale,
        separate_seeds=separate_seeds,
    )

    if row_label:
        ax_a.annotate(
            row_label,
            xy=(-0.30, 0.5),
            xycoords="axes fraction",
            fontsize=13,
            fontweight="bold",
            ha="center",
            va="center",
            rotation=90,
        )

    # Subplot (b) NC2 (Features)
    nc2_metrics_noweights = [
        m
        for m in all_metrics
        if m
        in (
            "eval/nc/nc2_equinorm",
            "eval/nc/nc2_equiangularity",
            "eval/nc/nc2_maximal_angle_equiangularity",
        )
    ]
    ax_b = axes_row[1]
    title_b = "(b) NC2 (Features)" if show_col_titles else None
    plot_aggregated(
        ax_b,
        model_df,
        nc2_metrics_noweights,
        title_b,
        "NC2",
        vline_x=first_step_100,
        vlines_per_seed=vlines_per_seed,
        show_xlabel=show_xlabel,
        log_xscale=log_xscale,
        separate_seeds=separate_seeds,
    )

    # Subplot (c) NC2 (Classifier Weights)
    nc2_metrics_weights = [
        m
        for m in all_metrics
        if m
        in (
            "eval/nc/nc2_equinorm_weights",
            "eval/nc/nc2_equiangularity_weights",
            "eval/nc/nc2_maximal_angle_equiangularity_weights",
        )
    ]
    ax_c = axes_row[2]
    title_c = "(c) NC2 (Weights)" if show_col_titles else None
    plot_aggregated(
        ax_c,
        model_df,
        nc2_metrics_weights,
        title_c,
        "NC2 Weights",
        vline_x=first_step_100,
        vlines_per_seed=vlines_per_seed,
        show_xlabel=show_xlabel,
        log_xscale=log_xscale,
        separate_seeds=separate_seeds,
    )

    # Subplot (d) NC3 and NC4
    nc3_metrics = [m for m in all_metrics if m == "eval/nc/nc3"]
    nc4_metrics = [m for m in all_metrics if m == "eval/nc/nc4"]
    ax_d = axes_row[3]
    ax_d_twin = ax_d.twinx() if nc4_metrics else None
    title_d = "(d) NC3 & NC4" if show_col_titles else None
    plot_aggregated(
        ax_d,
        model_df,
        nc3_metrics,
        title_d,
        "NC3",
        twin_ax=ax_d_twin,
        twin_metrics=nc4_metrics,
        vline_x=first_step_100,
        vlines_per_seed=vlines_per_seed,
        show_xlabel=show_xlabel,
        log_xscale=log_xscale,
        separate_seeds=separate_seeds,
    )
    if ax_d_twin:
        ax_d_twin.set_ylabel("NC4")


def generate_general_metrics_plot(
    df: pd.DataFrame,
    all_metrics: list[str],
    output_path: Path,
    dpi: int = 300,
    log_xscale: bool = False,
    separate_seeds: bool = False,
) -> None:
    """Generate and save the consolidated 8-row x 3-col general metrics overview."""
    n_rows = len(MODEL_CONFIGS)
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, n_cols)

    print(f"Generating consolidated general metrics figure for {n_rows} model setups...")
    for row_idx, cfg in enumerate(MODEL_CONFIGS):
        m_df = extract_model_df(df, cfg["run_name"])
        plot_general_metrics_row(
            axes[row_idx],
            m_df,
            all_metrics,
            show_col_titles=(row_idx == 0),
            show_xlabel=(row_idx == n_rows - 1),
            row_label=cfg["title"],
            log_xscale=log_xscale,
            separate_seeds=separate_seeds,
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pdf_path = output_path.with_suffix(".pdf")
    png_path = output_path.with_suffix(".png")

    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved vector PDF figure to: {pdf_path}")

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved high-res PNG figure to: {png_path}")

    plt.close(fig)


def generate_nc_metrics_plot(
    df: pd.DataFrame,
    all_metrics: list[str],
    output_path: Path,
    dpi: int = 300,
    log_xscale: bool = False,
    separate_seeds: bool = False,
) -> None:
    """Generate and save the consolidated 8-row x 4-col neural collapse metrics overview."""
    n_rows = len(MODEL_CONFIGS)
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, n_cols)

    print(f"Generating consolidated NC metrics figure for {n_rows} model setups...")
    for row_idx, cfg in enumerate(MODEL_CONFIGS):
        m_df = extract_model_df(df, cfg["run_name"])
        plot_nc_metrics_row(
            axes[row_idx],
            m_df,
            all_metrics,
            show_col_titles=(row_idx == 0),
            show_xlabel=(row_idx == n_rows - 1),
            row_label=cfg["title"],
            log_xscale=log_xscale,
            separate_seeds=separate_seeds,
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pdf_path = output_path.with_suffix(".pdf")
    png_path = output_path.with_suffix(".png")

    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved vector PDF figure to: {pdf_path}")

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved high-res PNG figure to: {png_path}")

    plt.close(fig)


@app.command()
def main(
    input_path: Annotated[
        Path | None,
        typer.Option("--input", "-i", help="Path to exported metrics parquet file."),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", "-o", help="Target output directory for plot files."),
    ] = None,
    output_general: Annotated[
        Path | None,
        typer.Option(
            "--output-general", help="Target path for general metrics figure (.pdf/.png)."
        ),
    ] = None,
    output_nc: Annotated[
        Path | None,
        typer.Option("--output-nc", help="Target path for NC metrics figure (.pdf/.png)."),
    ] = None,
    same_resolution: Annotated[
        bool,
        typer.Option(
            "--same-resolution/--no-same-resolution",
            help="Filter step points so every curve has uniform log-frequency resolution.",
        ),
    ] = True,
    log_frequency: Annotated[
        int,
        typer.Option(
            "--log-frequency",
            parser=parse_frequency,
            help="Log frequency step interval for uniform resolution filtering.",
        ),
    ] = 1000,
    log_xscale: Annotated[
        bool,
        typer.Option(
            "--log-xscale/--linear-xscale",
            help="Use logarithmic scale for x-axis (Step). Defaults to linear (normal).",
        ),
    ] = False,
    separate_seeds: Annotated[
        bool,
        typer.Option(
            "--separate-seeds/--aggregate-seeds",
            help="Plot individual seed curves with varying color intensity instead of mean ± std.",
        ),
    ] = False,
    dpi: Annotated[
        int,
        typer.Option("--dpi", help="DPI resolution for exported raster PNG figures."),
    ] = 300,
) -> None:
    """Render publication figures reproducing neural collapse and grokking across 8 models."""
    target_input = input_path or mlflow_export_cache_path("reproduction-nc-grokking-v1")
    if not target_input.exists():
        raise FileNotFoundError(
            f"Input parquet file not found at: {target_input}. "
            "Please export metrics first using export_mlflow_metrics."
        )

    print(f"Loading data from {target_input}...")
    df = pd.read_parquet(target_input)

    if same_resolution:
        initial_len = len(df)
        mask = (
            ((df["step"] < 50) & (df["step"] % 10 == 0))
            | ((df["step"] < log_frequency) & (df["step"] % 100 == 0))
            | (df["step"] % log_frequency == 0)
        )
        df = df[mask].copy()
        print(
            f"Applied same-resolution filter (log_frequency={log_frequency}): "
            f"retained {len(df)}/{initial_len} rows ({len(df['step'].unique())} unique steps)."
        )

    df = backfill_initial_nans(df)
    all_metrics = list(df["metric_name"].unique())

    # Resolve output paths using canonical helpers
    if output_general:
        target_general_path = output_general
    elif output_dir:
        target_general_path = output_dir / "general_metrics.pdf"
    else:
        target_general_path = plots_reproduction_general_metrics_path()

    if output_nc:
        target_nc_path = output_nc
    elif output_dir:
        target_nc_path = output_dir / "nc_metrics.pdf"
    else:
        target_nc_path = plots_reproduction_nc_metrics_path()

    # Generate both publication-grade figures
    generate_general_metrics_plot(
        df,
        all_metrics,
        target_general_path,
        dpi=dpi,
        log_xscale=log_xscale,
        separate_seeds=separate_seeds,
    )
    generate_nc_metrics_plot(
        df,
        all_metrics,
        target_nc_path,
        dpi=dpi,
        log_xscale=log_xscale,
        separate_seeds=separate_seeds,
    )


if __name__ == "__main__":
    app()
