#!/usr/bin/env python3
"""plot_canary_selection_attacks.py.

Generates publication-quality multi-panel overview figures for canary selection attacks
across canonical model configurations:
  1. Accuracy (Train and Test over time)
  2. NC1 over time (with RNC1 on a twin axis)
  3. Loss Attack and InfoRMIA over Time (AUC)
  4. Loss and InfoRMIA Attack 1% over time

Rows correspond to canonical model architectures and training setups.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from matplotlib.ticker import FuncFormatter

from privacy_and_grokking.artifacts.local import (
    plots_canary_accuracy_dir,
)

app = typer.Typer(pretty_exceptions_enable=False)

MODEL_CONFIGS = [
    {"run_name": "GROK_MNIST_MSE_MLP", "title": "(I) MLP (MSE)\nGrokking"},
    {"run_name": "NO_GROK_MNIST_MSE_MLP", "title": "(II) MLP (MSE)\nNo Grokking"},
    {"run_name": "GROK_MNIST_CE_MLP", "title": "(III) MLP (CE)\nGrokking"},
    {"run_name": "NO_GROK_MNIST_CE_MLP", "title": "(IV) MLP (CE)\nNo Grokking"},
    {"run_name": "GROK_MNIST_CE_VIT", "title": "(V) ViT (CE)\nGrokking"},
    {"run_name": "NO_GROK_MNIST_CE_VIT", "title": "(VI) ViT (CE)\nNo Grokking"},
    {"run_name": "GROK_MADD_CE_TRANSFORMER", "title": "(VII) MADD (CE)\nGrokking"},
]


def format_step(x: float, pos: int | None = None) -> str:
    if x <= 0:
        return "0"
    if x >= 1_000_000:
        return f"{x / 1_000_000:.1f}M" if (x % 1_000_000) != 0 else f"{int(x / 1_000_000)}M"
    if x >= 1_000:
        return f"{int(x / 1_000)}k" if x % 1_000 == 0 else f"{x / 1_000:.1f}k"
    return f"{int(x)}"


def get_polished_name(metric_name: str) -> str:
    m = metric_name.lower()
    if m == "eval/train/accuracy":
        return "Train Acc"
    if m == "eval/test/accuracy":
        return "Test Acc"
    if m == "eval/nc/nc1":
        return "NC1"
    if m in ("eval/nc/rnc1/train", "eval/nc/rnc1/test"):
        return "RNC1"
    if "auc" in m:
        if "informia" in m or "rmia" in m:
            return "InfoRMIA"
        return "Loss Attack (MSE)" if "mse" in m else "Loss Attack (CE)"
    if "tpr-at-fpr" in m:
        if "informia" in m or "rmia" in m:
            return "InfoRMIA 1%"
        return "Loss Attack (MSE) 1%" if "mse" in m else "Loss Attack (CE) 1%"
    return metric_name


def extract_model_df(df: pd.DataFrame, run_name: str) -> pd.DataFrame:
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
    show_xlabel: bool = True,
    log_xscale: bool = False,
) -> None:
    if df_plot.empty:
        ax.text(0.5, 0.5, "No data", horizontalalignment="center", verticalalignment="center")
        if title:
            ax.set_title(title)
        return

    colors = plt.cm.tab10.colors
    c_idx = 0
    plotted = False

    for m in metric_names:
        m_data = df_plot[df_plot["metric_name"] == m]
        if m_data.empty:
            continue
        plotted = True
        label_name = get_polished_name(m)
        base_color = colors[c_idx % len(colors)]

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

    if twin_ax is not None and twin_metrics is not None:
        for m in twin_metrics:
            m_data = df_plot[df_plot["metric_name"] == m]
            if m_data.empty:
                continue
            label_name = get_polished_name(m)
            base_color = colors[c_idx % len(colors)]

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

    if not plotted:
        ax.text(0.5, 0.5, "No data", horizontalalignment="center", verticalalignment="center")
    
    # Legend handling
    lines, labels = ax.get_legend_handles_labels()
    if twin_ax:
        lines2, labels2 = twin_ax.get_legend_handles_labels()
        lines += lines2
        labels += labels2
    if lines:
        ax.legend(lines, labels, fontsize="small")

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


def plot_row(
    axes_row: list[plt.Axes],
    model_df: pd.DataFrame,
    show_col_titles: bool = True,
    show_xlabel: bool = True,
    row_label: str | None = None,
    log_xscale: bool = False,
) -> None:
    # 1. Accuracy (Train and Test over time)
    acc_metrics = ["eval/train/accuracy", "eval/test/accuracy"]
    
    # 2. NC1 over time
    nc1_metrics = ["eval/nc/nc1"]
    rnc1_metrics = ["eval/nc/rnc1/train", "eval/nc/rnc1/test"]
    
    # Loss type for attack metrics
    loss_function = "cross_entropy"
    if "params.loss_function" in model_df.columns:
        val = model_df["params.loss_function"].dropna()
        if not val.empty:
            loss_function = str(val.iloc[0]).lower()

    if "mse" in loss_function:
        loss_prefix = "mse_loss"
    else:
        loss_prefix = "ce_loss"

    # 3. Loss Attack and InfoRMIA over Time (AUC)
    auc_metrics = [
        f"eval/attack/train_canary_vs_test/{loss_prefix}/auc",
        "eval/attack/informia/auc",
    ]
    
    # 4. Loss and InfoRMIA Attack 1% over time
    tpr_metrics = [
        f"eval/attack/train_canary_vs_test/{loss_prefix}/tpr-at-fpr/1",
        "eval/attack/informia/tpr-at-fpr/1",
    ]
    
    # Ensure metrics actually exist
    available_metrics = set(model_df["metric_name"].unique())
    auc_metrics = [m for m in auc_metrics if m in available_metrics]
    tpr_metrics = [m for m in tpr_metrics if m in available_metrics]
    
    rnc1_metrics_avail = [m for m in rnc1_metrics if m in available_metrics]

    titles = [
        "(a) Accuracy",
        "(b) NC1",
        "(c) Attack (AUC)",
        "(d) Attack (TPR@1%)",
    ]

    # Plot (a) Accuracy
    plot_aggregated(
        axes_row[0], model_df, acc_metrics, 
        titles[0] if show_col_titles else None, "Accuracy", 
        show_xlabel=show_xlabel, log_xscale=log_xscale
    )
    
    # Plot (b) NC1 + RNC1
    twin_ax = axes_row[1].twinx() if rnc1_metrics_avail else None
    plot_aggregated(
        axes_row[1], model_df, nc1_metrics, 
        titles[1] if show_col_titles else None, "NC1", 
        twin_ax=twin_ax, twin_metrics=rnc1_metrics_avail,
        show_xlabel=show_xlabel, log_xscale=log_xscale
    )
    
    # Plot (c) AUC
    plot_aggregated(
        axes_row[2], model_df, auc_metrics, 
        titles[2] if show_col_titles else None, "AUC", 
        show_xlabel=show_xlabel, log_xscale=log_xscale
    )
    
    # Plot (d) TPR@1%
    plot_aggregated(
        axes_row[3], model_df, tpr_metrics, 
        titles[3] if show_col_titles else None, "TPR@1%", 
        show_xlabel=show_xlabel, log_xscale=log_xscale
    )

    if row_label:
        axes_row[0].annotate(
            row_label,
            xy=(-0.35, 0.5),
            xycoords="axes fraction",
            fontsize=13,
            fontweight="bold",
            ha="center",
            va="center",
            rotation=90,
        )


@app.command()
def main(
    input_path: Annotated[
        Path,
        typer.Option("--input", "-i", help="Path to exported metrics parquet file."),
    ] = Path("cache/canary-selection-v1_mlflow_export.parquet"),
    informia_input: Annotated[
        str,
        typer.Option("--informia-input", help="Path to InfoRMIA parquet results"),
    ] = "cache/canary-selection/informia_results.parquet",
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", "-o", help="Target output directory for plot files."),
    ] = None,
    log_xscale: Annotated[
        bool,
        typer.Option(
            "--log-xscale/--linear-xscale",
            help="Use logarithmic scale for x-axis (Step).",
        ),
    ] = False,
    dpi: Annotated[
        int,
        typer.Option("--dpi", help="DPI resolution for exported raster PNG figures."),
    ] = 300,
) -> None:
    if not input_path.exists():
        print(f"Input parquet file not found at: {input_path}.")
        return

    print(f"Loading data from {input_path}...")
    df = pd.read_parquet(input_path)
    
    # Load InfoRMIA if exists
    df_info = pd.DataFrame()
    if os.path.exists(informia_input):
        print(f"Loading InfoRMIA results from '{informia_input}'...")
        df_info = pd.read_parquet(informia_input)
    else:
        print(f"Warning: InfoRMIA results file '{informia_input}' not found. InfoRMIA will not be plotted.")

    target_output_dir = output_dir or plots_canary_accuracy_dir()
    target_output_dir.mkdir(parents=True, exist_ok=True)

    n_rows = len(MODEL_CONFIGS)
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, n_cols)

    print(f"Generating consolidated attacks figure for {n_rows} model setups...")
    for row_idx, cfg in enumerate(MODEL_CONFIGS):
        m_df = extract_model_df(df, cfg["run_name"])
        
        # Merge InfoRMIA data if present
        if not df_info.empty:
            # InfoRMIA runs are typically formatted as {CANARY_TYPE}_{MODEL_GROUP}, e.g. LABEL_NOISE_GROK_MNIST_CE_MLP
            info_sub = df_info[df_info["run_name"].str.endswith(cfg["run_name"]) & (df_info["canary"] == True)].copy()
            if not info_sub.empty:
                # Add AUC
                if "auc" in info_sub.columns:
                    auc_df = info_sub[["step", "auc"]].copy()
                    auc_df.rename(columns={"auc": "value"}, inplace=True)
                    auc_df["metric_name"] = "eval/attack/informia/auc"
                    m_df = pd.concat([m_df, auc_df], ignore_index=True)
                
                # Add TPR@1%
                if "tpr_at_0.01_fpr" in info_sub.columns:
                    tpr_df = info_sub[["step", "tpr_at_0.01_fpr"]].copy()
                    tpr_df.rename(columns={"tpr_at_0.01_fpr": "value"}, inplace=True)
                    tpr_df["metric_name"] = "eval/attack/informia/tpr-at-fpr/1"
                    m_df = pd.concat([m_df, tpr_df], ignore_index=True)

        plot_row(
            axes[row_idx],
            m_df,
            show_col_titles=(row_idx == 0),
            show_xlabel=(row_idx == n_rows - 1),
            row_label=cfg["title"],
            log_xscale=log_xscale,
        )

    plt.tight_layout()

    pdf_path = target_output_dir / "canary_selection_attacks.pdf"
    png_path = target_output_dir / "canary_selection_attacks.png"

    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved vector PDF figure to: {pdf_path}")

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved high-res PNG figure to: {png_path}")

    plt.close(fig)


if __name__ == "__main__":
    app()
