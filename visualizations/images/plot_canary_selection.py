#!/usr/bin/env python3
"""plot_canary_selection.py.

Generates a publication-quality multi-panel overview figure for canary selection
across 8 canonical model configurations. Rows correspond to the model configurations,
and columns correspond to the canary types.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import pandas as pd
import typer
from matplotlib.ticker import FuncFormatter

from privacy_and_grokking.artifacts.local import (
    plots_canary_accuracy_dir,
)

app = typer.Typer(pretty_exceptions_enable=False)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

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

CANARY_TYPE_MAP = {
    "GAUSSIAN_NOISE": "Gaussian Noise",
    "UNIFORM_NOISE": "Uniform Noise",
    "OOD_NATURAL": "OOD Natural",
    "SQUARE_WATERMARK": "Watermark",
    "WATERMARK": "Watermark",
    "LABEL_NOISE": "Label Noise",
}

CANARY_ORDER = ["Gaussian Noise", "Uniform Noise", "OOD Natural", "Watermark", "Label Noise"]

METRIC_STYLES = {
    "Train": {"color": "#1f77b4", "linestyle": "-"},  # Blue solid
    "Test": {"color": "#ff7f0e", "linestyle": "-"},  # Orange solid
    "Canary Train": {"color": "#2ca02c", "linestyle": "--"},  # Green dashed
    "Canary Test": {"color": "#d62728", "linestyle": "--"},  # Red dashed
}

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def format_step(x: float, pos: int | None = None) -> str:
    """Format step numbers cleanly for linear axes (e.g. 0, 50k, 100k, 1M)."""
    if x <= 0:
        return "0"
    if x >= 1_000_000:
        return f"{x / 1_000_000:.1f}M" if (x % 1_000_000) != 0 else f"{int(x / 1_000_000)}M"
    if x >= 1_000:
        return f"{int(x / 1_000)}k" if x % 1_000 == 0 else f"{x / 1_000:.1f}k"
    return f"{int(x)}"


def get_polished_name(m: str) -> str:
    m_lower = m.lower()
    if "canary" in m_lower and "train" in m_lower:
        return "Canary Train"
    if "canary" in m_lower and "test" in m_lower:
        return "Canary Test"
    if "train" in m_lower and ("acc" in m_lower or "accuracy" in m_lower):
        return "Train"
    if "test" in m_lower and ("acc" in m_lower or "accuracy" in m_lower):
        return "Test"
    return m


def parse_run_name(rname: str) -> tuple[str, str]:
    if not isinstance(rname, str):
        return "Unknown Canary", "Default_Model"
    for k, v in CANARY_TYPE_MAP.items():
        if rname.startswith(k + "_"):
            model = rname[len(k) + 1 :]
            return v, model
    return "Unknown Canary", rname


def extract_model_df(df: pd.DataFrame, run_name: str) -> pd.DataFrame:
    """Filter metric dataframe for a specific canonical run configuration."""
    if "run_name" in df.columns:
        match = df[df["model_group"] == run_name]
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
    elif "grok" in r_lower and "madd" in r_lower and "params.train_size" in sub.columns:
        pass

    return sub

# -----------------------------------------------------------------------------
# Plotting Logic
# -----------------------------------------------------------------------------

def plot_canary_subplot(
    ax: plt.Axes,
    df_plot: pd.DataFrame,
    metric_names: list[str],
    title: str,
    ylabel: str = "Accuracy",
    linear_x: bool = False,
    show_ylabel: bool = True,
    show_xlabel: bool = True,
    show_legend: bool = True,
):
    if df_plot.empty:
        ax.text(0.5, 0.5, "N/A", horizontalalignment="center", verticalalignment="center")
        ax.set_title(title, fontsize=12)
        return

    colors = plt.cm.tab10.colors
    c_idx = 0
    plotted = False

    for m in metric_names:
        m_data = df_plot[df_plot["metric_name"] == m]
        if m_data.empty:
            continue
        plotted = True
        agg_data = m_data.groupby("step")["value"].agg(["mean", "std"]).reset_index()
        label_name = get_polished_name(m)

        style = METRIC_STYLES.get(
            label_name, {"color": colors[c_idx % len(colors)], "linestyle": "-"}
        )
        color = style["color"]
        linestyle = style.get("linestyle", "-")

        p = ax.plot(
            agg_data["step"],
            agg_data["mean"],
            label=label_name,
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
        )
        c = p[0].get_color()
        if not agg_data["std"].isna().all():
            ax.fill_between(
                agg_data["step"],
                agg_data["mean"] - agg_data["std"],
                agg_data["mean"] + agg_data["std"],
                color=c,
                alpha=0.2,
            )
        c_idx += 1

    if not plotted:
        ax.text(0.5, 0.5, "N/A", horizontalalignment="center", verticalalignment="center")

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    
    if show_xlabel:
        ax.set_xlabel("Step", fontsize=10)
    
    if not linear_x:
        ax.set_xscale("log")
    else:
        ax.xaxis.set_major_formatter(FuncFormatter(format_step))

    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3, linestyle="--")
    
    if show_legend and plotted:
        ax.legend(title="Metric", fontsize="small", loc="best", framealpha=0.8)


@app.command()
def main(
    input_path: Annotated[
        Path,
        typer.Option("--input", "-i", help="Path to exported metrics parquet file."),
    ] = Path("cache/canary-selection-v1_mlflow_export.parquet"),
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", "-o", help="Target output directory for plot files."),
    ] = None,
    linear_x: Annotated[
        bool,
        typer.Option("--linear-x/--log-x", help="Use linear scale for x-axis instead of log scale."),
    ] = False,
    dpi: Annotated[
        int,
        typer.Option("--dpi", help="DPI resolution for exported raster PNG figures."),
    ] = 300,
) -> None:
    """Render publication figures for canary selection across models."""
    if not input_path.exists():
        fallback_path = Path("canary-selection_mlflow_export.parquet")
        if fallback_path.exists():
            input_path = fallback_path
        else:
            print(f"Error: Could not find data file '{input_path}'.")
            return

    target_output_dir = output_dir or plots_canary_accuracy_dir()
    target_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {input_path}...")
    df = pd.read_parquet(input_path)

    # Parse canary type and model group
    if "run_name" in df.columns:
        canaries, models = zip(*df["run_name"].map(parse_run_name))
        df["canary_label"] = canaries
        df["model_group"] = models
    else:
        param_cols = [c for c in df.columns if c.startswith("params.")]
        group_cols = [c for c in param_cols if not c.endswith(".seed")]
        df["model_group"] = (
            df[group_cols].astype(str).agg("_".join, axis=1) if group_cols else "Default_Model"
        )
        df["canary_label"] = "Default Canary"

    all_metrics = df["metric_name"].unique()

    # Preferred order of metrics
    preferred_order = [
        "eval/train/accuracy",
        "eval/test/accuracy",
        "eval/train/canary_accuracy",
        "eval/test/canary_accuracy",
    ]
    target_accuracy_metrics = [pref for pref in preferred_order if pref in all_metrics]
    if not target_accuracy_metrics:
        target_accuracy_metrics = [
            m
            for m in all_metrics
            if ("accuracy" in m.lower() or "acc" in m.lower())
            and ("train" in m.lower() or "test" in m.lower() or "canary" in m.lower())
        ]

    print(f"Identified accuracy metrics: {target_accuracy_metrics}")

    # Determine unique canaries in the dataset that fall into our canonical order
    avail_canaries = set(df["canary_label"].unique())
    ordered_canaries = [c for c in CANARY_ORDER if c in avail_canaries]
    
    # Optional fallback if no canonical canaries found
    if not ordered_canaries:
        ordered_canaries = list(sorted(avail_canaries))
        
    n_rows = len(MODEL_CONFIGS)
    n_cols = len(ordered_canaries)
    
    if n_rows == 0 or n_cols == 0:
        print("No valid models or canaries found to plot.")
        return
        
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.5 * n_rows), sharey=True)
    if n_rows == 1: axes = [axes]
    
    for row_idx, cfg in enumerate(MODEL_CONFIGS):
        run_name = cfg["run_name"]
        row_title = cfg["title"]
        model_df = extract_model_df(df, run_name)
        
        ax_row = axes[row_idx] if n_rows > 1 else axes
        if n_cols == 1: ax_row = [ax_row]
        
        for col_idx, canary_name in enumerate(ordered_canaries):
            ax = ax_row[col_idx]
            canary_df = model_df[model_df["canary_label"] == canary_name]
            
            letter_prefix = chr(ord("a") + col_idx)
            subplot_title = f"({letter_prefix}) {canary_name}" if row_idx == 0 else ""
            
            plot_canary_subplot(
                ax,
                canary_df,
                target_accuracy_metrics,
                title=subplot_title,
                ylabel="Accuracy",
                linear_x=linear_x,
                show_ylabel=(col_idx == 0),
                show_xlabel=(row_idx == n_rows - 1),
                show_legend=(row_idx == 0 and col_idx == 0),
            )
            
            # Row annotation on the first column
            if col_idx == 0:
                ax.annotate(
                    row_title,
                    xy=(-0.35, 0.5),
                    xycoords="axes fraction",
                    fontsize=13,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    rotation=90,
                )

    plt.tight_layout()
    out_pdf = target_output_dir / "canary_selection_overview.pdf"
    out_png = target_output_dir / "canary_selection_overview.png"
    
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Canary selection plot saved to '{out_pdf}' and '{out_png}'.")


if __name__ == "__main__":
    app()
