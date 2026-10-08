#!/usr/bin/env python3
"""optimizer_sweep_overview.py.

Generates a publication-quality multi-page PDF visualizing the optimizer sweep results.
For each optimizer and model configuration, it plots:
  - Column (a): Train & Test Accuracy over step
  - Column (b): Train & Test Loss (log-scale) over step
  - Column (c): Total Weight Norm & Last Layer Weight Norm over step
  - Column (d): Optimizer Dynamics (m_t, sqrt(v_t), and global update norm on log-scale) over step

Each page displays parameter ablation studies as rows. For AdamW and RMSprop, 3 separate
pages correspond to the 3 model configurations (MLP MSE, MLP CE, ViT CE) with 3 ablation rows
each. For SGD, all 3 model configurations are consolidated onto a single page (1 row per model
ablating momentum), avoiding empty rows.

Required Inputs:
  - cache/optimizer-sweep-v1_mlflow_export.parquet: Exported metrics parquet.
  - MLflow server (or local cache): Access to training_config.json for run hyperparameters
    (downloaded automatically via MLflow HTTP endpoint and cached in cache/runs/{run_id}/).

Generated Outputs:
  - cache/plots/optimizer_sweep_overview.pdf: 7-page vector PDF publication overview.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import typer
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter

from privacy_and_grokking.artifacts import get_artifact_json
from privacy_and_grokking.artifacts.cache import DEFAULT_TRACKING_URI
from privacy_and_grokking.artifacts.local import (
    mlflow_export_cache_path,
    plots_optimizer_sweep_overview_path,
)

app = typer.Typer(pretty_exceptions_enable=False)

# Styling tokens matching publication standards
COLOR_TRAIN = "#ff7f0e"  # tab:orange
COLOR_TEST = "#1f77b4"  # tab:blue
COLOR_NORM_TOTAL = "#4338ca"  # deep indigo
COLOR_NORM_LAST = "#dc2626"  # crimson red
COLOR_TEXT = "#000000"


def format_step(x: float, pos: int) -> str:
    """Format step numbers cleanly (e.g. 0, 50k, 100k, 1M)."""
    if x <= 0:
        return "0"
    if x >= 1_000_000:
        return f"{x / 1_000_000:.1f}M"
    if x >= 1_000:
        return f"{int(x / 1_000)}k" if x % 1_000 == 0 else f"{x / 1_000:.1f}k"
    return f"{int(x)}"


def extract_optimizer_params(
    df: pl.DataFrame,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    max_workers: int = 8,
) -> pl.DataFrame:
    """Extract and parse optimizer parameters from cached or remote training_config.json."""
    unique_run_ids = df.select("run_id").unique().drop_nulls().to_series().to_list()
    print(
        f"Loading training config for {len(unique_run_ids)} runs (cached or from {tracking_uri})..."
    )

    def _fetch_config(run_id: str) -> dict:
        try:
            cfg = get_artifact_json(run_id, "training_config.json", tracking_uri=tracking_uri)
            opt_cfg = cfg.get("optimizer", {})
            parsed = {"run_id": run_id}
            for k, v in opt_cfg.items():
                if k == "name":
                    parsed["opt_name"] = v
                elif k == "betas" and isinstance(v, list) and len(v) == 2:
                    parsed["beta1"] = float(v[0])
                    parsed["beta2"] = float(v[1])
                elif isinstance(v, (int, float, str, bool)):
                    parsed[k] = v
            return parsed
        except Exception as e:
            print(f"Warning: could not fetch config for {run_id}: {e}")
            return {"run_id": run_id}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        parsed_rows = list(executor.map(_fetch_config, unique_run_ids))

    df_parsed = pl.DataFrame(parsed_rows)
    return df.join(df_parsed, on="run_id", how="left")


def plot_ablation_grid(
    df_data: pl.DataFrame,
    title: str,
    row_specs: list[dict],
    pdf: PdfPages,
):
    """Render a multi-row grid of subplots for a single PDF page."""
    num_rows = max(len(row_specs), 1)

    fig, axes = plt.subplots(
        nrows=num_rows,
        ncols=4,
        figsize=(20.5, 4.2 * num_rows),
        dpi=300,
        squeeze=False,
    )

    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.995)
    cmap = plt.get_cmap("viridis")

    for row_idx, spec in enumerate(row_specs):
        ablation_param = spec["name"]
        fixed_params = spec.get("fixed", {})
        row_title = spec.get("row_title", f"Sweep: {ablation_param}")
        loss_func = spec.get("loss_func", "cross_entropy")
        model_name = spec.get("model_name", "mlp")

        df_row = df_data
        for k, v in fixed_params.items():
            if k in df_row.columns:
                if isinstance(v, float):
                    df_row = df_row.filter((pl.col(k) - v).abs() < 1e-8)
                else:
                    df_row = df_row.filter(pl.col(k) == v)

        if df_row.is_empty() or ablation_param not in df_row.columns:
            for c in range(4):
                axes[row_idx, c].text(
                    0.5,
                    0.5,
                    f"No data for {ablation_param}\nwith {fixed_params}",
                    ha="center",
                    va="center",
                    transform=axes[row_idx, c].transAxes,
                    fontsize=9,
                    color="#64748b",
                )
            continue

        unique_vals = (
            df_row.select(ablation_param)
            .drop_nulls()
            .unique()
            .sort(ablation_param)
            .to_series()
            .to_list()
        )
        if not unique_vals:
            continue

        colors = [cmap(i / max(len(unique_vals) - 1, 1)) for i in range(len(unique_vals))]

        for ax_idx in range(4):
            ax = axes[row_idx, ax_idx]

            for val, color in zip(unique_vals, colors, strict=False):
                if isinstance(val, float):
                    df_val = df_row.filter((pl.col(ablation_param) - val).abs() < 1e-8)
                else:
                    df_val = df_row.filter(pl.col(ablation_param) == val)

                if df_val.is_empty():
                    continue

                if ax_idx == 0:
                    metrics = [
                        ("eval/train/accuracy", "-", "Train Acc"),
                        ("eval/test/accuracy", "--", "Test Acc"),
                    ]
                elif ax_idx == 1:
                    train_loss_metric = f"eval/train/loss/{loss_func}/mean"
                    test_loss_metric = f"eval/test/loss/{loss_func}/mean"
                    metrics = [
                        (train_loss_metric, "-", "Train Loss"),
                        (test_loss_metric, "--", "Test Loss"),
                    ]
                elif ax_idx == 2:
                    last_layer = "fc3.weight" if model_name == "mlp" else "vit.heads.head.weight"
                    metrics = [
                        ("eval/weight_norm/total", "-", "Total Norm"),
                        (f"eval/weight_norm/{last_layer}", "--", "Last Layer"),
                    ]
                else:
                    # Column 4: Optimizer Dynamics (m_t, sqrt(v_t), and update norm)
                    available_metrics = set(df_val["metric_name"].unique())
                    mt_metric = (
                        "optim/global_mt_norm"
                        if "optim/global_mt_norm" in available_metrics
                        else "optim/global_momentum_buffer_norm"
                    )
                    vt_metric = (
                        "optim/global_sqrt_vt_norm"
                        if "optim/global_sqrt_vt_norm" in available_metrics
                        else "optim/global_sqrt_square_avg_norm"
                    )
                    metrics = [
                        (mt_metric, "-", "m_t"),
                        (vt_metric, "--", "sqrt_v_t"),
                        ("optim/global_update_norm", ":", "Update Norm"),
                    ]

                val_str = f"{val:.1e}" if (isinstance(val, float) and 0 < val < 1e-3) else str(val)
                labeled_for_val = False

                for metric_name, linestyle, _label_type in metrics:
                    df_metric = df_val.filter(pl.col("metric_name") == metric_name)
                    if df_metric.is_empty():
                        continue

                    # Average over seeds
                    df_agg = df_metric.group_by("step").agg(pl.col("value").mean()).sort("step")

                    steps = df_agg["step"].to_numpy()
                    vals = df_agg["value"].to_numpy()

                    # Filter out non-positive values for log scale plots to prevent warnings
                    if ax_idx in (1, 3):
                        valid_mask = np.isfinite(vals) & (vals > 0)
                        if not np.any(valid_mask):
                            continue
                        steps = steps[valid_mask]
                        vals = vals[valid_mask]

                    # Assign legend label only to first plotted line per parameter value
                    line_label = f"{ablation_param}={val_str}" if not labeled_for_val else None
                    labeled_for_val = True

                    ax.plot(
                        steps,
                        vals,
                        color=color,
                        linestyle=linestyle,
                        linewidth=1.6,
                        alpha=0.85,
                        label=line_label,
                    )

            if ax_idx in (1, 3):
                ax.set_yscale("log")

            ax.set_xlabel("Step", fontsize=9.5)
            ax.xaxis.set_major_formatter(FuncFormatter(format_step))

            if ax_idx == 0:
                ax.set_ylabel("Accuracy", fontsize=9.5)
                ax.set_title(f"{row_title} — Accuracy", fontsize=11, fontweight="semibold")
                ax.set_ylim(-0.02, 1.05)
                legend_title = "Solid: Train | Dashed: Test"
            elif ax_idx == 1:
                ax.set_ylabel("Loss (Log Scale)", fontsize=9.5)
                ax.set_title(f"{row_title} — Loss", fontsize=11, fontweight="semibold")
                legend_title = "Solid: Train | Dashed: Test"
            elif ax_idx == 2:
                ax.set_ylabel("Weight Norm ||θ||", fontsize=9.5)
                ax.set_title(f"{row_title} — Weight Norm", fontsize=11, fontweight="semibold")
                legend_title = "Solid: Total | Dashed: Last Layer"
            else:
                ax.set_ylabel("Norm (Log Scale)", fontsize=9.5)
                ax.set_title(
                    f"{row_title} — Optimizer Dynamics", fontsize=11, fontweight="semibold"
                )
                legend_title = "Solid: ‖mₜ‖ | Dashed: ‖√vₜ‖ | Dotted: ‖Δθ‖"

            ax.grid(True, linestyle=":", alpha=0.55)

            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(
                    handles,
                    labels,
                    fontsize=7.8,
                    loc="best",
                    framealpha=0.85,
                    title=legend_title,
                    title_fontsize=7.5,
                )

    plt.tight_layout(rect=[0.01, 0.02, 0.99, 0.97])
    pdf.savefig(fig)
    plt.close(fig)


@app.command()
def main(
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Destination path for generated PDF."),
    ] = None,
    tracking_uri: Annotated[
        str,
        typer.Option("--tracking-uri", "-u", help="MLflow tracking server URI."),
    ] = DEFAULT_TRACKING_URI,
    workers: Annotated[
        int,
        typer.Option("--workers", "-w", help="Number of worker threads for fetching configs."),
    ] = 8,
):
    """Generate the optimizer sweep ablation study PDF."""
    input_path = mlflow_export_cache_path("optimizer-sweep-v1")
    output_path = output or plots_optimizer_sweep_overview_path()

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Required sweep export not found at {input_path}. Please export metrics first."
        )

    print(f"Reading optimizer sweep data from {input_path} ...")
    df = pl.read_parquet(input_path)

    # 1. Filter strictly for Grokking models by name (excluding NO_GROK)
    print("Filtering Grokking models (~run_name.contains('NO_GROK')) ...")
    df = df.filter(~pl.col("run_name").str.contains("NO_GROK"))

    # 2. Extract and cache optimizer hyperparameters
    df = extract_optimizer_params(df, tracking_uri=tracking_uri, max_workers=workers)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Generating publication PDF at {output_path} ...")

    with PdfPages(output_path) as pdf:
        # --- PAGES 1-3: AdamW ---
        adamw_models = [
            ("mlp", "mse", "AdamW — MLP (MSE) Grokking"),
            ("mlp", "cross_entropy", "AdamW — MLP (Cross Entropy) Grokking"),
            ("vit_torchvision", "cross_entropy", "AdamW — ViT (Cross Entropy) Grokking"),
        ]
        adamw_ablations = [
            {
                "name": "eps",
                "fixed": {"beta1": 0.9, "beta2": 0.999},
                "row_title": "Ablation: eps (fixed β₁=0.9, β₂=0.999)",
            },
            {
                "name": "beta1",
                "fixed": {"eps": 1e-8, "beta2": 0.999},
                "row_title": "Ablation: beta1 (fixed ε=1e-8, β₂=0.999)",
            },
            {
                "name": "beta2",
                "fixed": {"eps": 1e-8, "beta1": 0.9},
                "row_title": "Ablation: beta2 (fixed ε=1e-8, β₁=0.9)",
            },
        ]
        for model_key, loss_key, page_title in adamw_models:
            print(f"  Plotting page: {page_title} ...")
            df_subset = df.filter(
                (pl.col("opt_name") == "AdamW")
                & (pl.col("params.model_name") == model_key)
                & (pl.col("params.loss_function") == loss_key)
            )
            specs = [
                {**ab, "model_name": model_key, "loss_func": loss_key} for ab in adamw_ablations
            ]
            plot_ablation_grid(df_subset, page_title, specs, pdf)

        # --- PAGES 4-6: RMSprop ---
        rmsprop_models = [
            ("mlp", "mse", "RMSprop — MLP (MSE) Grokking"),
            ("mlp", "cross_entropy", "RMSprop — MLP (Cross Entropy) Grokking"),
            ("vit_torchvision", "cross_entropy", "RMSprop — ViT (Cross Entropy) Grokking"),
        ]
        rmsprop_ablations = [
            {
                "name": "eps",
                "fixed": {"alpha": 0.99, "momentum": 0.0},
                "row_title": "Ablation: eps (fixed α=0.99, momentum=0.0)",
            },
            {
                "name": "alpha",
                "fixed": {"eps": 1e-8, "momentum": 0.0},
                "row_title": "Ablation: alpha (fixed ε=1e-8, momentum=0.0)",
            },
            {
                "name": "momentum",
                "fixed": {"eps": 1e-8, "alpha": 0.99},
                "row_title": "Ablation: momentum (fixed ε=1e-8, α=0.99)",
            },
        ]
        for model_key, loss_key, page_title in rmsprop_models:
            print(f"  Plotting page: {page_title} ...")
            df_subset = df.filter(
                (pl.col("opt_name") == "RMSprop")
                & (pl.col("params.model_name") == model_key)
                & (pl.col("params.loss_function") == loss_key)
            )
            specs = [
                {**ab, "model_name": model_key, "loss_func": loss_key} for ab in rmsprop_ablations
            ]
            plot_ablation_grid(df_subset, page_title, specs, pdf)

        # --- PAGE 7: SGD (All Models Overview) ---
        print("  Plotting page: SGD — All Models Overview (Momentum Sweep) ...")
        sgd_specs = [
            {
                "name": "momentum",
                "fixed": {"params.model_name": "mlp", "params.loss_function": "mse"},
                "model_name": "mlp",
                "loss_func": "mse",
                "row_title": "MLP (MSE) — Momentum Sweep",
            },
            {
                "name": "momentum",
                "fixed": {"params.model_name": "mlp", "params.loss_function": "cross_entropy"},
                "model_name": "mlp",
                "loss_func": "cross_entropy",
                "row_title": "MLP (CE) — Momentum Sweep",
            },
            {
                "name": "momentum",
                "fixed": {
                    "params.model_name": "vit_torchvision",
                    "params.loss_function": "cross_entropy",
                },
                "model_name": "vit_torchvision",
                "loss_func": "cross_entropy",
                "row_title": "ViT (CE) — Momentum Sweep",
            },
        ]
        df_sgd = df.filter(pl.col("opt_name") == "SGD")
        plot_ablation_grid(
            df_sgd,
            "SGD — Grokking Models Overview (Momentum Ablation)",
            sgd_specs,
            pdf,
        )

    print(f"\nSuccessfully generated 7-page overview plot at:\n  {output_path}")


if __name__ == "__main__":
    app()
