#!/usr/bin/env python3
"""plot_os_ablation.py.

Generates one image per run_name (e.g., GROK_MNIST_MSE_MLP).
Each image contains rows for different optimizer ablations:
1. Base run (AdamW, wd=0.01)
2. Base run without weight decay (AdamW, wd=0)
3. SGD (SGD, wd=0)
4. b1 and b2 = 0 (AdamW, b1=0, b2=0)

Columns:
(a) Train/test accuracy (left y-axis) and train/test loss (right y-axis).
(b) Optimizer dynamics over time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from matplotlib.ticker import FuncFormatter

app = typer.Typer(pretty_exceptions_enable=False)

def format_step(x: float, pos: int | None = None) -> str:
    if x <= 0:
        return "0"
    if x >= 1_000_000:
        return f"{x / 1_000_000:.1f}M" if (x % 1_000_000) != 0 else f"{int(x / 1_000_000)}M"
    if x >= 1_000:
        return f"{int(x / 1_000)}k" if x % 1_000 == 0 else f"{x / 1_000:.1f}k"
    return f"{int(x)}"

def parse_frequency(val: int | str) -> int:
    if isinstance(val, int):
        return val
    val_s = str(val).strip().lower()
    if val_s.endswith("k"):
        return int(float(val_s[:-1]) * 1000)
    if val_s.endswith("m"):
        return int(float(val_s[:-1]) * 1000000)
    return int(val_s)

def backfill_initial_nans(df: pd.DataFrame) -> pd.DataFrame:
    nan_mask = df["value"].isna()
    if not nan_mask.any():
        return df
    
    bfill_groups = [c for c in ["run_id", "run_name", "metric_name"] if c in df.columns]
    if not bfill_groups:
        bfill_groups = ["metric_name"]
    sort_cols = [c for c in ["run_id", "run_name", "metric_name", "step"] if c in df.columns]
    
    df = df.sort_values(sort_cols)
    df["value"] = df.groupby(bfill_groups)["value"].bfill()
    return df

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
    dpi: Annotated[
        int,
        typer.Option("--dpi", help="DPI resolution for exported raster PNG figures."),
    ] = 300,
) -> None:
    if input_path is None:
        input_path = Path("cache/optimizer-sweep-v1_mlflow_export.parquet")
    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet file not found at: {input_path}")
        
    if output_dir is None:
        output_dir = Path("cache/plots/os_ablation")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {input_path}...")
    df = pd.read_parquet(input_path)
    df = backfill_initial_nans(df)
    
    if "params.beta1" not in df.columns:
        df["params.beta1"] = None
    if "params.beta2" not in df.columns:
        df["params.beta2"] = None

    run_names = df["run_name"].dropna().unique()
    
    missing_configs = []

    for run_name in run_names:
        print(f"Plotting for {run_name}...")
        df_run = df[df["run_name"] == run_name]
        
        # Configurations to extract
        # 1. Base: AdamW, wd=0.01 (assuming default b1, b2 if None)
        # 2. No WD: AdamW, wd=0
        # 3. SGD: SGD
        # 4. b1=b2=0: AdamW, b1=0, b2=0
        
        configs = [
            {"title": "(I) Base (AdamW, wd=0.01)", "opt": "AdamW", "wd": "0.01", "b1": None, "b2": None},
            {"title": "(II) AdamW (wd=0)", "opt": "AdamW", "wd": "0", "b1": None, "b2": None},
            {"title": "(III) SGD", "opt": "SGD", "wd": "0", "b1": None, "b2": None},
            {"title": "(IV) AdamW (b1=0, b2=0)", "opt": "AdamW", "wd": "0.01", "b1": "0", "b2": "0"},
        ]
        
        fig, axes = plt.subplots(4, 2, figsize=(12, 16))
        
        for row_idx, cfg in enumerate(configs):
            mask = (df_run["params.optimizer"] == cfg["opt"])
            if cfg["wd"] is not None:
                mask &= (df_run["params.weight_decay"] == cfg["wd"])
            
            # Since beta might not be present or recorded as None in our data
            # For the base, if beta1/beta2 in data is None, we accept it.
            if cfg["b1"] is not None:
                mask &= (df_run["params.beta1"] == cfg["b1"])
            
            df_cfg = df_run[mask]
            
            ax_acc = axes[row_idx, 0]
            ax_dyn = axes[row_idx, 1]
            
            if df_cfg.empty:
                missing_configs.append(f"{run_name}: {cfg['title']}")
                ax_acc.text(0.5, 0.5, "Not available", ha="center", va="center")
                ax_dyn.text(0.5, 0.5, "Not available", ha="center", va="center")
            
            if row_idx == 0:
                ax_acc.set_title("(a) Train and Test Acc & Loss")
                ax_dyn.set_title("(b) Optimizer Dynamics")
            else:
                ax_acc.set_title(None)
                ax_dyn.set_title(None)
                
            ax_acc.annotate(
                cfg["title"],
                xy=(-0.30, 0.5),
                xycoords="axes fraction",
                fontsize=13,
                fontweight="bold",
                ha="center",
                va="center",
                rotation=90,
            )
            
            if df_cfg.empty:
                continue
            
            # Plot Col A: Acc and Loss
            ax_loss = ax_acc.twinx()
            metrics_a = {
                "eval/train/accuracy": ("Train Acc", ax_acc, "solid", "C0"),
                "eval/test/accuracy": ("Test Acc", ax_acc, "solid", "C1"),
            }
            # Add loss metrics based on loss function
            loss_func = df_cfg["params.loss_function"].iloc[0] if "params.loss_function" in df_cfg.columns else "cross_entropy"
            if loss_func == "mse":
                metrics_a["eval/train/loss/mse/mean"] = ("Train Loss", ax_loss, "dashed", "C0")
                metrics_a["eval/test/loss/mse/mean"] = ("Test Loss", ax_loss, "dashed", "C1")
            else:
                metrics_a["eval/train/loss/cross_entropy/mean"] = ("Train Loss", ax_loss, "dashed", "C0")
                metrics_a["eval/test/loss/cross_entropy/mean"] = ("Test Loss", ax_loss, "dashed", "C1")
                
            lines = []
            labels = []
            for m_name, (label, ax, ls, color) in metrics_a.items():
                m_df = df_cfg[df_cfg["metric_name"] == m_name]
                if not m_df.empty:
                    agg = m_df.groupby("step")["value"].mean().reset_index()
                    l, = ax.plot(agg["step"], agg["value"], label=label, linestyle=ls, color=color)
                    lines.append(l)
                    labels.append(label)
                    
            ax_acc.set_ylabel("Accuracy")
            ax_loss.set_ylabel("Loss")
            ax_loss.set_yscale("log")
            if lines:
                ax_acc.legend(lines, labels, loc="best")
            ax_acc.set_xlabel("Step")
            ax_acc.xaxis.set_major_formatter(FuncFormatter(format_step))

            # Plot Col B: Optimizer dynamics
            if cfg["opt"] == "AdamW":
                metrics_b = {
                    "optim/global_mt_norm": "1st Momentum",
                    "optim/global_sqrt_vt_norm": "2nd Momentum",
                    "optim/global_update_norm": "Update Step",
                    "optim/global_weight_reg_norm": "Weight Decay Term"
                }
            else: # SGD
                metrics_b = {
                    "optim/global_momentum_buffer_norm": "Momentum",
                    "optim/global_update_norm": "Update Step",
                    "optim/global_weight_reg_norm": "Weight Decay Term"
                }
                
            for m_name, label in metrics_b.items():
                m_df = df_cfg[df_cfg["metric_name"] == m_name]
                if not m_df.empty:
                    agg = m_df.groupby("step")["value"].mean().reset_index()
                    ax_dyn.plot(agg["step"], agg["value"], label=label)
            
            ax_dyn.set_yscale("log")
            ax_dyn.set_ylabel("Norm")
            ax_dyn.legend(loc="best")
            ax_dyn.set_xlabel("Step")
            ax_dyn.xaxis.set_major_formatter(FuncFormatter(format_step))

        plt.tight_layout()
        out_path = output_dir / f"{run_name}.pdf"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")

    print("\nMissing Configurations:")
    for m in missing_configs:
        print(m)

if __name__ == "__main__":
    app()
