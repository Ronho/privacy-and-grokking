#!/usr/bin/env python3
"""plot_hyper_sweep_ablation.py

Generates a 3x3 ablation plot for each model configuration from the hyper sweep data.
Rows:
- Row 0: Vary Train Size (fixed scale, decay)
- Row 1: Vary Init Scale (fixed size, decay)
- Row 2: Vary Weight Decay (fixed size, scale)

Columns:
- (a) Train and Test Accuracy over time
- (b) NC1 over time
- (c) Loss Attack AUC over time
- (d) 1% attack accuracy over time
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

# Canonical config mapping
CONFIG_LOSS_MAP = {
    "GROK_MNIST_CE_MLP": "cross_entropy",
    "NO_GROK_MNIST_CE_MLP": "cross_entropy",
    "GROK_MNIST_MSE_MLP": "mse",
    "NO_GROK_MNIST_MSE_MLP": "mse",
    "GROK_MNIST_CE_VIT": "cross_entropy",
    "NO_GROK_MNIST_CE_VIT": "cross_entropy",
    "GROK_MADD_CE_TRANSFORMER": "cross_entropy",
    "GROK_MADD_MSE_TRANSFORMER": "mse",
}

CONFIG_PRETTY_TITLES = {
    "GROK_MNIST_CE_MLP": "MLP (Cross Entropy, MNIST) [Grokking]",
    "NO_GROK_MNIST_CE_MLP": "MLP (Cross Entropy, MNIST) [Standard / No-Grok]",
    "GROK_MNIST_MSE_MLP": "MLP (MSE, MNIST) [Grokking]",
    "NO_GROK_MNIST_MSE_MLP": "MLP (MSE, MNIST) [Standard / No-Grok]",
    "GROK_MNIST_CE_VIT": "Vision Transformer (Cross Entropy, MNIST) [Grokking]",
    "NO_GROK_MNIST_CE_VIT": "Vision Transformer (Cross Entropy, MNIST) [Standard / No-Grok]",
    "GROK_MADD_CE_TRANSFORMER": "Modular Transformer (Cross Entropy, Modular Addition)",
    "GROK_MADD_MSE_TRANSFORMER": "Modular Transformer (MSE, Modular Addition)",
}

def extract_base_config_name(run_name: str) -> str:
    if not isinstance(run_name, str):
        return "Unknown"
    parts = run_name.split("_")
    if len(parts) >= 4:
        return "_".join(parts[3:])
    return run_name

def get_distinct_colors(n: int, cmap_name: str = "viridis", min_val: float = 0.02, max_val: float = 0.82) -> list:
    cmap = plt.get_cmap(cmap_name)
    if n <= 1:
        return [cmap((min_val + max_val) / 2)]
    return [cmap(x) for x in np.linspace(min_val, max_val, n)]

def format_param_val(val: float | int, param_type: str) -> str:
    if param_type == "train_size":
        return f"{int(round(float(val))):,}".replace(",", ".")
    elif param_type == "initialization_scale":
        fval = float(val)
        return f"{fval:.1f}" if fval % 1 else f"{int(fval)}"
    elif param_type == "weight_decay":
        fval = float(val)
        if fval >= 1.0:
            return f"{fval:.1f}" if fval % 1 else f"{int(fval)}"
        elif fval >= 0.01:
            return f"{fval:g}"
        else:
            return f"{fval:.0e}"
    return str(val)

def format_step(x: float, pos: int | None = None) -> str:
    if x <= 0: return "0"
    if x >= 1_000_000: return f"{x/1_000_000:.1f}M" if (x % 1_000_000) != 0 else f"{int(x/1_000_000)}M"
    if x >= 1_000: return f"{int(x/1_000)}k" if x % 1_000 == 0 else f"{x/1_000:.1f}k"
    return f"{int(x)}"

def plot_ablation_for_config(
    config_name: str,
    df_config: pd.DataFrame,
    output_dir: Path,
    log_x: bool = False,
    save_png: bool = True,
):
    print(f"Plotting {config_name} ablation...")
    
    def_scale = float(df_config["params.initialization_scale"].mode()[0])
    def_decay = float(df_config["params.weight_decay"].mode()[0])
    def_size = int(float(df_config["params.train_size"].mode()[0]))

    loss_type = CONFIG_LOSS_MAP.get(config_name, "cross_entropy")
    is_ce = loss_type == "cross_entropy"

    acc_train_key = "eval/train/accuracy"
    acc_test_key = "eval/test/accuracy"
    nc1_key = "eval/nc/nc1"

    loss_attack_auc_key = "eval/attack/ce_loss/auc" if is_ce else "eval/attack/mse_loss/auc"
    attack_tpr1_key = "eval/attack/ce_loss/tpr-at-fpr/1" if is_ce else "eval/attack/mse_loss/tpr-at-fpr/1"
    
    slices = [
        {
            "row_idx": 0,
            "title": f"(I) Variation: Train Size\n(Scale={format_param_val(def_scale, 'initialization_scale')}, Decay={format_param_val(def_decay, 'weight_decay')})",
            "param_col": "params.train_size",
            "param_type": "train_size",
            "param_label": "Train Size",
            "mask": (df_config["params.initialization_scale"].astype(float) == def_scale) & (df_config["params.weight_decay"].astype(float) == def_decay)
        },
        {
            "row_idx": 1,
            "title": f"(II) Variation: Init Scale\n(Size={format_param_val(def_size, 'train_size')}, Decay={format_param_val(def_decay, 'weight_decay')})",
            "param_col": "params.initialization_scale",
            "param_type": "initialization_scale",
            "param_label": "Init Scale",
            "mask": (df_config["params.weight_decay"].astype(float) == def_decay) & (df_config["params.train_size"].astype(int) == def_size)
        },
        {
            "row_idx": 2,
            "title": f"(III) Variation: Weight Decay\n(Size={format_param_val(def_size, 'train_size')}, Scale={format_param_val(def_scale, 'initialization_scale')})",
            "param_col": "params.weight_decay",
            "param_type": "weight_decay",
            "param_label": "Weight Decay",
            "mask": (df_config["params.initialization_scale"].astype(float) == def_scale) & (df_config["params.train_size"].astype(int) == def_size)
        },
    ]
    
    fig, axes = plt.subplots(3, 4, figsize=(24, 12), sharex="col")

    for sl in slices:
        row = sl["row_idx"]
        col_df = df_config[sl["mask"]]
        if col_df.empty:
            continue
        
        param_vals = sorted(col_df[sl["param_col"]].dropna().unique(), key=lambda x: float(x))
        colors = get_distinct_colors(len(param_vals), "viridis")
        color_map = {v: colors[i] for i, v in enumerate(param_vals)}

        ax_acc = axes[row, 0]
        ax_nc1 = axes[row, 1]
        ax_loss = axes[row, 2]
        ax_tpr = axes[row, 3]

        # Titles for the first row
        if row == 0:
            ax_acc.set_title("(a) Train and Test Accuracy", fontweight="bold")
            ax_nc1.set_title("(b) NC1", fontweight="bold")
            ax_loss.set_title("(c) Loss Attack AUC", fontweight="bold")
            ax_tpr.set_title("(d) 1% Attack TPR", fontweight="bold")

        # Row label
        ax_acc.annotate(sl["title"], xy=(-0.35, 0.5), xycoords="axes fraction",
                        fontsize=12, fontweight="bold", ha="center", va="center", rotation=90)

        for val in param_vals:
            run_df = col_df[col_df[sl["param_col"]] == val]
            c = color_map[val]

            # Accuracy
            t_acc = run_df[run_df["metric_name"] == acc_train_key].sort_values("step")
            v_acc = run_df[run_df["metric_name"] == acc_test_key].sort_values("step")
            if not t_acc.empty: ax_acc.plot(t_acc["step"], t_acc["value"], color=c, linestyle="-", lw=1.5)
            if not v_acc.empty: ax_acc.plot(v_acc["step"], v_acc["value"], color=c, linestyle="--", lw=1.5)

            # NC1
            nc1 = run_df[run_df["metric_name"] == nc1_key].sort_values("step")
            if not nc1.empty: ax_nc1.plot(nc1["step"], nc1["value"], color=c, linestyle="-", lw=1.5)

            # Loss Attack AUC
            loss_auc = run_df[run_df["metric_name"] == loss_attack_auc_key].sort_values("step")
            if not loss_auc.empty: ax_loss.plot(loss_auc["step"], loss_auc["value"], color=c, linestyle="-", lw=1.5)

            # 1% TPR (Column 4)
            tpr1 = run_df[run_df["metric_name"] == attack_tpr1_key].sort_values("step")
            if not tpr1.empty: ax_tpr.plot(tpr1["step"], tpr1["value"], color=c, linestyle="-.", lw=1.5, alpha=0.9)

        ax_acc.set_ylim(-0.02, 1.04)
        ax_loss.set_ylim(-0.02, 1.04)
        ax_tpr.set_ylim(-0.02, 1.04)
        ax_nc1.set_yscale("log")

        ax_acc.grid(True, ls="--", alpha=0.35)
        ax_nc1.grid(True, ls="--", alpha=0.35)
        ax_loss.grid(True, ls="--", alpha=0.35)
        ax_tpr.grid(True, ls="--", alpha=0.35)
        
        # Legends
        param_handles = [Line2D([0],[0], color=color_map[v], lw=2, label=format_param_val(v, sl["param_type"])) for v in param_vals]
        ax_acc.legend(handles=param_handles, title=sl["param_label"], fontsize=8, title_fontsize=9, loc="lower right")

        if row == 2:
            ax_acc.set_xlabel("Step")
            ax_nc1.set_xlabel("Step")
            ax_loss.set_xlabel("Step")
            ax_tpr.set_xlabel("Step")
            if not log_x:
                ax_acc.xaxis.set_major_formatter(FuncFormatter(format_step))
                ax_nc1.xaxis.set_major_formatter(FuncFormatter(format_step))
                ax_loss.xaxis.set_major_formatter(FuncFormatter(format_step))
                ax_tpr.xaxis.set_major_formatter(FuncFormatter(format_step))
            
        if log_x:
            ax_acc.set_xscale("log")
            ax_nc1.set_xscale("log")
            ax_loss.set_xscale("log")
            ax_tpr.set_xscale("log")



    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.93])
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_out = output_dir / f"hyper_sweep_ablation_{config_name}.pdf"
    plt.savefig(pdf_out, bbox_inches="tight")
    if save_png:
        plt.savefig(output_dir / f"hyper_sweep_ablation_{config_name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {pdf_out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, default="cache/hyper-sweep-v2_mlflow_export.parquet")
    parser.add_argument("--output-dir", "-o", type=str, default="cache/plots/hyper_sweep_ablation")
    parser.add_argument("--configs", nargs="+", default=["all"])
    parser.add_argument("--models", nargs="+", default=["all"])
    parser.add_argument("--log-x", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return

    df = pd.read_parquet(input_path)
    df["base_config"] = df["run_name"].apply(extract_base_config_name)
    all_configs = sorted(df["base_config"].unique())

    if args.models and "all" not in args.models:
        df = df[df["params.model_name"].isin(args.models)]
        all_configs = sorted(df["base_config"].unique())
    if args.configs and "all" not in args.configs:
        all_configs = [c for c in all_configs if c in args.configs]

    if not all_configs:
        print("No configurations found to plot.")
        return

    for cfg in all_configs:
        cfg_df = df[df["base_config"] == cfg]
        plot_ablation_for_config(cfg, cfg_df, Path(args.output_dir), log_x=args.log_x)

if __name__ == "__main__":
    main()
