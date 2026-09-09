"""Visualizes training and attack dynamics over time (steps) across models and hyperparameter slices
(weight decay, initialization scale, and train size) from hyper-sweep-v2.

Author: Antigravity
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

# Canonical configuration display names and target loss functions
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
    """Extracts base configuration name from run_name like '8.0_0.01_1000_GROK_MNIST_CE_MLP'."""
    if not isinstance(run_name, str):
        return "Unknown"
    parts = run_name.split("_")
    if len(parts) >= 4:
        return "_".join(parts[3:])
    return run_name


def get_distinct_colors(
    n: int, cmap_name: str = "viridis", min_val: float = 0.02, max_val: float = 0.82
) -> list:
    """Generates n visually distinct colors along a colormap, avoiding glaring end colors like neon yellow."""
    cmap = plt.get_cmap(cmap_name)
    if n <= 1:
        return [cmap((min_val + max_val) / 2)]
    return [cmap(x) for x in np.linspace(min_val, max_val, n)]


def format_param_val(val: float | int, param_type: str) -> str:
    """Formats hyperparameter values cleanly for labels."""
    if param_type == "train_size":
        ival = int(round(float(val)))
        return f"{ival:,}".replace(",", ".")
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


def plot_config_dynamics(
    config_name: str,
    df_config: pd.DataFrame,
    output_dir: Path,
    log_x: bool = False,
    include_nc: bool = True,
    save_png: bool = True,
    pdf_writer: PdfPages | None = None,
) -> str | None:
    """Generates the multi-panel dynamics plot (3 columns x 3-4 rows) for a given config."""
    print(f"\n[Plotting] {config_name} ({len(df_config['run_id'].unique())} runs)...")

    # 1. Determine baseline parameters (mode values)
    def_scale = float(df_config["params.initialization_scale"].mode()[0])
    def_decay = float(df_config["params.weight_decay"].mode()[0])
    def_size = int(float(df_config["params.train_size"].mode()[0]))

    loss_type = CONFIG_LOSS_MAP.get(config_name, "cross_entropy")
    is_ce = loss_type == "cross_entropy"

    # Define key metrics
    acc_train_key = "eval/train/accuracy"
    acc_test_key = "eval/test/accuracy"

    loss_train_key = (
        "eval/train/loss/cross_entropy/mean" if is_ce else "eval/train/loss/mse/mean"
    )
    loss_test_key = (
        "eval/test/loss/cross_entropy/mean" if is_ce else "eval/test/loss/mse/mean"
    )

    attack_member_key = "eval/attack/ce_loss/auc" if is_ce else "eval/attack/mse_loss/auc"
    attack_canary_key = (
        "eval/attack/canary_ce_loss/auc" if is_ce else "eval/attack/canary_mse_loss/auc"
    )

    attack_member_tpr1_key = (
        "eval/attack/ce_loss/tpr-at-fpr/1" if is_ce else "eval/attack/mse_loss/tpr-at-fpr/1"
    )
    attack_canary_tpr1_key = (
        "eval/attack/canary_ce_loss/tpr-at-fpr/1"
        if is_ce else "eval/attack/canary_mse_loss/tpr-at-fpr/1"
    )
    loss_abbr = "CE" if is_ce else "MSE"
    nc_key = "eval/nc/nc1"

    # Define the 3 parameter slices
    # Slice 1: vary train_size (fixed scale, decay)
    s1_mask = (df_config["params.initialization_scale"].astype(float) == def_scale) & (
        df_config["params.weight_decay"].astype(float) == def_decay
    )
    # Slice 2: vary init_scale (fixed decay, size)
    s2_mask = (df_config["params.weight_decay"].astype(float) == def_decay) & (
        df_config["params.train_size"].astype(int) == def_size
    )
    # Slice 3: vary weight_decay (fixed scale, size)
    s3_mask = (df_config["params.initialization_scale"].astype(float) == def_scale) & (
        df_config["params.train_size"].astype(int) == def_size
    )

    slices = [
        {
            "col_idx": 0,
            "title": f"Variation: Train Size\n(Init Scale = {format_param_val(def_scale, 'initialization_scale')}, Weight Decay = {format_param_val(def_decay, 'weight_decay')})",
            "param_col": "params.train_size",
            "param_type": "train_size",
            "param_label": "Train Size",
            "mask": s1_mask,
            "cmap": "viridis",
        },
        {
            "col_idx": 1,
            "title": f"Variation: Init Scale\n(Train Size = {format_param_val(def_size, 'train_size')}, Weight Decay = {format_param_val(def_decay, 'weight_decay')})",
            "param_col": "params.initialization_scale",
            "param_type": "initialization_scale",
            "param_label": "Init Scale",
            "mask": s2_mask,
            "cmap": "viridis",
        },
        {
            "col_idx": 2,
            "title": f"Variation: Weight Decay\n(Train Size = {format_param_val(def_size, 'train_size')}, Init Scale = {format_param_val(def_scale, 'initialization_scale')})",
            "param_col": "params.weight_decay",
            "param_type": "weight_decay",
            "param_label": "Weight Decay",
            "mask": s3_mask,
            "cmap": "viridis",
        },
    ]

    num_rows = 5 if include_nc else 4
    fig, axes = plt.subplots(
        num_rows, 3, figsize=(20, 3.8 * num_rows), sharex="col", squeeze=False
    )

    pretty_title = CONFIG_PRETTY_TITLES.get(config_name, config_name)
    fig.suptitle(
        f"Trainings- & Angriffsdynamiken über Trainingsschritte\n{pretty_title}",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    for sl in slices:
        col_idx = sl["col_idx"]
        col_df = df_config[sl["mask"]]
        if col_df.empty:
            continue

        param_vals = sorted(
            col_df[sl["param_col"]].dropna().unique(), key=lambda x: float(x)
        )
        colors = get_distinct_colors(len(param_vals), sl["cmap"])
        param_color_map = {v: colors[i] for i, v in enumerate(param_vals)}

        # Setup column headers
        axes[0, col_idx].set_title(sl["title"], fontsize=12, fontweight="bold", pad=12)

        # ----------------- Row 0: Accuracy -----------------
        ax_acc = axes[0, col_idx]
        for val in param_vals:
            run_df = col_df[col_df[sl["param_col"]] == val]
            c = param_color_map[val]
            label_val = format_param_val(val, sl["param_type"])

            train_acc = run_df[run_df["metric_name"] == acc_train_key].sort_values("step")
            test_acc = run_df[run_df["metric_name"] == acc_test_key].sort_values("step")

            if not train_acc.empty:
                ax_acc.plot(
                    train_acc["step"],
                    train_acc["value"],
                    color=c,
                    linestyle="-",
                    linewidth=1.7,
                    alpha=0.85,
                    label=f"{label_val}",
                )
            if not test_acc.empty:
                ax_acc.plot(
                    test_acc["step"],
                    test_acc["value"],
                    color=c,
                    linestyle="--",
                    linewidth=1.7,
                    alpha=0.9,
                )

        ax_acc.set_ylim(-0.02, 1.04)
        ax_acc.grid(True, linestyle="--", alpha=0.35)
        if col_idx == 0:
            ax_acc.set_ylabel("Accuracy\n(solid=Train, dash=Test)", fontsize=11, fontweight="semibold")

        # Color legend for parameter values
        param_handles = [
            Line2D([0], [0], color=param_color_map[v], lw=2.5, label=format_param_val(v, sl["param_type"]))
            for v in param_vals
        ]
        leg_param = ax_acc.legend(
            handles=param_handles,
            title=sl["param_label"],
            fontsize=8.5,
            title_fontsize=9,
            loc="lower right",
            framealpha=0.9,
        )
        ax_acc.add_artist(leg_param)

        # ----------------- Row 1: Loss -----------------
        ax_loss = axes[1, col_idx]
        for val in param_vals:
            run_df = col_df[col_df[sl["param_col"]] == val]
            c = param_color_map[val]

            train_loss = run_df[run_df["metric_name"] == loss_train_key].sort_values("step")
            test_loss = run_df[run_df["metric_name"] == loss_test_key].sort_values("step")

            if not train_loss.empty:
                ax_loss.plot(
                    train_loss["step"],
                    train_loss["value"],
                    color=c,
                    linestyle="-",
                    linewidth=1.6,
                    alpha=0.85,
                )
            if not test_loss.empty:
                ax_loss.plot(
                    test_loss["step"],
                    test_loss["value"],
                    color=c,
                    linestyle="--",
                    linewidth=1.6,
                    alpha=0.9,
                )

        ax_loss.set_yscale("log")
        ax_loss.grid(True, linestyle="--", alpha=0.35)
        if col_idx == 0:
            ax_loss.set_ylabel(f"Loss ({loss_abbr}, Log Scale)\n(solid=Train, dash=Test)", fontsize=11, fontweight="semibold")

        # ----------------- Row 2: Attack AUC -----------------
        ax_atk = axes[2, col_idx]
        for val in param_vals:
            run_df = col_df[col_df[sl["param_col"]] == val]
            c = param_color_map[val]

            atk_member = run_df[run_df["metric_name"] == attack_member_key].sort_values("step")
            atk_canary = run_df[run_df["metric_name"] == attack_canary_key].sort_values("step")

            if not atk_member.empty:
                ax_atk.plot(
                    atk_member["step"],
                    atk_member["value"],
                    color=c,
                    linestyle="-",
                    linewidth=1.8,
                    alpha=0.85,
                )
            if not atk_canary.empty:
                ax_atk.plot(
                    atk_canary["step"],
                    atk_canary["value"],
                    color=c,
                    linestyle="--",
                    linewidth=1.7,
                    alpha=0.85,
                )

        ax_atk.axhline(0.5, color="#6b7280", linestyle="-.", linewidth=1.1, label="Random Guess (0.5)")
        ax_atk.set_ylim(0.40, 1.02)
        ax_atk.grid(True, linestyle="--", alpha=0.35)
        if col_idx == 0:
            ax_atk.set_ylabel(
                f"Attack AUC [{loss_abbr} used in training]\n(solid=Member, dash=Canary)",
                fontsize=10,
                fontweight="semibold",
            )

        # ----------------- Row 3: TPR @ 1% FPR -----------------
        ax_tpr = axes[3, col_idx]
        for val in param_vals:
            run_df = col_df[col_df[sl["param_col"]] == val]
            c = param_color_map[val]

            tpr_member = run_df[run_df["metric_name"] == attack_member_tpr1_key].sort_values("step")
            tpr_canary = run_df[run_df["metric_name"] == attack_canary_tpr1_key].sort_values("step")

            if not tpr_member.empty:
                ax_tpr.plot(
                    tpr_member["step"],
                    tpr_member["value"],
                    color=c,
                    linestyle="-",
                    linewidth=1.8,
                    alpha=0.85,
                )
            if not tpr_canary.empty:
                ax_tpr.plot(
                    tpr_canary["step"],
                    tpr_canary["value"],
                    color=c,
                    linestyle="--",
                    linewidth=1.7,
                    alpha=0.85,
                )

        ax_tpr.axhline(0.01, color="#6b7280", linestyle="-.", linewidth=1.1, label="Random Guess (0.01)")
        ax_tpr.set_ylim(-0.02, 1.04)
        ax_tpr.grid(True, linestyle="--", alpha=0.35)
        if col_idx == 0:
            ax_tpr.set_ylabel(
                f"TPR @ 1% FPR [{loss_abbr} used in training]\n(solid=Member, dash=Canary)",
                fontsize=10,
                fontweight="semibold",
            )

        # ----------------- Row 4 (Optional): Neural Collapse -----------------
        if include_nc:
            ax_nc = axes[4, col_idx]
            has_nc = False
            for val in param_vals:
                run_df = col_df[col_df[sl["param_col"]] == val]
                c = param_color_map[val]
                nc_data = run_df[run_df["metric_name"] == nc_key].sort_values("step")
                if not nc_data.empty:
                    has_nc = True
                    ax_nc.plot(
                        nc_data["step"],
                        nc_data["value"],
                        color=c,
                        linestyle="-",
                        linewidth=1.6,
                        alpha=0.85,
                    )
            ax_nc.grid(True, linestyle="--", alpha=0.35)
            if col_idx == 0:
                ax_nc.set_ylabel("Neural Collapse (NC1)", fontsize=11, fontweight="semibold")
            if not has_nc:
                ax_nc.text(0.5, 0.5, "NC1 not logged", ha="center", va="center", transform=ax_nc.transAxes, color="gray")

        # Bottom axis formatting
        bottom_ax = axes[-1, col_idx]
        bottom_ax.set_xlabel("Training Steps", fontsize=11, fontweight="semibold")
        if log_x:
            for r in range(num_rows):
                axes[r, col_idx].set_xscale("log")
        else:
            for r in range(num_rows):
                axes[r, col_idx].set_xlim(left=0, right=150000)

    # Global line style legend at the top
    style_handles = [
        Line2D([0], [0], color="#1f2937", lw=2, ls="-", label="Train / Member (solid)"),
        Line2D([0], [0], color="#1f2937", lw=2, ls="--", label="Test / Canary (dashed)"),
    ]
    fig.legend(
        handles=style_handles,
        loc="upper right",
        ncol=2,
        bbox_to_anchor=(0.99, 0.997),
        fontsize=9.5,
        framealpha=0.85,
    )

    plt.tight_layout(rect=[0.02, 0.02, 0.99, 0.97])

    out_file = output_dir / f"{config_name}_dynamics.pdf"
    plt.savefig(out_file, bbox_inches="tight")
    if pdf_writer is not None:
        pdf_writer.savefig(fig, bbox_inches="tight")
    if save_png:
        out_png = output_dir / f"{config_name}_dynamics.png"
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_file}")
    return str(out_file)


def main():
    parser = argparse.ArgumentParser(
        description="Plot training and attack dynamics over time across hyperparameter slices from hyper-sweep-v2."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="cache/hyper-sweep-v2_mlflow_export.parquet",
        help="Path to hyper-sweep MLflow export parquet file (default: cache/hyper-sweep-v2_mlflow_export.parquet)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="plots/hyper_sweep_dynamics",
        help="Directory to save the plots (default: plots/hyper_sweep_dynamics)",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["all"],
        help="Specific configs to plot (default: 'all', or list e.g. GROK_MNIST_CE_MLP GROK_MNIST_CE_VIT)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Filter by model family (default: 'all', or 'mlp', 'vit_torchvision', 'modular_transformer')",
    )
    parser.add_argument(
        "--log-x",
        action="store_true",
        help="Use logarithmic scale for training step axis (default: linear scale 0..150000)",
    )
    parser.add_argument(
        "--no-nc",
        action="store_true",
        help="Do not include the Neural Collapse (NC1) row in the plots (default: included)",
    )
    parser.add_argument(
        "--save-png",
        action="store_true",
        default=True,
        help="Save high-resolution 300 DPI PNG alongside the PDF (default: True)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' not found.")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from '{input_path}'...")
    needed_metrics = [
        "eval/train/accuracy",
        "eval/test/accuracy",
        "eval/train/loss/cross_entropy/mean",
        "eval/test/loss/cross_entropy/mean",
        "eval/train/loss/mse/mean",
        "eval/test/loss/mse/mean",
        "eval/attack/ce_loss/auc",
        "eval/attack/mse_loss/auc",
        "eval/attack/canary_ce_loss/auc",
        "eval/attack/canary_mse_loss/auc",
        "eval/attack/ce_loss/tpr-at-fpr/1",
        "eval/attack/mse_loss/tpr-at-fpr/1",
        "eval/attack/canary_ce_loss/tpr-at-fpr/1",
        "eval/attack/canary_mse_loss/tpr-at-fpr/1",
        "eval/nc/nc1",
    ]

    needed_cols = [
        "run_id",
        "run_name",
        "metric_name",
        "value",
        "step",
        "params.initialization_scale",
        "params.weight_decay",
        "params.train_size",
        "params.model_name",
        "params.loss_function",
    ]

    df = pd.read_parquet(
        input_path,
        columns=needed_cols,
        filters=[("metric_name", "in", needed_metrics)],
    )

    df["base_config"] = df["run_name"].apply(extract_base_config_name)
    all_configs = sorted(df["base_config"].unique())
    print(f"Found configs in export: {all_configs}")

    if args.models and "all" not in args.models:
        df = df[df["params.model_name"].isin(args.models)]
        all_configs = sorted(df["base_config"].unique())

    if args.configs and "all" not in args.configs:
        all_configs = [c for c in all_configs if c in args.configs]

    if not all_configs:
        print("No matching configurations found to plot.")
        return

    generated_pdfs = []
    combined_pdf_path = out_dir / "all_models_hyper_sweep_dynamics.pdf"
    print(f"\nCompiling all configurations into combined PDF '{combined_pdf_path}'...")
    with PdfPages(combined_pdf_path) as combined_pdf:
        for cfg_name in all_configs:
            cfg_df = df[df["base_config"] == cfg_name]
            pdf_path = plot_config_dynamics(
                config_name=cfg_name,
                df_config=cfg_df,
                output_dir=out_dir,
                log_x=args.log_x,
                include_nc=not args.no_nc,
                save_png=args.save_png,
                pdf_writer=combined_pdf,
            )
            if pdf_path:
                generated_pdfs.append(pdf_path)

    print(f"\n[Done] Successfully generated {len(generated_pdfs)} configuration plots and combined PDF in '{out_dir}'.")


if __name__ == "__main__":
    main()
