"""Generates 2D hyperparameter heatmaps from hyper-sweep-v2.

Layout:
  - 3 Columns: Pairwise combinations of parameters (weight decay, init scale, train size) with 1 fixed default.
  - 4-5 Rows: Key training and attack metrics (Test Accuracy, Test Loss, Attack AUC, TPR @ 1% FPR, Neural Collapse).
  - Target Timepoints: Specific training epochs (or steps) from STATIC_EVAL_POINTS (early vs late / grokked)
    or custom epochs specified via CLI.

Author: Antigravity
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

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

# Default static evaluation checkpoints (analogous to info_rmia.py)
STATIC_EVAL_POINTS = {
    "MNIST_CE_MLP": {"mode": "epoch", "points": [80, 400]},
    "MNIST_MSE_MLP": {"mode": "epoch", "points": [40, 400]},
    "MNIST_CE_VIT": {"mode": "epoch", "points": [80, 500]},
    "MADD_CE_TRANSFORMER": {"mode": "step", "points": [5000, 50000]},
    "MADD_MSE_TRANSFORMER": {"mode": "step", "points": [4000, 50000]},
}


def extract_base_config_name(run_name: str) -> str:
    """Extracts base configuration name from run_name like '8.0_0.01_1000_GROK_MNIST_CE_MLP'."""
    if not isinstance(run_name, str):
        return "Unknown"
    parts = run_name.split("_")
    if len(parts) >= 4:
        return "_".join(parts[3:])
    return run_name


def get_eval_spec_for_config(config_name: str) -> dict:
    """Finds matching static eval points spec for a configuration."""
    for key, spec in STATIC_EVAL_POINTS.items():
        if key in config_name:
            return spec
    # Fallback default
    return {"mode": "epoch", "points": [80, 400]}


def get_blue_cmap(n: int = 256) -> mcolors.Colormap:
    """Creates a colormap varying intensity of the standard matplotlib blue (#1f77b4)."""
    # From very light blue to standard C0 blue
    return mcolors.LinearSegmentedColormap.from_list("custom_blue", ["#ebf3f9", "#1f77b4"], N=n)


def format_param_val(val: float | int, param_type: str) -> str:
    """Formats hyperparameter values cleanly for axes ticks and titles."""
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


def format_metric_value(val: float | None, metric_key: str) -> str:
    """Formats cell values for heatmap display."""
    if val is None or np.isnan(val):
        return "—"
    if "loss" in metric_key:
        if val >= 10.0:
            return f"{val:.1f}"
        elif val >= 0.01:
            return f"{val:.3f}"
        else:
            return f"{val:.1e}"
    elif "nc1" in metric_key:
        if val >= 10.0:
            return f"{val:.1f}"
        elif val >= 0.01:
            return f"{val:.2f}"
        else:
            return f"{val:.1e}"
    else:
        # Accuracy, AUC, TPR
        return f"{val:.2f}"


def get_metric_value_at_checkpoint(
    run_df: pd.DataFrame,
    metric_name: str,
    target_point: float | int,
    mode: str = "epoch",
) -> float | None:
    """Extracts the metric value closest to the target epoch or step for a single run."""
    if run_df.empty:
        return None

    if mode == "epoch":
        ep_rows = run_df[run_df["metric_name"] == "epoch"].sort_values("step")
        if ep_rows.empty:
            # Fallback to step if epoch not present
            target_step = int(target_point)
            chosen_step = target_step
        else:
            diffs = (ep_rows["value"] - target_point).abs()
            chosen_step = ep_rows.iloc[diffs.argmin()]["step"]
    else:
        # Mode is step
        chosen_step = int(target_point)

    m_rows = run_df[run_df["metric_name"] == metric_name]
    if m_rows.empty:
        return None

    # Exact match on chosen_step
    exact = m_rows[m_rows["step"] == chosen_step]
    if not exact.empty:
        return float(exact["value"].iloc[0])

    # Nearest step match for this metric
    diffs = (m_rows["step"] - chosen_step).abs()
    best_row = m_rows.iloc[diffs.argmin()]
    return float(best_row["value"])


def plot_config_heatmaps_at_checkpoint(
    config_name: str,
    df_config: pd.DataFrame,
    output_dir: Path,
    target_point: float | int,
    mode: str = "epoch",
    include_nc: bool = True,
    annot: bool = True,
    save_png: bool = True,
    pdf_writer: PdfPages | None = None,
) -> str | None:
    """Generates the 3 columns x 4-5 rows 2D heatmap panel for a configuration at a given epoch/step."""
    loss_type = CONFIG_LOSS_MAP.get(config_name, "cross_entropy")
    is_ce = loss_type == "cross_entropy"
    loss_abbr = "CE" if is_ce else "MSE"

    # Baseline default values
    def_scale = float(df_config["params.initialization_scale"].mode()[0])
    def_decay = float(df_config["params.weight_decay"].mode()[0])
    def_size = int(float(df_config["params.train_size"].mode()[0]))

    # Define metric rows
    acc_metric = "eval/test/accuracy"
    loss_metric = "eval/test/loss/cross_entropy/mean" if is_ce else "eval/test/loss/mse/mean"
    auc_metric = "eval/attack/ce_loss/auc" if is_ce else "eval/attack/mse_loss/auc"
    tpr_metric = "eval/attack/ce_loss/tpr-at-fpr/1" if is_ce else "eval/attack/mse_loss/tpr-at-fpr/1"
    nc_metric = "eval/nc/nc1"

    metric_rows = [
        {"key": acc_metric, "name": "Test Accuracy", "is_loss": False, "fixed_range": (0.0, 1.0)},
        {"key": loss_metric, "name": f"Test Loss [{loss_abbr}]", "is_loss": True, "fixed_range": None},
        {"key": auc_metric, "name": f"Attack AUC [{loss_abbr} used in training]", "is_loss": False, "fixed_range": (0.45, 1.0)},
        {"key": tpr_metric, "name": f"TPR @ 1% FPR [{loss_abbr} used in training]", "is_loss": False, "fixed_range": (0.0, 1.0)},
    ]

    # Check if NC1 has any data in this config
    has_nc = not df_config[df_config["metric_name"] == nc_metric].empty
    if include_nc and has_nc:
        metric_rows.append({"key": nc_metric, "name": "Neural Collapse (NC1)", "is_loss": False, "fixed_range": None})

    num_rows = len(metric_rows)

    # Define the 3 parameter pairwise columns
    # Col 0: X = weight_decay, Y = initialization_scale (fixed train_size)
    # Col 1: X = weight_decay, Y = train_size (fixed initialization_scale)
    # Col 2: X = initialization_scale, Y = train_size (fixed weight_decay)
    slices = [
        {
            "col_idx": 0,
            "title": f"Init Scale vs. Weight Decay\n(Train Size = {format_param_val(def_size, 'train_size')})",
            "x_param": "params.weight_decay",
            "x_type": "weight_decay",
            "x_label": "Weight Decay",
            "y_param": "params.initialization_scale",
            "y_type": "initialization_scale",
            "y_label": "Init Scale",
            "mask": (df_config["params.train_size"].astype(int) == def_size),
        },
        {
            "col_idx": 1,
            "title": f"Train Size vs. Weight Decay\n(Init Scale = {format_param_val(def_scale, 'initialization_scale')})",
            "x_param": "params.weight_decay",
            "x_type": "weight_decay",
            "x_label": "Weight Decay",
            "y_param": "params.train_size",
            "y_type": "train_size",
            "y_label": "Train Size",
            "mask": (df_config["params.initialization_scale"].astype(float) == def_scale),
        },
        {
            "col_idx": 2,
            "title": f"Train Size vs. Init Scale\n(Weight Decay = {format_param_val(def_decay, 'weight_decay')})",
            "x_param": "params.initialization_scale",
            "x_type": "initialization_scale",
            "x_label": "Init Scale",
            "y_param": "params.train_size",
            "y_type": "train_size",
            "y_label": "Train Size",
            "mask": (df_config["params.weight_decay"].astype(float) == def_decay),
        },
    ]

    # Precompute cell values for all columns and rows
    # slice_matrices[row_idx][col_idx] = (matrix, x_vals, y_vals)
    slice_matrices = [[None for _ in range(3)] for _ in range(num_rows)]

    # Group runs for fast lookup
    runs_by_id = {r_id: r_df for r_id, r_df in df_config.groupby("run_id")}

    for sl in slices:
        c_idx = sl["col_idx"]
        col_df = df_config[sl["mask"]]
        if col_df.empty:
            continue

        runs_info = col_df[["run_id", sl["x_param"], sl["y_param"]]].drop_duplicates("run_id")
        x_vals = sorted(runs_info[sl["x_param"]].dropna().unique(), key=lambda v: float(v))
        y_vals = sorted(runs_info[sl["y_param"]].dropna().unique(), key=lambda v: float(v))

        x_to_idx = {v: i for i, v in enumerate(x_vals)}
        y_to_idx = {v: i for i, v in enumerate(y_vals)}

        for r_idx, m_info in enumerate(metric_rows):
            mat = np.full((len(y_vals), len(x_vals)), np.nan)
            for _, r_row in runs_info.iterrows():
                r_id = r_row["run_id"]
                x_v = r_row[sl["x_param"]]
                y_v = r_row[sl["y_param"]]
                if r_id in runs_by_id:
                    val = get_metric_value_at_checkpoint(
                        run_df=runs_by_id[r_id],
                        metric_name=m_info["key"],
                        target_point=target_point,
                        mode=mode,
                    )
                    if val is not None:
                        mat[y_to_idx[y_v], x_to_idx[x_v]] = val

            slice_matrices[r_idx][c_idx] = (mat, x_vals, y_vals)

    # Set up Matplotlib GridSpec
    fig = plt.figure(figsize=(19, 3.8 * num_rows))
    gs = fig.add_gridspec(
        num_rows,
        4,
        width_ratios=[1.0, 1.0, 1.0, 0.035],
        wspace=0.28,
        hspace=0.36,
    )

    cmap = get_blue_cmap()
    pretty_title = CONFIG_PRETTY_TITLES.get(config_name, config_name)
    mode_str = "Epoche" if mode == "epoch" else "Trainingsschritt"
    fig.suptitle(
        f"Hyperparameter-Heatmaps: {pretty_title}\nAuswertungszeitpunkt: {mode_str} = {target_point:,}".replace(",", "."),
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    for r_idx, m_info in enumerate(metric_rows):
        # 1. Determine common color normalization across the 3 columns for this row
        all_row_vals = []
        for c_idx in range(3):
            entry = slice_matrices[r_idx][c_idx]
            if entry is not None:
                mat = entry[0]
                valid_vals = mat[~np.isnan(mat)]
                if len(valid_vals) > 0:
                    all_row_vals.extend(valid_vals.tolist())

        if not all_row_vals:
            continue

        if m_info["fixed_range"] is not None:
            vmin, vmax = m_info["fixed_range"]
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        elif m_info["is_loss"]:
            min_pos = min([v for v in all_row_vals if v > 0] or [1e-4])
            max_pos = max(all_row_vals)
            norm = mcolors.LogNorm(vmin=max(min_pos, 1e-4), vmax=max(max_pos, 1e-3))
        else:
            vmin = min(all_row_vals)
            vmax = max(all_row_vals)
            if np.isclose(vmin, vmax):
                vmax = vmin + 1.0
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        im_last = None

        # 2. Plot the 3 subplots in this row
        for c_idx in range(3):
            ax = fig.add_subplot(gs[r_idx, c_idx])
            entry = slice_matrices[r_idx][c_idx]
            sl = slices[c_idx]

            if entry is None:
                ax.axis("off")
                continue

            mat, x_vals, y_vals = entry
            x_vals_float = [float(v) for v in x_vals]
            y_vals_float = [float(v) for v in y_vals]
            
            X, Y = np.meshgrid(x_vals_float, y_vals_float)
            X_flat, Y_flat, Z_flat = X.flatten(), Y.flatten(), mat.flatten()
            
            mask = ~np.isnan(Z_flat)
            if np.any(mask):
                im = ax.scatter(
                    X_flat[mask],
                    Y_flat[mask],
                    c=Z_flat[mask],
                    s=300,
                    cmap=cmap,
                    norm=norm,
                    edgecolors="black",
                    linewidth=0.5,
                    alpha=0.95
                )
                im_last = im

            ax.set_xscale("log")
            ax.set_yscale("log")

            # Titles on top row
            if r_idx == 0:
                ax.set_title(sl["title"], fontsize=12, fontweight="bold", pad=12)

            # Axis labels
            ax.set_xlabel(sl["x_label"], fontsize=10.5, fontweight="semibold")
            ax.set_ylabel(sl["y_label"], fontsize=10.5, fontweight="semibold")

            # Ticks
            ax.set_xticks(x_vals_float)
            ax.set_xticklabels([format_param_val(v, sl["x_type"]) for v in x_vals], rotation=30, ha="right", fontsize=9)
            ax.set_yticks(y_vals_float)
            ax.set_yticklabels([format_param_val(v, sl["y_type"]) for v in y_vals], fontsize=9)
            ax.minorticks_off()

            # Annotate cells
            if annot:
                for yi in range(len(y_vals)):
                    for xi in range(len(x_vals)):
                        cell_v = mat[yi, xi]
                        if np.isnan(cell_v):
                            continue
                            
                        text_str = format_metric_value(cell_v, m_info["key"])
                        normed_val = norm(cell_v)
                        normed_clip = np.clip(normed_val, 0.0, 1.0)
                        text_color = "#ffffff" if normed_clip > 0.5 else "#09090b"

                        ax.text(
                            x_vals_float[xi],
                            y_vals_float[yi],
                            text_str,
                            ha="center",
                            va="center",
                            color=text_color,
                            fontsize=8.5,
                            fontweight="semibold",
                        )

        # 3. Add row colorbar in the 4th column (gs[r_idx, 3])
        cax = fig.add_subplot(gs[r_idx, 3])
        if im_last is not None:
            cbar = fig.colorbar(im_last, cax=cax)
            cbar.set_label(m_info["name"], fontsize=10.5, fontweight="bold", labelpad=8)
            cbar.ax.tick_params(labelsize=9)

    # Save figures
    mode_tag = f"{mode}_{target_point}"
    out_pdf = output_dir / f"{config_name}_heatmaps_{mode_tag}.pdf"
    plt.savefig(out_pdf, bbox_inches="tight")
    if pdf_writer is not None:
        pdf_writer.savefig(fig, bbox_inches="tight")

    if save_png:
        out_png = output_dir / f"{config_name}_heatmaps_{mode_tag}.png"
        plt.savefig(out_png, dpi=300, bbox_inches="tight")

    plt.close(fig)
    print(f"Saved: {out_pdf}")
    return str(out_pdf)


def main():
    parser = argparse.ArgumentParser(
        description="Plot 2D hyperparameter heatmaps (3 parameter combinations x 5 metrics) at target training epochs."
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
        default="plots/hyper_sweep_heatmaps",
        help="Directory to save the heatmap plots (default: plots/hyper_sweep_heatmaps)",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["all"],
        help="Specific configs to plot (default: 'all', or list e.g. GROK_MNIST_CE_MLP)",
    )
    parser.add_argument(
        "--epochs",
        nargs="+",
        default=None,
        help="Custom target points (in epochs or steps depending on model). If not specified, uses STATIC_EVAL_POINTS.",
    )
    parser.add_argument(
        "--no-nc",
        action="store_true",
        help="Do not include the Neural Collapse (NC1) row (default: included where available)",
    )
    parser.add_argument(
        "--no-annot",
        action="store_true",
        help="Do not display numeric text values inside heatmap cells",
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
        "epoch",
        "eval/test/accuracy",
        "eval/test/loss/cross_entropy/mean",
        "eval/test/loss/mse/mean",
        "eval/attack/ce_loss/auc",
        "eval/attack/mse_loss/auc",
        "eval/attack/ce_loss/tpr-at-fpr/1",
        "eval/attack/mse_loss/tpr-at-fpr/1",
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

    if args.configs and "all" not in args.configs:
        all_configs = [c for c in all_configs if c in args.configs]

    if not all_configs:
        print("No matching configurations found to plot.")
        return

    combined_pdf_path = out_dir / "all_models_heatmaps.pdf"
    print(f"\nCompiling all heatmap panels into combined PDF '{combined_pdf_path}'...")
    generated_count = 0

    with PdfPages(combined_pdf_path) as combined_pdf:
        for cfg_name in all_configs:
            cfg_df = df[df["base_config"] == cfg_name]
            eval_spec = get_eval_spec_for_config(cfg_name)

            if args.epochs:
                target_pts = [int(p) if p.isdigit() else float(p) for p in args.epochs]
                eval_mode = eval_spec.get("mode", "epoch")
            else:
                target_pts = eval_spec.get("points", [80, 400])
                eval_mode = eval_spec.get("mode", "epoch")

            print(f"\n[Plotting Heatmaps] {cfg_name} for {eval_mode}s: {target_pts}...")
            for pt in target_pts:
                pdf_path = plot_config_heatmaps_at_checkpoint(
                    config_name=cfg_name,
                    df_config=cfg_df,
                    output_dir=out_dir,
                    target_point=pt,
                    mode=eval_mode,
                    include_nc=not args.no_nc,
                    annot=not args.no_annot,
                    save_png=args.save_png,
                    pdf_writer=combined_pdf,
                )
                if pdf_path:
                    generated_count += 1

    print(f"\n[Done] Successfully generated {generated_count} heatmap plots and combined PDF in '{out_dir}'.")


if __name__ == "__main__":
    main()
