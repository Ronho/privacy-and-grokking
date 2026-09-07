#!/usr/bin/env python3
"""plot_comparison_grid.py.

Generates a publication-quality 3x2 multi-panel grid comparing 3 grokking models:
  Row 1: (I) MLP (MSE)   [Run: 39ca6f59220d438d9d14de7fdb0be8c9]
  Row 2: (II) MLP (CE)    [Run: 82c4f9ab35bb4b7180fdacc194ce7e07]
  Row 3: (III) ViT (CE)   [Run: b75317f9447346249f0b811acec427ce]

Columns:
  Column 1: Principal Component Trajectory (PC1 vs PC2, all steps, 50k tags, step colorbar)
  Column 2: Loss (left y-axis, log scale) and Weight Distance (right y-axis, linear scale)
            over Step (log scale with 10^1..10^5 ticks), matching reproduction plot aesthetics.

Strictly exports a single PDF without super-headers or colored/bold axis labels.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, LogFormatterMathtext, LogLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_CACHE_DIR = PROJECT_DIR / "cache"
DEFAULT_PLOTS_DIR = PROJECT_DIR / "plots"
DEFAULT_TRACKING_URI = "http://localhost:5051"

# Ordered list of models exactly matching the reference specification
MODELS = [
    {
        "run_id": "39ca6f59220d438d9d14de7fdb0be8c9",
        "label": "(I) MLP (MSE)",
        "loss_metric": "eval/train/loss/mse/mean",
        "test_loss_metric": "eval/test/loss/mse/mean",
    },
    {
        "run_id": "82c4f9ab35bb4b7180fdacc194ce7e07",
        "label": "(II) MLP (CE)",
        "loss_metric": "eval/train/loss/cross_entropy/mean",
        "test_loss_metric": "eval/test/loss/cross_entropy/mean",
    },
    {
        "run_id": "b75317f9447346249f0b811acec427ce",
        "label": "(III) ViT (CE)",
        "loss_metric": "eval/train/loss/cross_entropy/mean",
        "test_loss_metric": "eval/test/loss/cross_entropy/mean",
    },
]

# Consistent styling tokens across all rows matching plot_reproduction_nc_grokking.py
COLOR_TRAIN_LOSS = "#ff7f0e"       # tab:orange (Train in reference image)
COLOR_TEST_LOSS = "#1f77b4"        # tab:blue (Test in reference image)
COLOR_DIST_START = "#7c3aed"       # Purple/Indigo (distinct from Test Blue)
COLOR_STEP_DIST = "#059669"        # Emerald green
COLOR_PCA_LINE = "#cbd5e1"         # Slate connecting line
COLOR_TEXT = "#000000"


def format_step_cbar(x, pos):
    """Clean tick formatting for colorbar steps (0, 50k, 100k, 150k)."""
    if x <= 0:
        return "0"
    if x >= 1_000_000:
        return f"{x / 1_000_000:.1f}M"
    if x >= 1_000:
        return f"{int(x / 1_000)}k" if x % 1_000 == 0 else f"{x / 1_000:.1f}k"
    return f"{int(x)}"


def step_tag_label(step: int) -> str:
    """Formats 50k step tags (0, 50k, 100k, 150k)."""
    if step == 0:
        return "0"
    if step >= 1_000_000:
        return f"{int(step / 1_000_000)}M" if step % 1_000_000 == 0 else f"{step / 1_000_000:.1f}M"
    if step >= 1_000:
        return f"{int(step / 1_000)}k" if step % 1_000 == 0 else f"{step / 1_000:.1f}k"
    return str(step)


def get_loss_dataframe(
    run_id: str,
    loss_metric: str,
    test_loss_metric: str,
    cache_dir: Path,
    df_pca: pd.DataFrame,
    tracking_uri: str = DEFAULT_TRACKING_URI,
) -> pd.DataFrame:
    """Loads full loss history from cache or queries MLflow tracking server."""
    cached_path = cache_dir / f"{run_id}_loss_full.parquet"
    if cached_path.is_file():
        try:
            return pd.read_parquet(cached_path)
        except Exception:
            pass

    # Try fetching from MLflow
    try:
        import mlflow

        client = mlflow.tracking.MlflowClient(tracking_uri)
        hist_train = client.get_metric_history(run_id, loss_metric)
        df_train = (
            pd.DataFrame([{"step": m.step, "loss": m.value} for m in hist_train])
            .groupby("step", as_index=False)["loss"]
            .mean()
        )

        try:
            hist_test = client.get_metric_history(run_id, test_loss_metric)
            df_test = (
                pd.DataFrame([{"step": m.step, "test_loss": m.value} for m in hist_test])
                .groupby("step", as_index=False)["test_loss"]
                .mean()
            )
            df_train = pd.merge(df_train, df_test, on="step", how="left")
        except Exception:
            pass

        cache_dir.mkdir(parents=True, exist_ok=True)
        df_train.to_parquet(cached_path, index=False)
        return df_train
    except Exception:
        cols = ["step", "loss"]
        if "test_loss" in df_pca.columns:
            cols.append("test_loss")
        return df_pca[cols].copy()


def plot_comparison_grid(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    output_path: Path | None = None,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    include_test_loss: bool = True,
) -> Path:
    """Renders the publication-ready 3x2 grid figure and saves as a single PDF."""
    if output_path is None:
        DEFAULT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = DEFAULT_PLOTS_DIR / "model_comparison_grid.pdf"
    else:
        output_path = Path(output_path)
        if output_path.suffix.lower() != ".pdf":
            output_path = output_path.with_suffix(".pdf")
        output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(13.0, 10.5),
        dpi=300,
        gridspec_kw={"width_ratios": [1.0, 2.2], "wspace": 0.24, "hspace": 0.24},
    )

    for row_idx, model in enumerate(MODELS):
        run_id = model["run_id"]
        row_label = model["label"]
        pca_file = cache_dir / f"{run_id}_pca.parquet"

        if not pca_file.is_file():
            raise FileNotFoundError(
                f"PCA cache file not found for {row_label} ({run_id}): {pca_file}\n"
                "Please run scripts/compute_checkpoint_pca.py first."
            )

        df_pca = pd.read_parquet(pca_file).sort_values(by="step").reset_index(drop=True)
        df_loss = get_loss_dataframe(
            run_id=run_id,
            loss_metric=model["loss_metric"],
            test_loss_metric=model["test_loss_metric"],
            cache_dir=cache_dir,
            df_pca=df_pca,
            tracking_uri=tracking_uri,
        )

        # Load PCA metadata for explained variance
        meta_file = cache_dir / f"{run_id}_pca_meta.json"
        pc1_ratio = None
        pc2_ratio = None
        if meta_file.is_file():
            try:
                import json

                with open(meta_file, encoding="utf-8") as f:
                    meta_data = json.load(f)
                ev = meta_data.get("explained_variance_ratio", [])
                if len(ev) > 0:
                    pc1_ratio = ev[0] * 100
                if len(ev) > 1:
                    pc2_ratio = ev[1] * 100
            except Exception:
                pass

        if pc1_ratio is None or pc2_ratio is None:
            try:
                var1 = float(np.var(df_pca["pc1"].values))
                var2 = float(np.var(df_pca["pc2"].values))
                tot = var1 + var2
                if tot > 0:
                    pc1_ratio = (var1 / tot) * 100
                    pc2_ratio = (var2 / tot) * 100
            except Exception:
                pass

        pc1_label = f"PC 1 ({pc1_ratio:.1f}%)" if pc1_ratio is not None else "PC 1"
        pc2_label = f"PC 2 ({pc2_ratio:.1f}%)" if pc2_ratio is not None else "PC 2"

        ax_pca = axes[row_idx, 0]
        ax_loss = axes[row_idx, 1]

        # Make left PCA plot strictly square (1:1 display aspect ratio)
        ax_pca.set_box_aspect(1)

        # -------------------------------------------------------------
        # Left Margin: Model Row Label matching plot_reproduction_nc_grokking.py
        # -------------------------------------------------------------
        ax_pca.annotate(
            row_label,
            xy=(-0.40, 0.5),
            xycoords="axes fraction",
            fontsize=12.5,
            fontweight="bold",
            ha="center",
            va="center",
            rotation=90,
            color=COLOR_TEXT,
        )

        # -------------------------------------------------------------
        # Column 1: Principal Component Display with Step Colorbar
        # -------------------------------------------------------------
        pc1 = df_pca["pc1"].values
        pc2 = df_pca["pc2"].values
        steps_pca = df_pca["step"].values

        # Trajectory connecting line
        ax_pca.plot(
            pc1,
            pc2,
            color=COLOR_PCA_LINE,
            linestyle="-",
            linewidth=1.3,
            alpha=0.75,
            zorder=2,
        )

        # Circle markers colored by training step with colormap
        sc = ax_pca.scatter(
            pc1,
            pc2,
            c=steps_pca,
            cmap="viridis",
            s=26,
            edgecolors="#64748b",
            linewidths=0.5,
            zorder=3,
        )

        # Dedicated colorbar matching the exact height of the square plot
        cax = inset_axes(
            ax_pca,
            width="5%",
            height="100%",
            loc="lower left",
            bbox_to_anchor=(1.04, 0.0, 1.0, 1.0),
            bbox_transform=ax_pca.transAxes,
            borderpad=0,
        )
        cbar = fig.colorbar(sc, cax=cax)
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(format_step_cbar))
        cbar.ax.tick_params(labelsize=8, colors=COLOR_TEXT)

        # Add tags ONLY every 50k steps (start, 50k, 100k, end)
        # Keep them strictly inside the plot boundaries by directing inward
        x_min, x_max = np.min(pc1), np.max(pc1)
        y_min, y_max = np.min(pc2), np.max(pc2)
        x_range = max(x_max - x_min, 1e-3)
        y_range = max(y_max - y_min, 1e-3)

        # Set padded limits so points near borders have margin for tags
        left_pad = 0.22 if row_idx == 0 else 0.14
        right_pad = 0.18 if row_idx == 1 else 0.14
        ax_pca.set_xlim(x_min - left_pad * x_range, x_max + right_pad * x_range)
        ax_pca.set_ylim(y_min - 0.14 * y_range, y_max + 0.14 * y_range)

        tagged_indices = []
        for idx, step_val in enumerate(steps_pca):
            is_start = idx == 0
            is_end = idx == len(steps_pca) - 1
            is_50k = (step_val % 50_000 == 0) and (step_val > 0)
            if is_start or is_end or is_50k:
                tagged_indices.append(idx)

        # Visually highlight the tagged checkpoint circles with light grey highlight around the circle
        tagged_indices = np.array(tagged_indices)
        ax_pca.scatter(
            pc1[tagged_indices],
            pc2[tagged_indices],
            s=90,
            color="#e2e8f0",
            edgecolors="#cbd5e1",
            linewidths=1.2,
            zorder=3.5,
        )
        ax_pca.scatter(
            pc1[tagged_indices],
            pc2[tagged_indices],
            c=steps_pca[tagged_indices],
            cmap="viridis",
            s=40,
            edgecolors="#94a3b8",
            linewidths=1.0,
            zorder=4,
        )

        # Tag positioning with unambiguous arrow pointers
        for idx in tagged_indices:
            step_val = steps_pca[idx]
            tag = step_tag_label(step_val)
            px = pc1[idx]
            py = pc2[idx]

            # Custom offsets per row to ensure zero overlap and clear arrow pointers
            if row_idx == 0:
                if step_val == 0:
                    offset_x, offset_y = -18, 14
                    ha, va = "right", "bottom"
                elif step_val == 50_000:
                    offset_x, offset_y = 26, -18
                    ha, va = "left", "top"
                elif step_val == 100_000:
                    offset_x, offset_y = -28, 6
                    ha, va = "right", "bottom"
                elif step_val == 150_000:
                    offset_x, offset_y = 24, 16
                    ha, va = "left", "bottom"
                else:
                    offset_x, offset_y = 15, 15
                    ha, va = "left", "bottom"
            elif row_idx == 1:
                if step_val == 0:
                    offset_x, offset_y = -18, 14
                    ha, va = "right", "bottom"
                elif step_val == 50_000:
                    offset_x, offset_y = -20, -16
                    ha, va = "right", "top"
                elif step_val == 100_000:
                    offset_x, offset_y = -22, 14
                    ha, va = "right", "bottom"
                elif step_val == 150_000:
                    offset_x, offset_y = -12, 18
                    ha, va = "right", "bottom"
                else:
                    offset_x, offset_y = 15, 15
                    ha, va = "left", "bottom"
            elif row_idx == 2:
                if step_val == 0:
                    offset_x, offset_y = 18, -8
                    ha, va = "left", "top"
                elif step_val == 50_000:
                    offset_x, offset_y = -18, 14
                    ha, va = "right", "bottom"
                elif step_val == 100_000:
                    offset_x, offset_y = -6, -22
                    ha, va = "center", "top"
                elif step_val == 150_000:
                    offset_x, offset_y = -22, 16
                    ha, va = "right", "bottom"
                else:
                    offset_x, offset_y = 15, 15
                    ha, va = "left", "bottom"

            ax_pca.annotate(
                tag,
                xy=(px, py),
                xytext=(offset_x, offset_y),
                textcoords="offset points",
                fontsize=8.0,
                fontweight="bold",
                color="#0f172a",
                ha=ha,
                va=va,
                bbox=dict(
                    boxstyle="round,pad=0.22",
                    fc="white",
                    ec="#475569",
                    lw=0.7,
                    alpha=0.95,
                ),
                arrowprops=dict(
                    arrowstyle="->",
                    color="#1e293b",
                    lw=0.9,
                    shrinkA=2,
                    shrinkB=4,
                    mutation_scale=10,
                ),
                zorder=6,
            )

        # Axis labels with explained variance
        ax_pca.set_xlabel(pc1_label, fontsize=10, fontweight="normal", color=COLOR_TEXT)
        ax_pca.set_ylabel(pc2_label, fontsize=10, fontweight="normal", color=COLOR_TEXT)
        ax_pca.tick_params(axis="both", colors=COLOR_TEXT, labelsize=9)
        ax_pca.grid(True, linestyle="--", alpha=0.3)

        # -------------------------------------------------------------
        # Column 2: Loss (Left Y) & Weight Distance (Right Y) over Step
        # -------------------------------------------------------------
        if "distance_from_start" in df_pca.columns:
            d_start = df_pca["distance_from_start"].values
            d_start = d_start - d_start[0]
        else:
            coords = df_pca[["pc1", "pc2"]].values
            d_start = np.linalg.norm(coords - coords[0], axis=1)

        if "step_distance" in df_pca.columns:
            step_diffs = df_pca["step_distance"].values
        else:
            coords = df_pca[["pc1", "pc2"]].values
            diffs = np.linalg.norm(coords[1:] - coords[:-1], axis=1)
            step_diffs = np.concatenate([[0.0], diffs])

        # Step mapping for log scale (start at min non-zero or step 1)
        x_pca = np.where(steps_pca == 0, 1, steps_pca)
        x_loss = np.where(df_loss["step"] == 0, 1, df_loss["step"])
        y_train_loss = df_loss["loss"].values
        has_test = (
            include_test_loss
            and "test_loss" in df_loss.columns
            and df_loss["test_loss"].notna().any()
        )

        current_lines = []

        # Left Y-axis: Loss (Log Scale, normal font, clean lines)
        # Test Loss (Blue, plotted first to match legend order in reference)
        if has_test:
            y_test_loss = df_loss["test_loss"].values
            l_test = ax_loss.plot(
                x_loss,
                y_test_loss,
                color=COLOR_TEST_LOSS,
                linestyle="-",
                linewidth=1.6,
                alpha=0.90,
                label="Test",
                zorder=4,
            )
            current_lines.extend(l_test)

        # Train Loss (Orange)
        l_train = ax_loss.plot(
            x_loss,
            y_train_loss,
            color=COLOR_TRAIN_LOSS,
            linestyle="-",
            linewidth=1.6,
            alpha=0.95,
            label="Train",
            zorder=4,
        )
        current_lines.extend(l_train)

        ax_loss.set_yscale("log")
        ax_loss.set_ylabel("Loss", fontsize=10, fontweight="normal", color=COLOR_TEXT)
        ax_loss.tick_params(axis="both", colors=COLOR_TEXT, labelsize=9)

        # Right Y-axis: Weight Distance (Linear Scale, normal font, clean lines)
        ax_dist = ax_loss.twinx()
        l_dist = ax_dist.plot(
            x_pca,
            d_start,
            color=COLOR_DIST_START,
            linestyle="-",
            linewidth=1.8,
            label="Distance from Start",
            zorder=3,
        )
        l_step = ax_dist.plot(
            x_pca[1:],
            step_diffs[1:],
            color=COLOR_STEP_DIST,
            linestyle="--",
            linewidth=1.4,
            alpha=0.90,
            label="Step Displacement",
            zorder=2,
        )
        current_lines.extend(l_dist + l_step)

        ax_dist.set_ylabel(
            "Weight Distance", fontsize=10, fontweight="normal", color=COLOR_TEXT
        )
        ax_dist.tick_params(axis="y", colors=COLOR_TEXT, labelsize=9)
        max_dist = max(max(d_start), max(step_diffs[1:]) if len(step_diffs) > 1 else 1.0)
        ax_dist.set_ylim(bottom=0.0, top=max_dist * 1.15)

        # X-axis configuration matching user's image exactly (10^1 to 10^5 with minor ticks)
        ax_loss.set_xscale("log")
        ax_loss.set_xlim(left=7.0, right=1.6e5)
        ax_loss.xaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
        ax_loss.xaxis.set_major_formatter(LogFormatterMathtext())
        ax_loss.xaxis.set_minor_locator(
            LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100)
        )
        # Major grid lines only, exactly matching reference image
        ax_loss.grid(True, which="major", linestyle="-", alpha=0.3)

        # X-axis label ONLY on the last row
        if row_idx == 2:
            ax_loss.set_xlabel("Step", fontsize=10, fontweight="normal", color=COLOR_TEXT)
        else:
            ax_loss.set_xlabel("")

        # Display legend in all plots in column 2 (middle left placement)
        labels_all = [ln.get_label() for ln in current_lines]
        leg = ax_dist.legend(
            current_lines,
            labels_all,
            title="Metric",
            loc="center left",
            bbox_to_anchor=(0.03, 0.48),
            framealpha=1.0,
            facecolor="white",
            edgecolor="#cbd5e1",
            fontsize=7.5,
            title_fontsize=8.0,
        )
        leg.set_zorder(100)
        leg.get_frame().set_linewidth(0.8)

        # Top column headers for row 0
        if row_idx == 0:
            ax_pca.set_title(
                "(a) Principal Component Trajectory",
                fontsize=11,
                fontweight="normal",
                pad=10,
                color=COLOR_TEXT,
            )
            ax_loss.set_title(
                "(b) Train and Test Loss & Weight Distance",
                fontsize=11,
                fontweight="normal",
                pad=10,
                color=COLOR_TEXT,
            )

    fig.subplots_adjust(left=0.13, right=0.92, top=0.94, bottom=0.06)
    plt.savefig(output_path, format="pdf")
    plt.close()

    print(f"Successfully generated comparison grid PDF: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate publication 3x2 grid comparing PCA trajectory and distance/loss."
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Destination PDF file path (default: plots/model_comparison_grid.pdf).",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=str(DEFAULT_CACHE_DIR),
        help=f"Cache directory (default: {DEFAULT_CACHE_DIR}).",
    )
    parser.add_argument(
        "--uri",
        "-u",
        type=str,
        default=DEFAULT_TRACKING_URI,
        help=f"MLflow tracking URI (default: {DEFAULT_TRACKING_URI}).",
    )
    parser.add_argument(
        "--no-test-loss",
        action="store_true",
        help="Do not display test loss curve.",
    )

    args = parser.parse_args()

    plot_comparison_grid(
        cache_dir=Path(args.cache_dir),
        output_path=Path(args.output) if args.output else None,
        tracking_uri=args.uri,
        include_test_loss=not args.no_test_loss,
    )


if __name__ == "__main__":
    main()
