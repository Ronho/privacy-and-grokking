#!/usr/bin/env python3
"""plot_comparison_grid.py.

Generates an all-in-one publication-quality 3x3 multi-panel grid comparing 3 grokking models:
  Row 1: (I) MLP (MSE)   [Run: 39ca6f59220d438d9d14de7fdb0be8c9]
  Row 2: (II) MLP (CE)    [Run: 82c4f9ab35bb4b7180fdacc194ce7e07]
  Row 3: (III) ViT (CE)   [Run: b75317f9447346249f0b811acec427ce]

Columns:
  Column 1: (a) 2D Training Loss Landscape with overlaid weight trajectory,
            sampling grid (40x40), milestone tags, and dual colorbars (Loss + Step).
  Column 2: (b) 2D Weight Norm Landscape ||theta||_2 with overlaid weight trajectory,
            sampling grid (40x40), milestone tags, and dual colorbars (||theta||_2 + Step).
  Column 3: (c) Train & Test Loss (left y-axis, log scale) and Weight Distance
            (right y-axis, linear scale) over Step (x-axis, log scale 10^1..10^5).

Exports both PDF and PNG in a single unified execution.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import FuncFormatter, LogFormatterMathtext, LogLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
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

# Styling tokens matching publication standards
COLOR_TRAIN_LOSS = "#ff7f0e"       # tab:orange
COLOR_TEST_LOSS = "#1f77b4"        # tab:blue
COLOR_GRAD_NORM = "#dc2626"        # crimson red
COLOR_SHARPNESS = "#d97706"        # amber / gold
COLOR_WEIGHT_NORM = "#4338ca"      # deep indigo
COLOR_DIST_START = "#9333ea"       # purple
COLOR_STEP_DIST = "#059669"        # emerald green
COLOR_EOS_LINE = "#64748b"         # slate gray
COLOR_TOP5_SHARE = "#e11d48"       # vivid rose / crimson
COLOR_LOGIT_MAG = "#0891b2"        # deep cyan / teal
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
    """Formats 50k step milestone tags (0, 50k, 100k, 150k)."""
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
    """Loads full metrics history (train loss, test loss, grad norm, weight norm) from cache or queries MLflow tracking server."""
    cached_path = cache_dir / f"{run_id}_loss_full.parquet"
    if cached_path.is_file():
        try:
            df_cached = pd.read_parquet(cached_path)
            if "grad_norm" in df_cached.columns and "weight_norm" in df_cached.columns:
                return df_cached
        except Exception:
            pass

    # Fallback / refresh from MLflow
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
            df_train = pd.merge(df_train, df_test, on="step", how="outer")
        except Exception:
            pass

        try:
            hist_grad = client.get_metric_history(run_id, "eval/grad_norm/total")
            df_grad = (
                pd.DataFrame([{"step": m.step, "grad_norm": m.value} for m in hist_grad])
                .groupby("step", as_index=False)["grad_norm"]
                .mean()
            )
            df_train = pd.merge(df_train, df_grad, on="step", how="outer")
        except Exception:
            pass

        try:
            hist_wnorm = client.get_metric_history(run_id, "eval/weight_norm/total")
            df_wnorm = (
                pd.DataFrame([{"step": m.step, "weight_norm": m.value} for m in hist_wnorm])
                .groupby("step", as_index=False)["weight_norm"]
                .mean()
            )
            df_train = pd.merge(df_train, df_wnorm, on="step", how="outer")
        except Exception:
            pass

        df_train = df_train.sort_values("step").reset_index(drop=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        df_train.to_parquet(cached_path, index=False)
        return df_train
    except Exception:
        cols = ["step", "loss"]
        if "test_loss" in df_pca.columns:
            cols.append("test_loss")
        return df_pca[cols].copy()


def render_pca_landscape_panel(
    ax: plt.Axes,
    fig: plt.Figure,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    landscape_vals: np.ndarray,
    metric_type: str,
    df_pca: pd.DataFrame,
    pc1_label: str,
    pc2_label: str,
    row_idx: int,
    step_colormap: str = "plasma",
    tag_interval: int = 50_000,
    custom_title: str | None = None,
) -> None:
    """Renders a square PCA plot with background landscape, 40x40 sampling grid, and dual colorbars."""
    ax.set_box_aspect(1)

    pc1 = df_pca["pc1"].values
    pc2 = df_pca["pc2"].values
    steps_pca = df_pca["step"].values

    # 1. Background landscape contours
    valid = landscape_vals[np.isfinite(landscape_vals)]
    if len(valid) == 0:
        raise ValueError("No finite values in landscape grid!")

    v_min = float(np.min(valid))
    v_max = float(np.max(valid))

    if metric_type in ("loss", "sharpness"):
        pos = valid[valid > 0]
        p_min = float(np.min(pos)) if len(pos) > 0 else 1e-4
        norm = LogNorm(vmin=p_min, vmax=v_max)
        levels = np.logspace(np.log10(p_min), np.log10(v_max), 30)
        cmap = "Blues_r"
        cf = ax.contourf(
            grid_x, grid_y, landscape_vals,
            levels=levels, cmap=cmap, norm=norm,
            extend="both", zorder=1,
        )
    else:  # weight_norm
        norm = Normalize(vmin=v_min, vmax=v_max)
        levels = np.linspace(v_min, v_max, 30)
        cmap = "PuBu"
        cf = ax.contourf(
            grid_x, grid_y, landscape_vals,
            levels=levels, cmap=cmap, norm=norm,
            extend="neither", zorder=1,
        )

    # 2. Evaluated 40x40 sampling grid and vertex dot markings
    for i in range(grid_x.shape[0]):
        ax.plot(grid_x[i, :], grid_y[i, :], color="#64748b", alpha=0.15, linewidth=0.3, zorder=1.5)
    for j in range(grid_x.shape[1]):
        ax.plot(grid_x[:, j], grid_y[:, j], color="#64748b", alpha=0.15, linewidth=0.3, zorder=1.5)
    ax.scatter(grid_x.ravel(), grid_y.ravel(), s=2.0, color="#334155", alpha=0.3, linewidths=0, zorder=1.6)

    # 3. Trajectory connecting lines (high-contrast outline)
    ax.plot(pc1, pc2, color="#0f172a", linestyle="-", linewidth=2.0, alpha=0.7, zorder=2)
    ax.plot(pc1, pc2, color="#f8fafc", linestyle="-", linewidth=1.2, alpha=0.95, zorder=3)

    # 4. Checkpoint markers colored by training step
    sc = ax.scatter(
        pc1, pc2,
        c=steps_pca,
        cmap=step_colormap,
        s=26,
        edgecolors="#1e293b",
        linewidths=0.5,
        zorder=4,
    )

    # 5. Highlight circles for milestone checkpoints
    tagged_indices = []
    for idx, step_val in enumerate(steps_pca):
        is_start = idx == 0
        is_end = idx == len(steps_pca) - 1
        is_50k = (step_val % tag_interval == 0) and (step_val > 0)
        if is_start or is_end or is_50k:
            tagged_indices.append(idx)

    ax.scatter(
        pc1[tagged_indices],
        pc2[tagged_indices],
        s=80,
        color="#e2e8f0",
        edgecolors="#cbd5e1",
        linewidths=1.0,
        zorder=4.5,
    )
    ax.scatter(
        pc1[tagged_indices],
        pc2[tagged_indices],
        c=steps_pca[tagged_indices],
        cmap=step_colormap,
        s=28,
        edgecolors="#1e293b",
        linewidths=0.5,
        zorder=4.6,
    )

    # 6. Milestone step tags directed cleanly inward
    x_min, x_max = float(np.min(grid_x)), float(np.max(grid_x))
    y_min, y_max = float(np.min(grid_y)), float(np.max(grid_y))
    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)

    for idx in tagged_indices:
        p1 = pc1[idx]
        p2 = pc2[idx]
        step_val = steps_pca[idx]
        dx = -16 if p1 > x_mid else 14
        dy = -12 if p2 > y_mid else 12

        ax.annotate(
            step_tag_label(step_val),
            xy=(p1, p2),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7.5,
            fontweight="bold",
            color="#0f172a",
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor="#ffffff",
                alpha=0.90,
                edgecolor="#94a3b8",
                linewidth=0.6,
            ),
            zorder=6,
        )

    # 7. Start & End Markers
    ax.scatter([pc1[0]], [pc2[0]], color="#22c55e", edgecolors="#0f172a", s=85, marker="o", linewidths=0.9, zorder=5)
    ax.scatter([pc1[-1]], [pc2[-1]], color="#ef4444", edgecolors="#0f172a", s=95, marker="*", linewidths=0.9, zorder=5)

    # 8. Dual Colorbars to the right of the square plot
    # (a) Landscape Colorbar (Blue tone mapping)
    cax_land = inset_axes(
        ax,
        width="4.5%",
        height="100%",
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0, 1.0, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    cbar_land = fig.colorbar(
        cf,
        cax=cax_land,
        extend="both" if metric_type == "loss" else "neither",
    )
    if custom_title is not None:
        land_title = custom_title
    elif metric_type == "loss":
        land_title = "Loss"
    elif metric_type == "weight_norm":
        land_title = r"$\|\theta\|_2$"
    else:
        land_title = r"$\|H\|_2$"
    cbar_land.ax.set_title(land_title, fontsize=7.5, fontweight="bold", pad=5, color=COLOR_TEXT)
    cbar_land.ax.tick_params(labelsize=6.5, colors=COLOR_TEXT)

    if metric_type in ("loss", "sharpness") and v_max > 0:
        min_exp = np.floor(np.log10(p_min))
        max_exp = np.ceil(np.log10(v_max))
        if max_exp - min_exp <= 4:
            cand_ticks = 10.0 ** np.arange(min_exp, max_exp + 1)
        else:
            cand_ticks = 10.0 ** np.linspace(min_exp, max_exp, 4)
        lticks = [t for t in cand_ticks if p_min * 0.95 <= t <= v_max * 1.05]
        if len(lticks) < 3:
            lticks = np.logspace(np.log10(p_min), np.log10(v_max), 4)
        cbar_land.set_ticks(lticks)

        def fmt_log_val(x, _):
            if x >= 100:
                return f"{x:.0f}"
            elif x >= 1:
                return f"{x:.1f}".rstrip("0").rstrip(".")
            elif x >= 0.01:
                return f"{x:.2f}"
            else:
                return f"{x:.1e}"

        cbar_land.ax.yaxis.set_major_formatter(FuncFormatter(fmt_log_val))
    else:
        norm_ticks = np.linspace(v_min, v_max, 4)
        cbar_land.set_ticks(norm_ticks)
        cbar_land.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}"))

    # (b) Step Progression Colorbar
    cax_step = inset_axes(
        ax,
        width="4.5%",
        height="100%",
        loc="lower left",
        bbox_to_anchor=(1.17, 0.0, 1.0, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    cbar_step = fig.colorbar(sc, cax=cax_step)
    cbar_step.ax.set_title("Step", fontsize=7.5, fontweight="bold", pad=5, color=COLOR_TEXT)
    cbar_step.ax.yaxis.set_major_formatter(FuncFormatter(format_step_cbar))
    cbar_step.ax.tick_params(labelsize=6.5, colors=COLOR_TEXT)

    # 9. Axes limits and labels
    x_pad = (x_max - x_min) * 0.02
    y_pad = (y_max - y_min) * 0.02
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    ax.set_xlabel(pc1_label, fontsize=9, labelpad=4, color=COLOR_TEXT)
    ax.set_ylabel(pc2_label, fontsize=9, labelpad=4, color=COLOR_TEXT)
    ax.tick_params(labelsize=8, colors=COLOR_TEXT)


def plot_comparison_grid(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    output_path: Path | None = None,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    include_test_loss: bool = True,
) -> Path:
    """Renders the publication-ready 3x3 grid (Loss Landscape, Weight Norm Landscape, Metrics over time)."""
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
        ncols=7,
        figsize=(37.8, 10.8),
        dpi=300,
        gridspec_kw={
            "width_ratios": [1.0, 1.0, 1.0, 1.0, 1.65, 1.35, 1.25],
            "wspace": 0.38,
            "hspace": 0.28,
        },
    )

    for row_idx, model in enumerate(MODELS):
        run_id = model["run_id"]
        row_label = model["label"]
        
        traj_dir = cache_dir / "runs" / run_id / "trajectories"
        grid_file = traj_dir / "grid.json"
        proj_file = traj_dir / "projection.json"
        config_file = cache_dir / "runs" / run_id / "training_config.json"

        if not grid_file.is_file() or not proj_file.is_file():
            raise FileNotFoundError(
                f"Trajectory files not found for {row_label} ({run_id}) at {traj_dir}\n"
                "Please run visualizations/preprocessing/extract_trajectories.py first."
            )

        with open(grid_file, 'r', encoding="utf-8") as f:
            grid_data = json.load(f)
            
        with open(proj_file, 'r', encoding="utf-8") as f:
            proj_data = json.load(f)
            
        weight_decay = 0.0
        if config_file.is_file():
            try:
                with open(config_file, 'r', encoding="utf-8") as f:
                    cfg = json.load(f)
                    weight_decay = float(cfg.get("optimizer", {}).get("weight_decay", 0.0))
            except Exception:
                pass

        x_coords = np.array(grid_data["x_coords"], dtype=float)
        y_coords = np.array(grid_data["y_coords"], dtype=float)
        grid_x, grid_y = np.meshgrid(x_coords, y_coords)

        losses = grid_data.get("losses", {})
        if isinstance(losses, dict) and "train" in losses:
            raw_loss = np.array(losses["train"], dtype=float)
        elif "train_losses" in grid_data:
            raw_loss = np.array(grid_data["train_losses"], dtype=float)
        else:
            raw_loss = np.zeros(len(x_coords) * len(y_coords))
            
        train_loss_grid = raw_loss.reshape(len(y_coords), len(x_coords))
        weight_norm_grid = np.array(grid_data["weight_norms"], dtype=float).reshape(len(y_coords), len(x_coords))
        
        if "sharpness" in grid_data:
            sharpness_grid = np.array(grid_data["sharpness"], dtype=float).reshape(len(y_coords), len(x_coords))
        else:
            sharpness_grid = np.zeros_like(train_loss_grid)
            
        reg_loss_grid = train_loss_grid + weight_decay * weight_norm_grid

        steps = proj_data.get("trajectories", {}).get("steps", [])
        pca_coords = np.array(proj_data.get("trajectories", {}).get("pca_coords", []))
        
        if len(pca_coords) > 0:
            df_pca = pd.DataFrame({
                "step": steps,
                "pc1": pca_coords[:, 0],
                "pc2": pca_coords[:, 1]
            })
        else:
            df_pca = pd.DataFrame(columns=["step", "pc1", "pc2"])

        df_loss = get_loss_dataframe(
            run_id=run_id,
            loss_metric=model["loss_metric"],
            test_loss_metric=model["test_loss_metric"],
            cache_dir=cache_dir,
            df_pca=df_pca,
            tracking_uri=tracking_uri,
        )

        evr = proj_data.get("explained_variance_ratio", [])
        pc1_ratio = evr[0] * 100 if len(evr) > 0 else None
        pc2_ratio = evr[1] * 100 if len(evr) > 1 else None

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

        ax_pca_loss = axes[row_idx, 0]
        ax_pca_norm = axes[row_idx, 1]
        ax_pca_reg = axes[row_idx, 2]
        ax_pca_sharp = axes[row_idx, 3]
        ax_loss = axes[row_idx, 4]
        ax_acc = axes[row_idx, 5]
        ax_samples = axes[row_idx, 6]

        # -------------------------------------------------------------
        # Left Margin: Model Row Label
        # -------------------------------------------------------------
        ax_pca_loss.annotate(
            row_label,
            xy=(-0.36, 0.5),
            xycoords="axes fraction",
            fontsize=12.0,
            fontweight="bold",
            ha="center",
            va="center",
            rotation=90,
            color=COLOR_TEXT,
        )

        # -------------------------------------------------------------
        # Column 1: Loss Landscape
        # -------------------------------------------------------------
        render_pca_landscape_panel(
            ax=ax_pca_loss,
            fig=fig,
            grid_x=grid_x,
            grid_y=grid_y,
            landscape_vals=train_loss_grid,
            metric_type="loss",
            df_pca=df_pca,
            pc1_label=pc1_label,
            pc2_label=pc2_label,
            row_idx=row_idx,
            step_colormap="plasma",
        )

        # -------------------------------------------------------------
        # Column 2: Weight Norm Landscape
        # -------------------------------------------------------------
        render_pca_landscape_panel(
            ax=ax_pca_norm,
            fig=fig,
            grid_x=grid_x,
            grid_y=grid_y,
            landscape_vals=weight_norm_grid,
            metric_type="weight_norm",
            df_pca=df_pca,
            pc1_label=pc1_label,
            pc2_label=pc2_label,
            row_idx=row_idx,
            step_colormap="plasma",
        )

        # -------------------------------------------------------------
        # Column 3: Regularized Loss Landscape
        # -------------------------------------------------------------
        render_pca_landscape_panel(
            ax=ax_pca_reg,
            fig=fig,
            grid_x=grid_x,
            grid_y=grid_y,
            landscape_vals=reg_loss_grid,
            metric_type="loss",
            df_pca=df_pca,
            pc1_label=pc1_label,
            pc2_label=pc2_label,
            row_idx=row_idx,
            step_colormap="plasma",
            custom_title=f"Loss + {weight_decay:.3g}" + r"$\|\theta\|_2$" if weight_decay > 0 else "Loss",
        )

        # -------------------------------------------------------------
        # Column 4: Sharpness Landscape
        # -------------------------------------------------------------
        render_pca_landscape_panel(
            ax=ax_pca_sharp,
            fig=fig,
            grid_x=grid_x,
            grid_y=grid_y,
            landscape_vals=sharpness_grid,
            metric_type="sharpness",
            df_pca=df_pca,
            pc1_label=pc1_label,
            pc2_label=pc2_label,
            row_idx=row_idx,
            step_colormap="plasma",
        )

        # -------------------------------------------------------------
        # Column 4: Loss, Sharpness & Weight Dynamics Over Time
        # -------------------------------------------------------------
        x_loss = df_loss["step"].values
        y_train_loss = df_loss["loss"].values

        # Weight distances computed from projected PCA weights
        pca_coords = df_pca[["pc1", "pc2"]].values
        w0 = pca_coords[0]
        d_start = np.linalg.norm(pca_coords - w0, axis=1)
        step_diffs = np.zeros(len(pca_coords))
        if len(pca_coords) > 1:
            step_diffs[1:] = np.linalg.norm(np.diff(pca_coords, axis=0), axis=1)

        has_test = (
            include_test_loss
            and "test_loss" in df_loss.columns
            and df_loss["test_loss"].notna().any()
        )

        current_lines = []

        # Left Y-axis: Loss, Gradient Norm, and Sharpness (Log Scale)
        if has_test:
            y_test_loss = df_loss["test_loss"].values
            valid_te = (y_test_loss > 0) & np.isfinite(y_test_loss)
            l_test = ax_loss.plot(
                x_loss[valid_te],
                y_test_loss[valid_te],
                color=COLOR_TEST_LOSS,
                linestyle="-",
                linewidth=1.6,
                alpha=0.90,
                label="Test Loss",
                zorder=4,
            )
            current_lines.extend(l_test)

        valid_tr = (y_train_loss > 0) & np.isfinite(y_train_loss)
        l_train = ax_loss.plot(
            x_loss[valid_tr],
            y_train_loss[valid_tr],
            color=COLOR_TRAIN_LOSS,
            linestyle="-",
            linewidth=1.6,
            alpha=0.95,
            label="Train Loss",
            zorder=4,
        )
        current_lines.extend(l_train)

        if "grad_norm" in df_loss.columns:
            y_grad_norm = df_loss["grad_norm"].values
            valid_gr = (y_grad_norm > 0) & np.isfinite(y_grad_norm)
            if valid_gr.any():
                l_grad = ax_loss.plot(
                    x_loss[valid_gr],
                    y_grad_norm[valid_gr],
                    color=COLOR_GRAD_NORM,
                    linestyle="-.",
                    linewidth=1.4,
                    alpha=0.90,
                    label=r"Grad Norm $\|\nabla \mathcal{L}\|_2$",
                    zorder=4,
                )
                current_lines.extend(l_grad)

        # Plot Sharpness curve
        sharpness_vals = None
        if "sharpness" in df_loss.columns and df_loss["sharpness"].notna().any():
            sharpness_vals = df_loss["sharpness"].values
            x_sharp = x_loss
        elif "sharpness" in df_pca.columns and df_pca["sharpness"].notna().any():
            sharpness_vals = df_pca["sharpness"].values
            x_sharp = df_pca["step"].values

        if sharpness_vals is not None:
            valid_sh = (sharpness_vals > 0) & np.isfinite(sharpness_vals)
            if valid_sh.any():
                l_sharp = ax_loss.plot(
                    x_sharp[valid_sh],
                    sharpness_vals[valid_sh],
                    color=COLOR_SHARPNESS,
                    linestyle="-",
                    linewidth=1.7,
                    alpha=0.92,
                    label=r"Sharpness $\lambda_{\max}(H)$",
                    zorder=4,
                )
                current_lines.extend(l_sharp)

        # Load hard samples and logit magnitude data if available
        hs_file = cache_dir / f"{run_id}_hard_samples.npz"
        has_hs = hs_file.is_file()
        hs_data = np.load(hs_file) if has_hs else None

        # Plot Mean Logit Magnitude curve
        if has_hs and "mean_logit_magnitude" in hs_data:
            logit_mag_vals = hs_data["mean_logit_magnitude"]
            hs_steps = hs_data["steps"]
            valid_lm = (logit_mag_vals > 0) & np.isfinite(logit_mag_vals)
            if valid_lm.any():
                l_logit = ax_loss.plot(
                    hs_steps[valid_lm],
                    logit_mag_vals[valid_lm],
                    color=COLOR_LOGIT_MAG,
                    linestyle="--",
                    linewidth=1.7,
                    alpha=0.92,
                    label=r"Mean Logit $|z|$",
                    zorder=4,
                )
                current_lines.extend(l_logit)

        # Edge of Stability 2/eta reference line (eta = 0.001 -> 2/eta = 2000)
        l_eos = ax_loss.axhline(
            y=2000.0,
            color=COLOR_EOS_LINE,
            linestyle=":",
            linewidth=1.4,
            alpha=0.85,
            label=r"$2/\eta$ (EoS Threshold)",
            zorder=2,
        )
        current_lines.append(l_eos)

        ax_loss.set_yscale("log")
        ax_loss.set_ylabel(
            r"Loss / $\|\nabla \mathcal{L}\|_2$ / $\lambda_{\max}(H)$ / Mean $|z|$",
            fontsize=9.0,
            fontweight="normal",
            color=COLOR_TEXT,
        )
        ax_loss.tick_params(axis="both", colors=COLOR_TEXT, labelsize=9)

        # Right Y-axis: Weight Norm and Distances (Linear Scale)
        ax_dist = ax_loss.twinx()
        if "weight_norm" in df_loss.columns:
            y_wnorm = df_loss["weight_norm"].values
            valid_wn = (y_wnorm > 0) & np.isfinite(y_wnorm)
            if valid_wn.any():
                l_wnorm = ax_dist.plot(
                    x_loss[valid_wn],
                    y_wnorm[valid_wn],
                    color=COLOR_WEIGHT_NORM,
                    linestyle="-",
                    linewidth=1.7,
                    alpha=0.90,
                    label=r"Weight Norm $\|\theta\|_2$",
                    zorder=3,
                )
                current_lines.extend(l_wnorm)

        l_dist = ax_dist.plot(
            df_pca["step"].values,
            d_start,
            color=COLOR_DIST_START,
            linestyle="--",
            linewidth=1.5,
            alpha=0.85,
            label="Distance from Start",
            zorder=3,
        )
        current_lines.extend(l_dist)

        l_step = ax_dist.plot(
            df_pca["step"].values,
            step_diffs,
            color=COLOR_STEP_DIST,
            linestyle=":",
            linewidth=1.4,
            alpha=0.85,
            label="Step Displacement",
            zorder=3,
        )
        current_lines.extend(l_step)

        # Plot Top-5 Loss Share curve
        if has_hs and "top5_share" in hs_data:
            hs_top5_share = hs_data["top5_share"]
            l_top5 = ax_dist.plot(
                hs_steps,
                hs_top5_share,
                color=COLOR_TOP5_SHARE,
                linestyle="-",
                linewidth=1.8,
                alpha=0.92,
                label="Top-5 Loss Share (%)",
                zorder=5,
            )
            current_lines.extend(l_top5)

        ax_dist.set_ylabel(
            r"Weight Norm & Top-5 Share (%)", fontsize=9.5, fontweight="normal", color=COLOR_TEXT
        )
        ax_dist.tick_params(axis="y", colors=COLOR_TEXT, labelsize=9)
        top_w = float(df_loss["weight_norm"].max()) if "weight_norm" in df_loss.columns and df_loss["weight_norm"].notna().any() else 0.0
        max_dist = max(top_w, max(d_start), max(step_diffs[1:]) if len(step_diffs) > 1 else 1.0)
        ax_dist.set_ylim(bottom=0.0, top=max(105.0, max_dist * 1.15))

        # X-axis configuration (log scale 10^1 to 10^5)
        ax_loss.set_xscale("log")
        ax_loss.set_xlim(left=7.0, right=1.6e5)
        ax_loss.xaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
        ax_loss.xaxis.set_major_formatter(LogFormatterMathtext())
        ax_loss.xaxis.set_minor_locator(
            LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100)
        )
        ax_loss.grid(True, which="major", linestyle="-", alpha=0.3)

        if row_idx == 2:
            ax_loss.set_xlabel("Step", fontsize=10, fontweight="normal", color=COLOR_TEXT)
        else:
            ax_loss.set_xlabel("")

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
            fontsize=6.8,
            title_fontsize=7.5,
        )
        leg.set_zorder(100)
        leg.get_frame().set_linewidth(0.8)

        # -------------------------------------------------------------
        # Column 5: (e) Train & Test Accuracy over Time
        # -------------------------------------------------------------
        if has_hs and "train_acc" in hs_data and "test_acc" in hs_data:
            acc_steps = hs_data["steps"]
            acc_train = hs_data["train_acc"] * 100.0
            acc_test = hs_data["test_acc"] * 100.0

            l_tr_acc = ax_acc.plot(
                acc_steps,
                acc_train,
                color=COLOR_TRAIN_LOSS,
                linestyle="-",
                linewidth=2.0,
                alpha=0.95,
                label="Train Acc",
                zorder=4,
            )
            l_te_acc = ax_acc.plot(
                acc_steps,
                acc_test,
                color=COLOR_TEST_LOSS,
                linestyle="-",
                linewidth=2.0,
                alpha=0.95,
                label="Test Acc",
                zorder=4,
            )
            # Reference 100% line
            ax_acc.axhline(
                100.0,
                color="#94a3b8",
                linestyle=":",
                linewidth=1.0,
                alpha=0.7,
                zorder=1,
            )

            ax_acc.set_ylabel("Accuracy (%)", fontsize=10, color=COLOR_TEXT)
            ax_acc.set_ylim(bottom=0.0, top=105.0)
            ax_acc.set_yticks([0, 20, 40, 60, 80, 100])
            ax_acc.tick_params(axis="y", colors=COLOR_TEXT, labelsize=9)

            ax_acc.set_xscale("log")
            ax_acc.set_xlim(left=7.0, right=1.6e5)
            ax_acc.xaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
            ax_acc.xaxis.set_major_formatter(LogFormatterMathtext())
            ax_acc.xaxis.set_minor_locator(
                LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100)
            )
            ax_acc.grid(True, which="major", linestyle="-", alpha=0.3)

            if row_idx == 2:
                ax_acc.set_xlabel("Step", fontsize=10, fontweight="normal", color=COLOR_TEXT)
            else:
                ax_acc.set_xlabel("")

            # Legend
            leg_acc = ax_acc.legend(
                loc="lower right",
                framealpha=0.95,
                facecolor="white",
                edgecolor="#cbd5e1",
                fontsize=7.5,
            )
            leg_acc.set_zorder(10)
            leg_acc.get_frame().set_linewidth(0.8)

        # Top column headers for row 0
        if row_idx == 0:
            ax_pca_loss.set_title(
                "(a) Loss Landscape",
                fontsize=11.5,
                fontweight="normal",
                pad=10,
                color=COLOR_TEXT,
            )
            ax_pca_norm.set_title(
                "(b) Weight Norm Landscape",
                fontsize=11.5,
                fontweight="normal",
                pad=10,
                color=COLOR_TEXT,
            )
            ax_pca_reg.set_title(
                "(c) Regularized Loss Landscape",
                fontsize=11.5,
                fontweight="normal",
                pad=10,
                color=COLOR_TEXT,
            )
            ax_pca_sharp.set_title(
                r"(d) Spectral Norm Landscape $\|H\|_2$",
                fontsize=11.5,
                fontweight="normal",
                pad=10,
                color=COLOR_TEXT,
            )
            ax_loss.set_title(
                r"(e) Loss, Curvature, $|z|$ & Top-5 Share",
                fontsize=11.5,
                fontweight="normal",
                pad=10,
                color=COLOR_TEXT,
            )
            ax_acc.set_title(
                "(f) Train & Test Accuracy",
                fontsize=11.5,
                fontweight="normal",
                pad=10,
                color=COLOR_TEXT,
            )
            ax_samples.set_title(
                "(g) Persistent Hard Samples (Step > 10k)",
                fontsize=11.5,
                fontweight="normal",
                pad=10,
                color=COLOR_TEXT,
            )

        # -------------------------------------------------------------
        # Column 7: Persistent Hard Samples (Top-5 Loss Contributors)
        # -------------------------------------------------------------
        ax_samples.axis("off")
        if has_hs:
            hs_imgs = hs_data["hard_sample_images"]
            hs_lbls = hs_data["hard_sample_labels"]
            hs_idxs = hs_data["hard_sample_indices"]
            hs_freqs = hs_data["hard_sample_frequencies"]

            # Subtitle explaining what the percentage represents
            ax_samples.text(
                0.5,
                0.96,
                "% = Anteil der 14 Ckpts (Step > 10k) in Top-5",
                transform=ax_samples.transAxes,
                ha="center",
                va="top",
                fontsize=7.8,
                color="#475569",
                fontstyle="italic",
            )

            n_display = min(5, len(hs_imgs))
            w_box = 0.17
            gap = (1.0 - n_display * w_box) / (n_display + 1)
            for k in range(n_display):
                x0 = gap + k * (w_box + gap)
                y0 = 0.14
                h_box = 0.62
                sub_ax = ax_samples.inset_axes([x0, y0, w_box, h_box])
                sub_ax.imshow(hs_imgs[k], cmap="gray_r", vmin=0, vmax=1)
                sub_ax.set_xticks([])
                sub_ax.set_yticks([])
                for spine in sub_ax.spines.values():
                    spine.set_edgecolor("#94a3b8")
                    spine.set_linewidth(0.9)
                sub_ax.set_title(
                    f"y = {hs_lbls[k]}",
                    fontsize=8.5,
                    fontweight="bold",
                    pad=3,
                    color=COLOR_TEXT,
                )
                sub_ax.set_xlabel(
                    f"#{hs_idxs[k]}\n({hs_freqs[k]:.0f}% Ckpts)",
                    fontsize=7.2,
                    labelpad=2,
                    color="#334155",
                )

    fig.subplots_adjust(left=0.038, right=0.965, top=0.93, bottom=0.07)
    plt.savefig(output_path, format="pdf")
    png_path = Path(output_path).with_suffix(".png")
    plt.savefig(png_path, dpi=300)
    plt.close()

    print(f"Successfully generated 3x6 comparison grid PDF and PNG:\n  - {output_path}\n  - {png_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate unified 3x3 publication grid comparing Loss Landscape, Weight Norm Landscape, and Metrics."
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
