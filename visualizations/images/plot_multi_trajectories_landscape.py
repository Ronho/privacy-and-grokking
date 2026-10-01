#!/usr/bin/env python3
"""plot_multi_trajectories_landscape.py.

Renders a publication-grade 2D visualization of multiple optimization trajectories
overlaid on a single, shared joint PCA loss landscape.

Features:
  - 2D logarithmic loss contours (LogNorm 10^1 down to 10^-8) revealing flat grokking valleys.
  - Multi-trajectory overlays color-coded by initialization scale and styled by model index.
  - Start (circle) and end/minimizer (star) markers on each trajectory path.
  - Dual-panel layout option comparing the joint Loss Landscape and Weight Norm (||theta||_2) Surface.
  - Fully automated execution: if joint preprocessing artifacts are missing, they are automatically
    computed and cached on demand.

Required Inputs:
  - cache/joint_trajectories/{group_id}/projection.json: Joint PCA coordinates and metadata.
  - cache/joint_trajectories/{group_id}/grid.json: Shared 2D loss and weight norm grid.

Generated Outputs:
  - plots/joint_trajectories/joint_loss_landscape_{group_id}.pdf (vector)
  - plots/joint_trajectories/joint_loss_landscape_{group_id}.png (300 DPI raster preview)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import typer
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterMathtext, LogLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from privacy_and_grokking.artifacts.cache import DEFAULT_TRACKING_URI
from privacy_and_grokking.artifacts.local import (
    joint_trajectory_grid_path,
    joint_trajectory_projection_path,
    plots_joint_trajectories_path,
)
from visualizations.preprocessing.extract_multi_trajectories import (
    generate_group_id,
    main as extract_main,
)

app = typer.Typer(pretty_exceptions_enable=False)


def format_step_tag(step: int) -> str:
    """Format step integer into concise tag label (e.g. 50k, 100k, 0)."""
    if step == 0:
        return "0"
    if step >= 1_000_000:
        val = step / 1_000_000
        return f"{val:.1f}M" if val % 1 != 0 else f"{int(val)}M"
    if step >= 1_000:
        val = step / 1_000
        return f"{val:.1f}k" if val % 1 != 0 else f"{int(val)}k"
    return str(step)


def render_joint_landscape_panel(
    ax: plt.Axes,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    landscape_vals: np.ndarray,
    metric_type: str,
    runs_data: dict[str, dict],
    pc1_label: str,
    pc2_label: str,
    scale_colors: dict[float, tuple | str],
    min_loss: float = 1e-8,
    max_loss: float = 10.0,
    landscape_cmap: str = "Blues_r",
    xscale: str = "symlog",
    yscale: str = "symlog",
    linthresh: float = 0.05,
    linscale: float = 1.0,
) -> None:
    """Renders a single square landscape panel with contours and multi-trajectory overlay."""
    ax.set_box_aspect(1)

    # 1. Background landscape contours
    valid = landscape_vals[np.isfinite(landscape_vals)]
    if len(valid) == 0:
        raise ValueError("No finite values in landscape grid!")

    if metric_type == "loss":
        actual_pos = valid[valid > 0]
        eff_min = min(min_loss, float(np.min(actual_pos))) if len(actual_pos) > 0 else min_loss
        eff_max = max(max_loss, float(np.max(valid)))

        clamped_vals = np.clip(landscape_vals, eff_min, eff_max)
        norm = LogNorm(vmin=eff_min, vmax=eff_max)
        min_exp = int(np.floor(np.log10(eff_min)))
        max_exp = int(np.ceil(np.log10(eff_max)))
        levels = 10.0 ** np.arange(min_exp, max_exp + 1)
        cmap = landscape_cmap

        cf = ax.contourf(
            grid_x,
            grid_y,
            clamped_vals,
            levels=levels,
            norm=norm,
            cmap=cmap,
            extend="both",
            alpha=0.92,
        )
        cbar_label = r"Training Loss (MSE, $\log_{10}$ scale)"
    else:
        norm = Normalize(vmin=float(np.min(valid)), vmax=float(np.max(valid)))
        cf = ax.contourf(
            grid_x,
            grid_y,
            landscape_vals,
            levels=40,
            norm=norm,
            cmap="viridis",
            extend="both",
            alpha=0.92,
        )
        cbar_label = r"Weight Norm $\|\theta\|_2$"

    # 2. Colorbar on the right of the panel
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    if metric_type == "loss":
        cbar = plt.colorbar(
            cf,
            cax=cax,
            format=LogFormatterMathtext(base=10.0),
            ticks=LogLocator(base=10.0),
        )
    else:
        cbar = plt.colorbar(cf, cax=cax)

    cbar.set_label(cbar_label, fontsize=10, fontweight="medium", labelpad=8)
    cbar.ax.tick_params(labelsize=8)

    # 3. Trajectory Overlay for each run
    sorted_run_ids = sorted(
        runs_data.keys(),
        key=lambda r: (
            runs_data[r].get("model_index", 0),
            runs_data[r].get("init_scale", 1.0),
            r,
        ),
    )

    for r_id in sorted_run_ids:
        r_info = runs_data[r_id]
        coords = np.array(r_info["pca_coords"])
        scale = float(r_info.get("init_scale", 1.0))
        m_idx = int(r_info.get("model_index", 0))

        color = scale_colors.get(scale, "tab:blue")
        linestyle = "-" if m_idx == 0 else "--"

        # Trajectory path line
        ax.plot(
            coords[:, 0],
            coords[:, 1],
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
            alpha=0.88,
            zorder=4,
        )

        # Start point (Step 0) - circle marker
        ax.scatter(
            [coords[0, 0]],
            [coords[0, 1]],
            s=45,
            color=color,
            edgecolors="white",
            linewidth=1.0,
            marker="o",
            zorder=5,
        )

        # End point (Final minimizer) - star marker
        ax.scatter(
            [coords[-1, 0]],
            [coords[-1, 1]],
            s=80,
            color=color,
            edgecolors="black",
            linewidth=1.0,
            marker="*",
            zorder=6,
        )

    # Scale options
    if xscale == "symlog":
        ax.set_xscale("symlog", linthresh=linthresh, linscale=linscale)
    elif xscale == "log":
        ax.set_xscale("log")

    if yscale == "symlog":
        ax.set_yscale("symlog", linthresh=linthresh, linscale=linscale)
    elif yscale == "log":
        ax.set_yscale("log")

    ax.set_xlabel(pc1_label, fontsize=10, fontweight="medium", labelpad=6)
    ax.set_ylabel(pc2_label, fontsize=10, fontweight="medium", labelpad=6)
    ax.tick_params(labelsize=8.5)
    ax.grid(True, which="both", linestyle=":", alpha=0.35, color="gray")


def plot_joint_trajectories_landscape(
    group_id: str,
    output_path: Path,
    panels: str = "both",
    min_loss: float = 1e-8,
    max_loss: float = 10.0,
    landscape_cmap: str = "Blues_r",
    xscale: str = "symlog",
    yscale: str = "symlog",
    linthresh: float = 0.05,
    linscale: float = 1.0,
    dpi: int = 300,
) -> None:
    """Render publication-quality 2D figure with multiple trajectories over joint landscape."""
    proj_path = joint_trajectory_projection_path(group_id)
    grid_path = joint_trajectory_grid_path(group_id)

    if not proj_path.exists() or not grid_path.exists():
        raise FileNotFoundError(
            f"Required joint artifacts missing for group '{group_id}'. "
            f"Run extract_multi_trajectories first."
        )

    proj_data = json.loads(proj_path.read_bytes())
    grid_data = json.loads(grid_path.read_bytes())

    evr = proj_data["explained_variance_ratio"]
    pc1_evr = evr[0] * 100 if len(evr) > 0 else 0.0
    pc2_evr = evr[1] * 100 if len(evr) > 1 else 0.0
    total_evr = (evr[0] + evr[1]) * 100 if len(evr) > 1 else pc1_evr

    scale_tag_x = f" ({xscale})" if xscale != "linear" else ""
    scale_tag_y = f" ({yscale})" if yscale != "linear" else ""
    pc1_label = f"Joint PC1 ({pc1_evr:.1f}% Variance){scale_tag_x}"
    pc2_label = f"Joint PC2 ({pc2_evr:.1f}% Variance){scale_tag_y}"

    runs_data = proj_data["runs"]
    unique_scales = sorted(set(r["init_scale"] for r in runs_data.values()))
    unique_models = sorted(set(r.get("model_index", 0) for r in runs_data.values()))

    # Define color mapping for initialization scales
    if len(unique_scales) <= 6:
        palette = plt.cm.tab10(np.linspace(0, 0.8, max(len(unique_scales), 1)))
        scale_colors = {s: palette[i] for i, s in enumerate(unique_scales)}
    else:
        palette = plt.cm.plasma(np.linspace(0.1, 0.9, len(unique_scales)))
        scale_colors = {s: palette[i] for i, s in enumerate(unique_scales)}

    # Grid values
    x_coords = np.array(grid_data["x_coords"])
    y_coords = np.array(grid_data["y_coords"])
    xx, yy = np.meshgrid(x_coords, y_coords)

    loss_vals = np.array(grid_data["losses"].get("train", grid_data["losses"]["test"])).reshape(
        len(y_coords), len(x_coords)
    )
    norm_vals = np.array(grid_data["weight_norms"]).reshape(len(y_coords), len(x_coords))

    show_both = panels == "both"
    if show_both:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.8), dpi=dpi)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8.0, 6.8), dpi=dpi)
        ax2 = None

    # Panel 1: Loss Landscape
    render_joint_landscape_panel(
        ax=ax1,
        grid_x=xx,
        grid_y=yy,
        landscape_vals=loss_vals,
        metric_type="loss",
        runs_data=runs_data,
        pc1_label=pc1_label,
        pc2_label=pc2_label,
        scale_colors=scale_colors,
        min_loss=min_loss,
        max_loss=max_loss,
        landscape_cmap=landscape_cmap,
        xscale=xscale,
        yscale=yscale,
        linthresh=linthresh,
        linscale=linscale,
    )
    ax1.set_title("(a) 2D Training Loss Landscape", fontsize=11, fontweight="bold", pad=10)

    # Panel 2: Weight Norm
    if show_both and ax2 is not None:
        render_joint_landscape_panel(
            ax=ax2,
            grid_x=xx,
            grid_y=yy,
            landscape_vals=norm_vals,
            metric_type="norm",
            runs_data=runs_data,
            pc1_label=pc1_label,
            pc2_label=pc2_label,
            scale_colors=scale_colors,
            min_loss=min_loss,
            max_loss=max_loss,
            landscape_cmap=landscape_cmap,
            xscale=xscale,
            yscale=yscale,
            linthresh=linthresh,
            linscale=linscale,
        )
        ax2.set_title(r"(b) 2D Weight Norm Surface $\|\theta\|_2$", fontsize=11, fontweight="bold", pad=10)

    # Global Title
    fig.suptitle(
        f"Joint PCA Optimization Trajectories Across {len(runs_data)} Runs\n"
        f"(Top 2 Components Explain {total_evr:.1f}% Total Path Variance)",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )

    # Shared Legend below figure
    legend_elements = []
    # Scale elements
    for s in unique_scales:
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color=scale_colors[s],
                lw=2,
                label=f"Init Scale {s:g}",
            )
        )
    # Model Index elements (if multiple models)
    if len(unique_models) > 1:
        for m in unique_models:
            ls = "-" if m == 0 else "--"
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    color="gray",
                    lw=2,
                    linestyle=ls,
                    label=f"Model Index {m}",
                )
            )
    # Markers
    legend_elements.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markeredgecolor="white",
            markersize=7,
            label="Start (Step 0)",
        )
    )
    legend_elements.append(
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="gray",
            markeredgecolor="black",
            markersize=10,
            label="End (Minimizer)",
        )
    )

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=min(len(legend_elements), 6),
        frameon=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor="#cccccc",
        fontsize=9,
    )

    fig.tight_layout(rect=[0, 0.05, 1, 0.93])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    print(f"Saved vector PDF joint landscape to: {pdf_path}")

    png_path = output_path.with_suffix(".png")
    fig.savefig(png_path, format="png", dpi=dpi, bbox_inches="tight")
    print(f"Saved high-res PNG joint landscape to: {png_path}")

    plt.close(fig)


@app.command()
def main(
    run_ids: Annotated[
        list[str] | None,
        typer.Option(
            "--run-ids",
            "-r",
            help="List of MLflow run IDs to include in the joint trajectory landscape.",
        ),
    ] = None,
    group_id: Annotated[
        str | None,
        typer.Option("--group-id", "-g", help="Group identifier for the shared visualization."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Target output file path (.pdf or .png)."),
    ] = None,
    panels: Annotated[
        str,
        typer.Option("--panels", "-p", help="Panels to display: 'both' (loss + norm) or 'loss'."),
    ] = "both",
    xscale: Annotated[
        str, typer.Option("--xscale", help="Axis scaling for PC1 ('symlog', 'linear', 'log').")
    ] = "symlog",
    yscale: Annotated[
        str, typer.Option("--yscale", help="Axis scaling for PC2 ('symlog', 'linear', 'log').")
    ] = "symlog",
    linthresh: Annotated[
        float, typer.Option("--linthresh", help="Linear threshold near 0 for symlog scale.")
    ] = 0.05,
    linscale: Annotated[
        float, typer.Option("--linscale", help="Linear scale compression factor for symlog scale.")
    ] = 1.0,
    min_loss: Annotated[
        float, typer.Option("--min-loss", help="Lower loss clamp for logarithmic contours.")
    ] = 1e-8,
    max_loss: Annotated[
        float, typer.Option("--max-loss", help="Upper loss clamp for logarithmic contours.")
    ] = 10.0,
    landscape_cmap: Annotated[
        str, typer.Option("--cmap", "-c", help="Matplotlib colormap for loss landscape.")
    ] = "Blues_r",
    tracking_uri: Annotated[
        str, typer.Option("--tracking-uri", "-u", help="MLflow tracking server URI.")
    ] = DEFAULT_TRACKING_URI,
    force_recompute: Annotated[
        bool, typer.Option("--force", "-f", help="Force recomputing joint PCA/grid extraction.")
    ] = False,
    dpi: Annotated[
        int, typer.Option("--dpi", help="Raster resolution for PNG output.")
    ] = 300,
) -> None:
    """CLI entrypoint for rendering multi-trajectory joint PCA loss landscapes."""
    target_runs: list[str] = []
    if run_ids:
        for item in run_ids:
            target_runs.extend([x.strip() for x in item.split(",") if x.strip()])

    if not target_runs and not group_id:
        raise typer.BadParameter("Please provide either --run-ids or --group-id.")

    gid = group_id or generate_group_id(target_runs)
    proj_path = joint_trajectory_projection_path(gid)
    grid_path = joint_trajectory_grid_path(gid)

    if not proj_path.exists() or not grid_path.exists() or force_recompute:
        if not target_runs:
            raise typer.BadParameter(f"Artifacts missing for '{gid}'. Please specify --run-ids to extract.")
        print(f"Precomputed joint artifacts missing or force flag set for '{gid}'. Running extraction...")
        extract_main(
            run_ids=target_runs,
            group_id=gid,
            grid_spacing="symlog",
            linthresh=linthresh,
            tracking_uri=tracking_uri,
            force=force_recompute,
        )

    target_output = output or plots_joint_trajectories_path(gid)
    plot_joint_trajectories_landscape(
        group_id=gid,
        output_path=target_output,
        panels=panels,
        min_loss=min_loss,
        max_loss=max_loss,
        landscape_cmap=landscape_cmap,
        xscale=xscale,
        yscale=yscale,
        linthresh=linthresh,
        linscale=linscale,
        dpi=dpi,
    )


if __name__ == "__main__":
    app()
