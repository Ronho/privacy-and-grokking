#!/usr/bin/env python3
"""plot_pca_loss_landscape.py.

Visualizes the 2D loss landscape and weight-norm surface along the top principal
components, overlaying the model's optimization trajectory and milestone checkpoints.

Uses a logarithmic loss scale (spanning from 10^1 down to 10^-8) with mathematical
power-of-ten colorbar formatting, accurately exposing low-loss grokking plateaus.

Required Inputs:
  - cache/runs/{run_id}/trajectories/grid.json: 2D loss and weight-norm grid values.
  - cache/runs/{run_id}/trajectories/projection.json: Projected trajectory coordinates and metadata.

Generated Outputs:
  - plots/pca_loss_landscape/pca_loss_landscape_{run_id}.pdf (vector)
  - plots/pca_loss_landscape/pca_loss_landscape_{run_id}.png (300 dpi raster)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import LogFormatterMathtext, LogLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from privacy_and_grokking.artifacts.local import (
    plots_pca_loss_landscape_path,
    trajectory_grid_path,
    trajectory_projection_path,
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


def render_landscape_axis(
    ax: plt.Axes,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    landscape_vals: np.ndarray,
    metric_type: str,
    df_pca: pd.DataFrame,
    pc1_label: str,
    pc2_label: str,
    step_colormap: str = "plasma",
    tag_interval: int = 50_000,
    min_loss: float = 1e-8,
    max_loss: float = 10.0,
    num_levels: int = 50,
) -> None:
    """Renders a single square landscape panel with contours, sampling grid, and trajectory."""
    ax.set_box_aspect(1)

    pc1 = df_pca["pc1"].values
    pc2 = df_pca["pc2"].values
    steps_pca = df_pca["step"].values

    # 1. Background landscape contours
    valid = landscape_vals[np.isfinite(landscape_vals)]
    if len(valid) == 0:
        raise ValueError("No finite values in landscape grid!")

    if metric_type == "loss":
        # Logarithmic normalization with floor clamped at min_loss (e.g. 1e-8)
        # and ceiling clamped at max_loss (e.g. 10.0)
        actual_pos = valid[valid > 0]
        eff_min = min(min_loss, float(np.min(actual_pos))) if len(actual_pos) > 0 else min_loss
        eff_max = max(max_loss, float(np.max(valid)))

        # Clamp grid values to strictly positive range for LogNorm
        clamped_vals = np.clip(landscape_vals, eff_min, eff_max)
        norm = LogNorm(vmin=eff_min, vmax=eff_max)
        min_exp = int(np.floor(np.log10(eff_min)))
        max_exp = int(np.ceil(np.log10(eff_max)))
        levels = 10.0 ** np.arange(min_exp, max_exp + 1)
        cmap = "Blues_r"
        cf = ax.contourf(
            grid_x,
            grid_y,
            clamped_vals,
            levels=levels,
            cmap=cmap,
            norm=norm,
            extend="both",
            zorder=1,
        )
    else:  # weight_norm
        v_min = float(np.min(valid))
        v_max = float(np.max(valid))
        norm = Normalize(vmin=v_min, vmax=v_max)
        levels = np.linspace(v_min, v_max, num_levels)
        cmap = "PuBu_r"
        cf = ax.contourf(
            grid_x,
            grid_y,
            landscape_vals,
            levels=levels,
            cmap=cmap,
            norm=norm,
            extend="neither",
            zorder=1,
        )

    # 2. Evaluated sampling grid lines and vertex dots
    for i in range(grid_x.shape[0]):
        ax.plot(grid_x[i, :], grid_y[i, :], color="#64748b", alpha=0.15, linewidth=0.3, zorder=1.5)
    for j in range(grid_x.shape[1]):
        ax.plot(grid_x[:, j], grid_y[:, j], color="#64748b", alpha=0.15, linewidth=0.3, zorder=1.5)
    ax.scatter(
        grid_x.ravel(), grid_y.ravel(), s=2.0, color="#334155", alpha=0.3, linewidths=0, zorder=1.6
    )

    # 3. Trajectory connecting lines
    ax.plot(pc1, pc2, color="#0f172a", linestyle="-", linewidth=2.0, alpha=0.7, zorder=2)
    ax.plot(pc1, pc2, color="#f8fafc", linestyle="-", linewidth=1.2, alpha=0.95, zorder=3)

    # 4. Checkpoint markers colored by step
    sc = ax.scatter(
        pc1,
        pc2,
        c=steps_pca,
        cmap=step_colormap,
        s=26,
        edgecolors="#1e293b",
        linewidths=0.5,
        zorder=4,
    )

    # 5. Milestone checkpoints
    tagged_indices = []
    for idx, step_val in enumerate(steps_pca):
        is_start = idx == 0
        is_end = idx == len(steps_pca) - 1
        is_milestone = (step_val % tag_interval == 0) and (step_val > 0)
        if is_start or is_end or is_milestone:
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

    # 6. Milestone step annotations with smart collision avoidance
    x_min, x_max = float(np.min(grid_x)), float(np.max(grid_x))
    y_min, y_max = float(np.min(grid_y)), float(np.max(grid_y))
    x_span = max(1e-6, x_max - x_min)
    y_span = max(1e-6, y_max - y_min)

    # Group milestones that are coincident in normalized space (< 2.5% of axis span)
    grouped_milestones: list[dict] = []
    for idx in tagged_indices:
        p1 = float(pc1[idx])
        p2 = float(pc2[idx])
        step_val = int(steps_pca[idx])
        u = (p1 - x_min) / x_span
        v = (p2 - y_min) / y_span

        matched = False
        for g in grouped_milestones:
            dist = np.sqrt((u - g["u"]) ** 2 + (v - g["v"]) ** 2)
            if dist < 0.025:
                g["steps"].append(step_val)
                # Keep latest checkpoint coordinates as primary anchor
                g["p1"] = p1
                g["p2"] = p2
                g["u"] = u
                g["v"] = v
                matched = True
                break
        if not matched:
            grouped_milestones.append(
                {
                    "steps": [step_val],
                    "p1": p1,
                    "p2": p2,
                    "u": u,
                    "v": v,
                }
            )

    # Offset candidates in points (dx, dy)
    candidates = [
        (-26, 14),   # upper-left
        (-26, -16),  # lower-left
        (22, 14),    # upper-right
        (22, -16),   # lower-right
        (0, 24),     # top
        (0, -26),    # bottom
        (-30, 0),    # left
        (30, 0),     # right
    ]
    norm_scale = 300.0  # approximate square panel side length in points
    placed_boxes: list[tuple[float, float, float, float]] = []

    for g in grouped_milestones:
        tag_str = ", ".join(format_step_tag(s) for s in g["steps"])
        u_pt = g["u"]
        v_pt = g["v"]
        p1 = g["p1"]
        p2 = g["p2"]

        best_cand = candidates[0]
        best_penalty = float("inf")
        box_w = max(0.09, 0.04 + 0.018 * len(tag_str))
        box_h = 0.055

        for dx, dy in candidates:
            uc = u_pt + dx / norm_scale
            vc = v_pt + dy / norm_scale
            penalty = 0.0

            # 1. Axis boundary penalty
            if uc < 0.04 or uc > 0.96 or vc < 0.04 or vc > 0.96:
                penalty += 3000.0

            # 2. Overlap with previously placed badge bounding boxes
            for pu, pv, pw, ph in placed_boxes:
                du = abs(uc - pu)
                dv = abs(vc - pv)
                req_u = 0.5 * (box_w + pw) + 0.015
                req_v = 0.5 * (box_h + ph) + 0.015
                if du < req_u and dv < req_v:
                    penalty += 2000.0 + (req_u - du) * 8000.0 + (req_v - dv) * 8000.0

            # 3. Proximity to other milestone coordinates
            for og in grouped_milestones:
                if og is g:
                    continue
                du_m = abs(uc - og["u"])
                dv_m = abs(vc - og["v"])
                if du_m < 0.06 and dv_m < 0.04:
                    penalty += 500.0

            # 4. Tie-breaking preference for compact offsets
            penalty += (dx**2 + dy**2) * 0.01

            if penalty < best_penalty:
                best_penalty = penalty
                best_cand = (dx, dy)

        dx_chosen, dy_chosen = best_cand
        chosen_uc = u_pt + dx_chosen / norm_scale
        chosen_vc = v_pt + dy_chosen / norm_scale
        placed_boxes.append((chosen_uc, chosen_vc, box_w, box_h))

        ax.annotate(
            tag_str,
            xy=(p1, p2),
            xytext=(dx_chosen, dy_chosen),
            textcoords="offset points",
            fontsize=8.0,
            fontweight="bold",
            color="#0f172a",
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="#ffffff",
                alpha=0.90,
                edgecolor="#94a3b8",
                linewidth=0.6,
            ),
            zorder=6,
        )

    # 7. Start & End Markers
    ax.scatter(
        [pc1[0]],
        [pc2[0]],
        color="#22c55e",
        edgecolors="#0f172a",
        s=85,
        marker="o",
        linewidths=0.9,
        zorder=5,
    )
    ax.scatter(
        [pc1[-1]],
        [pc2[-1]],
        color="#ef4444",
        edgecolors="#0f172a",
        s=95,
        marker="*",
        linewidths=0.9,
        zorder=5,
    )

    # 8. Dual Colorbars via make_axes_locatable
    fig = ax.get_figure()
    divider = make_axes_locatable(ax)

    # (a) Landscape colorbar
    cax_land = divider.append_axes("right", size="5%", pad=0.08)
    if metric_type == "loss":
        cbar_land = fig.colorbar(
            cf,
            cax=cax_land,
            orientation="vertical",
            format=LogFormatterMathtext(),
        )
        cbar_land.locator = LogLocator(base=10.0, subs=(1.0,))
        cbar_land.update_ticks()
        cax_land.set_title("Loss", fontsize=9.0, pad=5, fontweight="bold")
    else:
        cbar_land = fig.colorbar(cf, cax=cax_land, orientation="vertical")
        cax_land.set_title(r"$\|\theta\|_2$", fontsize=9.5, pad=5, fontweight="bold")

    cbar_land.ax.tick_params(labelsize=7.5, pad=2)

    # (b) Step colorbar
    cax_step = divider.append_axes("right", size="5%", pad=0.38)
    cbar_step = fig.colorbar(sc, cax=cax_step, orientation="vertical")
    cax_step.set_title("Step", fontsize=9.0, pad=5, fontweight="bold")

    # Step colorbar tick formatting
    step_min, step_max = float(np.min(steps_pca)), float(np.max(steps_pca))
    tick_candidates = np.linspace(step_min, step_max, 6)
    cbar_step.set_ticks(tick_candidates)
    cbar_step.set_ticklabels([format_step_tag(int(t)) for t in tick_candidates])
    cbar_step.ax.tick_params(labelsize=7.5, pad=2)

    # Axis Labels & Limits
    ax.set_xlabel(pc1_label, fontsize=10.0, labelpad=5, fontweight="bold")
    ax.set_ylabel(pc2_label, fontsize=10.0, labelpad=5, fontweight="bold")
    ax.tick_params(labelsize=8.5)
    ax.grid(False)


def plot_pca_loss_landscape(
    run_id: str,
    output_path: Path | None = None,
    tag_interval: int = 50_000,
    min_loss: float = 1e-8,
    max_loss: float = 10.0,
    num_levels: int = 50,
    save_png: bool = True,
    dpi: int = 300,
) -> tuple[Path, Path | None]:
    """Generates a 2-panel figure showing Loss and Weight Norm landscapes for a given run."""
    grid_file = trajectory_grid_path(run_id)
    proj_file = trajectory_projection_path(run_id)

    if not grid_file.exists():
        msg = f"Trajectory grid file not found at {grid_file}. Run extract_trajectories first."
        raise FileNotFoundError(msg)
    if not proj_file.exists():
        msg = f"Projection file not found at {proj_file}. Run extract_trajectories first."
        raise FileNotFoundError(msg)

    grid_data = json.loads(grid_file.read_bytes())
    proj_data = json.loads(proj_file.read_bytes())

    x_coords = np.array(grid_data["x_coords"], dtype=float)
    y_coords = np.array(grid_data["y_coords"], dtype=float)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)

    raw_loss = np.array(grid_data["losses"]["train"], dtype=float)
    train_loss_grid = raw_loss.reshape(len(y_coords), len(x_coords))

    raw_norm = np.array(grid_data["weight_norms"], dtype=float)
    weight_norm_grid = raw_norm.reshape(len(y_coords), len(x_coords))

    trajectories = proj_data["trajectories"]
    steps = trajectories["steps"]
    pca_coords = np.array(trajectories["pca_coords"])
    df_pca = pd.DataFrame(
        {
            "step": steps,
            "pc1": pca_coords[:, 0],
            "pc2": pca_coords[:, 1],
        }
    )

    evr = proj_data.get("explained_variance_ratio", [])
    pc1_ratio = evr[0] * 100 if len(evr) > 0 else 0.0
    pc2_ratio = evr[1] * 100 if len(evr) > 1 else 0.0
    pc1_label = f"PC 1 ({pc1_ratio:.1f}%)"
    pc2_label = f"PC 2 ({pc2_ratio:.1f}%)"

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.5))

    # Panel 1: Loss Landscape
    render_landscape_axis(
        ax=axes[0],
        grid_x=grid_x,
        grid_y=grid_y,
        landscape_vals=train_loss_grid,
        metric_type="loss",
        df_pca=df_pca,
        pc1_label=pc1_label,
        pc2_label=pc2_label,
        tag_interval=tag_interval,
        min_loss=min_loss,
        max_loss=max_loss,
        num_levels=num_levels,
    )

    # Panel 2: Weight Norm Landscape
    render_landscape_axis(
        ax=axes[1],
        grid_x=grid_x,
        grid_y=grid_y,
        landscape_vals=weight_norm_grid,
        metric_type="weight_norm",
        df_pca=df_pca,
        pc1_label=pc1_label,
        pc2_label=pc2_label,
        tag_interval=tag_interval,
        min_loss=min_loss,
        max_loss=max_loss,
        num_levels=num_levels,
    )

    fig.subplots_adjust(wspace=0.42)

    pdf_path = output_path or plots_pca_loss_landscape_path(run_id, fmt="pdf")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    print(f"Saved PDF plot to: {pdf_path}")

    png_path: Path | None = None
    if save_png:
        png_path = pdf_path.with_suffix(".png")
        fig.savefig(png_path, format="png", dpi=dpi, bbox_inches="tight")
        print(f"Saved PNG plot to: {png_path}")

    plt.close(fig)
    return pdf_path, png_path


@app.command()
def main(
    run_id: Annotated[
        str,
        typer.Option(
            "--run-id", "-r", help="MLflow run ID whose loss landscape should be plotted."
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Custom output destination path for the PDF."),
    ] = None,
    tag_interval: Annotated[
        int,
        typer.Option(
            "--tag-interval", "-t", help="Step interval for displaying milestone checkpoint tags."
        ),
    ] = 50_000,
    min_loss: Annotated[
        float,
        typer.Option("--min-loss", help="Lower bound floor for logarithmic loss scale."),
    ] = 1e-8,
    max_loss: Annotated[
        float,
        typer.Option("--max-loss", help="Upper bound ceiling for logarithmic loss scale."),
    ] = 10.0,
    num_levels: Annotated[
        int,
        typer.Option("--num-levels", help="Number of contour levels."),
    ] = 50,
    png: Annotated[
        bool,
        typer.Option("--png/--no-png", help="Generate accompanying 300 dpi PNG alongside PDF."),
    ] = True,
    dpi: Annotated[
        int,
        typer.Option("--dpi", help="Rasterization resolution for PNG output."),
    ] = 300,
) -> None:
    """CLI command to render 2D PCA loss & weight norm landscapes with logarithmic scaling."""
    plot_pca_loss_landscape(
        run_id=run_id,
        output_path=output,
        tag_interval=tag_interval,
        min_loss=min_loss,
        max_loss=max_loss,
        num_levels=num_levels,
        save_png=png,
        dpi=dpi,
    )


if __name__ == "__main__":
    app()
