#!/usr/bin/env python3
"""plot_jump_planes.py.

Visualizes the 2D local subspace trajectory and loss landscape around each optimization jump (slingshot).
For each jump:
  - x-axis (u1): Direction of incoming trajectory drift towards the minimum
  - y-axis (u2): Direction of the jump (Gram-Schmidt orthogonalized against u1)
  - Background: 2D loss contours on a logarithmic scale (10^-8 to 10^2)
  - Trajectory: Incoming approach (x-axis), 90-degree orthogonal jump (y-axis), and outgoing path

Outputs:
  - cache/plots/jump_planes/jump_planes_2d_{run_id}.pdf
  - cache/plots/jump_planes/jump_planes_2d_{run_id}.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
import typer
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext, LogLocator

from privacy_and_grokking.artifacts.cache import DEFAULT_TRACKING_URI
from privacy_and_grokking.artifacts.local import (
    jump_planes_grid_path,
    plots_jump_planes_dir,
    plots_jump_planes_path,
    plots_jump_planes_legend_path,
)
from privacy_and_grokking.utils import Logger
from visualizations.preprocessing.extract_jump_planes import extract_jump_planes

app = typer.Typer(pretty_exceptions_enable=False)


def plot_jump_planes(
    run_id: str,
    output_dir: Path | None = None,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    force_extract: bool = False,
    short_descriptor: str | None = None,
) -> tuple[Path, Path]:
    """Plots a publication-grade 2x4 grid of 2D local subspace loss landscapes and jump trajectories."""
    legend_texts = []
    data_file = jump_planes_grid_path(run_id)

    if not data_file.exists() or force_extract:
        print(f"Extraction file missing. Extracting 2D jump planes for run '{run_id}'...")
        data = extract_jump_planes(run_id=run_id, tracking_uri=tracking_uri, force=True)
    else:
        data = json.loads(data_file.read_bytes())

    jumps = data["jumps"]
    n_jumps = len(jumps)
    cols = 4 if n_jumps >= 4 else n_jumps
    rows = int(np.ceil(n_jumps / cols))

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 14,
    })

    fig, axes = plt.subplots(
        rows, cols, figsize=(4.0 * cols, 3.6 * rows), squeeze=False, dpi=300
    )

    min_loss = 1e-8
    max_loss = 100.0
    norm = LogNorm(vmin=min_loss, vmax=max_loss)
    levels = np.logspace(-8, 2, 21)
    power_10_levels = np.logspace(-8, 2, 11)
    roman_numerals = ["(I)", "(II)", "(III)", "(IV)", "(V)", "(VI)", "(VII)", "(VIII)", "(IX)", "(X)"]

    cf = None

    for idx, jump in enumerate(jumps):
        r, c = divmod(idx, cols)
        ax = axes[r][c]

        s_pre = jump["step_pre"]
        s_post = jump["step_post"]
        turn_angle = jump["turn_angle_deg"]
        theta_rot = jump["theta_rot_deg"]
        jump_dist = jump["jump_distance"]

        grid_x = np.array(jump["grid_x"])
        grid_y = np.array(jump["grid_y"])
        loss_grid = np.clip(np.array(jump["loss_grid"]), min_loss, max_loss)

        X, Y = np.meshgrid(grid_x, grid_y)

        # 1. Background Loss Contours
        cf = ax.contourf(X, Y, loss_grid, levels=levels, norm=norm, cmap="PuBu_r", alpha=0.9, extend="both")
        cs = ax.contour(X, Y, loss_grid, levels=power_10_levels, colors="white", linewidths=0.6, alpha=0.5)
        try:
            ax.clabel(cs, inline=True, fontsize=6, fmt=LogFormatterMathtext(), colors='white')
        except Exception:
            pass

        # 2. Draw Trajectory Segments
        traj = jump["traj_points"]
        xs = np.array([p["x"] for p in traj])
        ys = np.array([p["y"] for p in traj])
        steps = np.array([p["step"] for p in traj])

        # Incoming segment (steps <= s_pre)
        mask_in = steps <= s_pre
        ax.plot(
            xs[mask_in],
            ys[mask_in],
            color="#33a02c",
            linewidth=2.0,
            linestyle="-",
            label="Incoming drift ($v_{\\mathrm{in}}$)",
            zorder=6,
        )
        ax.scatter(xs[mask_in], ys[mask_in], color="#33a02c", edgecolors="black", s=12, zorder=7)

        # The Jump vector (from (0,0) to (x_post, y_post))
        idx_pre = int(np.where(steps == s_pre)[0][0])
        idx_post = int(np.where(steps == s_post)[0][0])
        x_pre, y_pre = xs[idx_pre], ys[idx_pre]
        x_post, y_post = xs[idx_post], ys[idx_post]

        ax.annotate(
            "",
            xy=(x_post, y_post),
            xytext=(x_pre, y_pre),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#e31a1c",
                lw=2.5,
                mutation_scale=14,
            ),
            zorder=9,
        )

        # Outgoing segment (steps >= s_post)
        mask_out = steps >= s_post
        ax.plot(
            xs[mask_out],
            ys[mask_out],
            color="#33a02c",
            linewidth=2.0,
            linestyle="--",
            label="Outgoing drift ($v_{\\mathrm{out}}$)",
            zorder=6,
        )
        ax.scatter(xs[mask_out], ys[mask_out], color="#33a02c", edgecolors="black", s=12, zorder=7)

        # Key state markers
        ax.scatter(
            [0.0],
            [0.0],
            marker="o",
            s=60,
            color="#e31a1c",
            edgecolors="black",
            zorder=10,
            label=r"$\theta_{\mathrm{pre}}$ (Minimum)",
        )
        ax.scatter(
            [x_post],
            [y_post],
            marker="*",
            s=120,
            color="#fdbf6f",
            edgecolors="black",
            zorder=10,
            label=r"$\theta_{\mathrm{post}}$ (Jump Target)",
        )

        # Axes & Reference lines
        ax.axhline(0.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.axvline(0.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

        # Collect legend text instead of plotting it
        roman_num = roman_numerals[idx] if idx < len(roman_numerals) else f"({idx+1})"
        legend_texts.append(
            f"{roman_num} Step {s_pre:,} to {s_post:,}\n"
            f"Angle (v_in, v_jump): {turn_angle:.1f} degrees\n"
            f"Jump Distance ||v_jump||: {jump_dist:.1f}\n"
            f"Delta Theta: {theta_rot:.1f} degrees\n"
        )

        ax.set_title(
            f"{roman_num} Step {s_pre:,} to {s_post:,}",
            fontweight="bold",
            pad=6,
        )
        ax.grid(True, linestyle="--", alpha=0.25, color="white")

        if r == rows - 1:
            ax.set_xlabel(r"Incoming Direction $u_1$", fontweight="bold")
        if c == 0:
            ax.set_ylabel(r"Orthogonal Jump $u_2$", fontweight="bold")

    # Hide unused subplots
    for idx in range(n_jumps, rows * cols):
        r, c = divmod(idx, cols)
        fig.delaxes(axes[r][c])

    fig.subplots_adjust(right=0.95, top=0.90, hspace=0.32, wspace=0.25)

    if output_dir is None:
        target_dir = plots_jump_planes_dir()
    else:
        target_dir = output_dir

    target_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = plots_jump_planes_path(run_id, "pdf", short_descriptor)
    png_path = plots_jump_planes_path(run_id, "png", short_descriptor)
    txt_path = plots_jump_planes_legend_path(run_id, short_descriptor)

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    txt_path.write_text("\n".join(legend_texts))

    print(f"Saved 2D Jump Planes PDF plot to: {pdf_path}")
    print(f"Saved 2D Jump Planes PNG plot to: {png_path}")
    print(f"Saved 2D Jump Planes Legend data to: {txt_path}")
    return pdf_path, png_path


@app.command()
def main(
    run_id: Annotated[str, typer.Option("--run-id", help="Target MLflow Run ID.")],
    output_dir: Annotated[
        Path | None, typer.Option("--output-dir", help="Optional custom output directory.")
    ] = None,
    tracking_uri: Annotated[
        str, typer.Option("--tracking-uri", help="MLflow server URI.")
    ] = DEFAULT_TRACKING_URI,
    force_extract: Annotated[
        bool, typer.Option("--force-extract", help="Force re-extraction of jump planes data.")
    ] = False,
) -> None:
    """CLI endpoint to plot 2D local subspace trajectories and loss landscapes across jumps."""
    Logger().setup()
    plot_jump_planes(
        run_id=run_id,
        output_dir=output_dir,
        tracking_uri=tracking_uri,
        force_extract=force_extract,
    )


if __name__ == "__main__":
    app()
