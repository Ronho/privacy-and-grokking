#!/usr/bin/env python3
"""plot_filter_normalized_landscape.py.

Visualizes the 2D loss landscape and weight-norm surface around a trained checkpoint model
using filter-wise normalized random perturbation directions (Li et al., NeurIPS 2018).

Eliminates scale-invariance distortions (e.g. from ReLU/BatchNorm/LayerNorm) to present
an objective, publication-ready visual assessment of local sharpness, convexity, and basin geometry.

Required Inputs:
  - cache/runs/{run_id}/filter_normalized/grid_{step_or_final}.json:
      Precomputed 2D loss and weight-norm grid values.

Generated Outputs:
  - plots/filter_normalized_landscape/filter_normalized_landscape_{run_id}_{step_or_final}.pdf
  - plots/filter_normalized_landscape/filter_normalized_landscape_{run_id}_{step_or_final}.png
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import typer
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import LogFormatterMathtext, LogLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from privacy_and_grokking.artifacts.local import (
    filter_normalized_grid_path,
    plots_filter_normalized_landscape_path,
)

app = typer.Typer(pretty_exceptions_enable=False)


def render_landscape_panel(
    ax: plt.Axes,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    landscape_vals: np.ndarray,
    metric_type: str,
    step_label: str,
    contour_mode: str = "lines",
    cmap_override: str | None = None,
    min_loss: float = 1e-8,
    max_loss: float = 10.0,
    num_levels: int = 30,
    show_clabel: bool = True,
) -> None:
    """Renders a single square landscape panel with contour lines or filled contours."""
    ax.set_box_aspect(1)

    valid = landscape_vals[np.isfinite(landscape_vals)]
    if len(valid) == 0:
        raise ValueError("No finite values in landscape grid!")

    # 1. Setup normalization, levels, and colormap
    if metric_type == "loss":
        actual_pos = valid[valid > 0]
        eff_min = min(min_loss, float(np.min(actual_pos))) if len(actual_pos) > 0 else min_loss
        eff_max = max(max_loss, float(np.max(valid)))

        min_exp = int(np.floor(np.log10(eff_min)))
        max_exp = int(np.ceil(np.log10(eff_max)))
        clamped_vals = np.clip(landscape_vals, eff_min, eff_max)
        norm = LogNorm(vmin=eff_min, vmax=eff_max)
        levels = 10.0 ** np.arange(min_exp, max_exp + 1)
        cmap = cmap_override or ("viridis" if contour_mode == "lines" else "Blues_r")
    else:  # weight_norm
        v_min = float(np.min(valid))
        v_max = float(np.max(valid))
        norm = Normalize(vmin=v_min, vmax=v_max)
        levels = np.linspace(v_min, v_max, num_levels)
        cmap = cmap_override or ("viridis" if contour_mode == "lines" else "PuBu_r")
        clamped_vals = landscape_vals

    mappable = None

    # 2. Render contours based on contour_mode
    if contour_mode in ("filled", "both"):
        cf = ax.contourf(
            grid_x,
            grid_y,
            clamped_vals,
            levels=levels,
            cmap=cmap,
            norm=norm,
            extend="both" if metric_type == "loss" else "neither",
            zorder=1,
        )
        mappable = cf

    if contour_mode in ("lines", "both"):
        line_cmap = cmap if contour_mode == "lines" else None
        line_color = None if contour_mode == "lines" else "#1e293b"
        cs = ax.contour(
            grid_x,
            grid_y,
            clamped_vals,
            levels=levels,
            cmap=line_cmap,
            colors=line_color,
            norm=norm,
            linewidths=1.6 if contour_mode == "lines" else 0.8,
            alpha=1.0 if contour_mode == "lines" else 0.65,
            zorder=2,
        )
        if mappable is None:
            mappable = cs

        if show_clabel:
            # Inline numeric labels on contour lines (powers of 10 for loss)
            def fmt_label(x: float) -> str:
                if metric_type == "loss":
                    exp = int(np.round(np.log10(x)))
                    return rf"$10^{{{exp}}}$"
                if x < 10.0:
                    return f"{x:.2f}"
                return f"{x:.1f}"

            ax.clabel(
                cs,
                inline=True,
                fontsize=7.5,
                fmt=fmt_label,
                colors=None if contour_mode == "lines" else "#0f172a",
                inline_spacing=4,
            )

    # 3. Sampling grid lines (subtle in line mode)
    grid_alpha = 0.08 if contour_mode == "lines" else 0.15
    for i in range(grid_x.shape[0]):
        ax.plot(
            grid_x[i, :], grid_y[i, :], color="#64748b", alpha=grid_alpha, linewidth=0.3, zorder=1.5
        )
    for j in range(grid_x.shape[1]):
        ax.plot(
            grid_x[:, j], grid_y[:, j], color="#64748b", alpha=grid_alpha, linewidth=0.3, zorder=1.5
        )

    # 4. Center model marker theta* at (0, 0)
    ax.scatter(
        [0.0],
        [0.0],
        color="#ef4444",
        edgecolors="#0f172a",
        s=100,
        marker="*",
        linewidths=0.9,
        zorder=5,
    )
    ax.annotate(
        f"$\\theta^*$ ({step_label})",
        xy=(0.0, 0.0),
        xytext=(0, 18),
        textcoords="offset points",
        fontsize=8.5,
        fontweight="bold",
        color="#0f172a",
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="#ffffff",
            alpha=0.92,
            edgecolor="#94a3b8",
            linewidth=0.6,
        ),
        zorder=6,
    )

    # 5. Colorbar attached via make_axes_locatable
    fig = ax.get_figure()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.08)

    if metric_type == "loss":
        cbar = fig.colorbar(
            mappable,
            cax=cax,
            orientation="vertical",
            format=LogFormatterMathtext(),
        )
        cbar.locator = LogLocator(base=10.0, subs=(1.0,))
        cbar.update_ticks()
        cax.set_title("Loss", fontsize=9.0, pad=5, fontweight="bold")
    else:
        cbar = fig.colorbar(mappable, cax=cax, orientation="vertical")
        cax.set_title(r"$\|\theta\|_2$", fontsize=9.5, pad=5, fontweight="bold")

    cbar.ax.tick_params(labelsize=7.5, pad=2)

    # 6. Axis Labels & Styling
    ax.set_xlabel(
        r"Direction 1 ($\delta$, filter-norm)", fontsize=9.5, labelpad=5, fontweight="bold"
    )
    ax.set_ylabel(
        r"Direction 2 ($\eta$, filter-norm)", fontsize=9.5, labelpad=5, fontweight="bold"
    )
    ax.tick_params(labelsize=8.5)
    ax.grid(False)


def plot_filter_normalized_landscape(
    run_id: str,
    step: int | None = None,
    output_path: Path | None = None,
    contour_mode: str = "lines",
    cmap: str | None = None,
    show_clabel: bool = True,
    min_loss: float = 1e-8,
    max_loss: float = 10.0,
    num_levels: int = 30,
    save_png: bool = True,
    dpi: int = 300,
) -> tuple[Path, Path | None]:
    """Generates a 2-panel figure showing Loss & Weight Norm filter-normalized landscapes."""
    step_key = step if step is not None else "final"
    grid_file = filter_normalized_grid_path(run_id, step_key)

    if not grid_file.exists():
        msg = f"Grid file not found at {grid_file}. Run extract_filter_normalized_landscape first."
        raise FileNotFoundError(msg)

    grid_data = json.loads(grid_file.read_bytes())
    x_coords = np.array(grid_data["x_coords"], dtype=float)
    y_coords = np.array(grid_data["y_coords"], dtype=float)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)

    raw_loss = np.array(grid_data["losses"]["train"], dtype=float)
    train_loss_grid = raw_loss.reshape(len(y_coords), len(x_coords))

    raw_norm = np.array(grid_data["weight_norms"], dtype=float)
    weight_norm_grid = raw_norm.reshape(len(y_coords), len(x_coords))

    actual_step = grid_data.get("step", 0)
    step_label = f"Step {actual_step:,}" if actual_step > 0 else "Init"

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))

    # Panel 1: Loss Landscape
    render_landscape_panel(
        ax=axes[0],
        grid_x=grid_x,
        grid_y=grid_y,
        landscape_vals=train_loss_grid,
        metric_type="loss",
        step_label=step_label,
        contour_mode=contour_mode,
        cmap_override=cmap,
        min_loss=min_loss,
        max_loss=max_loss,
        num_levels=num_levels,
        show_clabel=show_clabel,
    )

    # Panel 2: Weight Norm Landscape
    render_landscape_panel(
        ax=axes[1],
        grid_x=grid_x,
        grid_y=grid_y,
        landscape_vals=weight_norm_grid,
        metric_type="weight_norm",
        step_label=step_label,
        contour_mode=contour_mode,
        cmap_override=cmap,
        min_loss=min_loss,
        max_loss=max_loss,
        num_levels=num_levels,
        show_clabel=show_clabel,
    )

    fig.subplots_adjust(wspace=0.35)

    pdf_path = output_path or plots_filter_normalized_landscape_path(run_id, step_key, fmt="pdf")
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
        str, typer.Option("--run-id", "-r", help="MLflow run ID whose landscape should be plotted.")
    ],
    step: Annotated[
        int | None,
        typer.Option("--step", "-s", help="Checkpoint step to plot (default: final)."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Custom output destination path for the PDF."),
    ] = None,
    contour_mode: Annotated[
        str,
        typer.Option(
            "--contour-mode",
            "-m",
            help="Contour rendering style: 'lines' (paper style), 'filled', or 'both'.",
        ),
    ] = "lines",
    cmap: Annotated[
        str | None,
        typer.Option("--cmap", help="Custom matplotlib colormap name (e.g. viridis, plasma)."),
    ] = None,
    clabel: Annotated[
        bool,
        typer.Option(
            "--clabel/--no-clabel",
            help="Display inline numerical labels on contour lines.",
        ),
    ] = True,
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
    ] = 30,
    png: Annotated[
        bool,
        typer.Option("--png/--no-png", help="Generate accompanying 300 dpi PNG alongside PDF."),
    ] = True,
    dpi: Annotated[
        int,
        typer.Option("--dpi", help="Rasterization resolution for PNG output."),
    ] = 300,
) -> None:
    """CLI to plot filter-wise normalized 2D loss and weight norm landscape."""
    plot_filter_normalized_landscape(
        run_id=run_id,
        step=step,
        output_path=output,
        contour_mode=contour_mode,
        cmap=cmap,
        show_clabel=clabel,
        min_loss=min_loss,
        max_loss=max_loss,
        num_levels=num_levels,
        save_png=png,
        dpi=dpi,
    )


if __name__ == "__main__":
    app()
