from __future__ import annotations

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from privacy_and_grokking.visualize.mlflow_data import MIA_AUC_KEYS, RunData
from privacy_and_grokking.visualize.tsne import plot_tsne_classes_on_ax

_NICE_NAMES: dict[str, str] = {
    "mia_prob/auc": "Prob",
    "mia_logit/auc": "Logit",
    "mia_ce_loss/auc": "CE Loss",
    "mia_mse_loss/auc": "MSE Loss",
    "mia_correctness/auc": "Correct",
    "mia_merlin_morgan_ce/auc": "MM CE",
    "mia_merlin_morgan_mse/auc": "MM MSE",
}


def _get(rd: RunData, key: str) -> tuple[np.ndarray, np.ndarray] | None:
    hist = rd.metrics.get(key)
    if hist is None:
        return None
    return hist.as_arrays()


def _label(rd: RunData) -> str:
    return rd.run_name


_OVERLAP_COLOR = "#f59e0b"

# Colour cycle for per-layer norm lines (total is always first)
_LAYER_COLORS = [
    "#7c3aed",
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#d97706",
    "#db2777",
    "#0891b2",
    "#65a30d",
]


def _plot_norm_panel(
    ax: matplotlib.axes.Axes,
    rd: RunData,
    *,
    prefix: str,
    log_scale: bool = True,
    x_log_scale: bool = True,
) -> None:
    """Plot per-parameter and total norm lines for *prefix*
    (e.g. ``'weight_norm/'`` or ``'grad_norm/'``).
    """
    total_key = f"{prefix}total"
    layer_keys = sorted(k for k in rd.metrics if k.startswith(prefix) and k != total_key)

    # Total – thick, first colour
    pair = _get(rd, total_key)
    if pair is not None:
        steps, values = pair
        ax.plot(steps, values, label="Total", linewidth=2, color=_LAYER_COLORS[0])

    # Per-layer – thin, rotated colours
    for i, key in enumerate(layer_keys):
        layer_name = key[len(prefix) :]  # strip prefix for legend label
        pair = _get(rd, key)
        if pair is not None:
            steps, values = pair
            color = _LAYER_COLORS[(i + 1) % len(_LAYER_COLORS)]
            ax.plot(steps, values, label=layer_name, linewidth=1, alpha=0.7, color=color)

    if log_scale:
        ax.set_yscale("log")
    if x_log_scale:
        ax.set_xscale("log")
    ax.legend(loc="best", fontsize=6)
    ax.grid(True, alpha=0.3, which="both")


def _plot_loss_panel(
    ax: matplotlib.axes.Axes,
    rd: RunData,
    *,
    log_scale: bool = True,
    x_log_scale: bool = True,
    show_ylabel: bool = True,
) -> None:
    """Plot train/test loss with ±1 std shading and distribution overlap on a twin axis.

    Falls back to ``validation.{split}.loss`` when extraction metrics are unavailable.
    """
    splits = [
        ("train", "#2563eb"),
        ("test", "#dc2626"),
    ]
    for split, color in splits:
        mean_key = f"extraction.{split}.loss.mean"
        std_key = f"extraction.{split}.loss.std"
        fallback_key = f"validation.{split}.loss"

        mean_pair = _get(rd, mean_key)
        std_pair = _get(rd, std_key)
        fallback_pair = _get(rd, fallback_key)

        if mean_pair is not None:
            steps, mean_vals = mean_pair
            ax.plot(steps, mean_vals, label=split.capitalize(), linewidth=2, color=color)
            if std_pair is not None:
                _, std_vals = std_pair
                lower = np.maximum(mean_vals - std_vals, 1e-9)
                upper = mean_vals + std_vals
                ax.fill_between(steps, lower, upper, alpha=0.18, color=color)
        elif fallback_pair is not None:
            steps, values = fallback_pair
            ax.plot(steps, values, label=split.capitalize(), linewidth=2, color=color)

    if log_scale:
        ax.set_yscale("log")
    if x_log_scale:
        ax.set_xscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=8)

    # Overlap on secondary y-axis
    overlap_pair = _get(rd, "extraction.loss.overlap")
    if overlap_pair is not None:
        ax2 = ax.twinx()
        ov_steps, ov_vals = overlap_pair
        ax2.plot(
            ov_steps,
            ov_vals,
            color=_OVERLAP_COLOR,
            linewidth=1.5,
            linestyle="--",
            label="Distribution Overlap",
            alpha=0.85,
        )
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_ylabel("Loss Distribution Overlap", color=_OVERLAP_COLOR, fontsize=9)
        ax2.tick_params(axis="y", labelcolor=_OVERLAP_COLOR)
        ax2.legend(loc="upper left", fontsize=7)


def plot_per_run_roc_auc(rd: RunData) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    for key in MIA_AUC_KEYS:
        pair = _get(rd, key)
        if pair is None:
            continue
        steps, values = pair
        ax.plot(steps, values, label=_NICE_NAMES.get(key, key), linewidth=1.5, alpha=0.85)

    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=2, alpha=0.5, label="Random")
    ax.set_xlabel("Training Step (log scale)")
    ax.set_ylabel("ROC AUC")
    ax.set_title("MIA ROC AUC Evolution")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    fig.text(0.02, 0.01, f"Run: {rd.config.full_name}", fontsize=9, alpha=0.6)
    fig.tight_layout()
    return fig


def plot_per_run_training(rd: RunData) -> Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    for key, label, color in [
        ("validation.train.accuracy", "Train", "#2563eb"),
        ("validation.test.accuracy", "Test", "#dc2626"),
    ]:
        pair = _get(rd, key)
        if pair is not None:
            steps, values = pair
            ax1.plot(steps, values, label=label, linewidth=2, color=color)
    ax1.set_xlabel("Training Step")
    ax1.set_xscale("log")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3, which="both")
    ax1.set_title("Accuracy")

    # Loss
    _plot_loss_panel(ax2, rd, log_scale=True, x_log_scale=True)
    ax2.set_xlabel("Training Step")
    ax2.set_xscale("log")
    ax2.set_ylabel("Loss")
    ax2.set_title("Loss")

    fig.text(0.02, 0.01, f"Run: {rd.config.full_name}", fontsize=9, alpha=0.6)
    fig.tight_layout()
    return fig


def plot_superplot(
    runs: list[RunData],
    *,
    log_scale: bool = True,
    x_log_scale: bool = True,
) -> Figure:
    # Row layout: Accuracy | Weight Norm | Gradient Norm | Loss | ROC AUC | t-SNE(opt)
    n_runs = len(runs)
    has_activations = any(rd.train_activations is not None for rd in runs)
    n_rows = 6 if has_activations else 5

    fig, axes = plt.subplots(
        n_rows,
        n_runs,
        figsize=(6 * n_runs, 3.5 * n_rows),
        squeeze=False,
    )

    x_scale = "log" if x_log_scale else "linear"

    for col, rd in enumerate(runs):
        # ── Row 0: Accuracy ──────────────────────────────────────────────────
        ax = axes[0, col]
        for key, label, color in [
            ("validation.train.accuracy", "Train", "#2563eb"),
            ("validation.test.accuracy", "Test", "#dc2626"),
        ]:
            pair = _get(rd, key)
            if pair is not None:
                steps, values = pair
                ax.plot(steps, values, label=label, linewidth=2, color=color)
        if col == 0:
            ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_title(_label(rd), fontsize=10, fontweight="bold")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3, which="both")
        ax.set_ylim(0, 1.05)
        ax.set_xscale(x_scale)

        # ── Row 1: Weight Norm (all layers) ──────────────────────────────────
        ax = axes[1, col]
        _plot_norm_panel(
            ax, rd, prefix="weight_norm/", log_scale=log_scale, x_log_scale=x_log_scale
        )
        if col == 0:
            label_txt = "Weight Norm (log)" if log_scale else "Weight Norm"
            ax.set_ylabel(label_txt, fontsize=11)

        # ── Row 2: Gradient Norm (all layers) ────────────────────────────────
        ax = axes[2, col]
        _plot_norm_panel(ax, rd, prefix="grad_norm/", log_scale=log_scale, x_log_scale=x_log_scale)
        if col == 0:
            label_txt = "Grad Norm (log)" if log_scale else "Grad Norm"
            ax.set_ylabel(label_txt, fontsize=11)

        # ── Row 3: Loss ───────────────────────────────────────────────────────
        ax = axes[3, col]
        _plot_loss_panel(ax, rd, log_scale=log_scale, x_log_scale=x_log_scale)
        if col == 0:
            label_txt = "Loss (log)" if log_scale else "Loss"
            ax.set_ylabel(label_txt, fontsize=11)

        # ── Row 4: ROC AUC ────────────────────────────────────────────────────
        ax = axes[4, col]
        for key in MIA_AUC_KEYS:
            pair = _get(rd, key)
            if pair is None:
                continue
            steps, values = pair
            ax.plot(
                steps,
                values,
                label=_NICE_NAMES.get(key, key),
                linewidth=1.5,
                alpha=0.85,
            )
        ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=2, alpha=0.5, label="Random")
        ax.set_xlabel(f"Training Step ({x_scale} scale)", fontsize=11)
        if col == 0:
            ax.set_ylabel("ROC AUC", fontsize=11)
        ax.legend(loc="best", fontsize=7)
        ax.grid(True, alpha=0.3, which="both")
        ax.set_ylim(0, 1.05)
        ax.set_xscale(x_scale)

        # ── Row 5: t-SNE (optional) ───────────────────────────────────────────
        if has_activations:
            ax = axes[5, col]
            if (
                rd.train_activations is not None
                and rd.test_activations is not None
                and rd.train_labels is not None
                and rd.test_labels is not None
            ):
                plot_tsne_classes_on_ax(
                    ax,
                    rd.train_activations,
                    rd.test_activations,
                    rd.train_labels,
                    rd.test_labels,
                    title="",
                )
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No activations",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=11,
                    alpha=0.4,
                )
                ax.set_axis_off()

    y_label = "Log" if log_scale else "Linear"
    x_label = "Log" if x_log_scale else "Linear"
    fig.suptitle(
        f"Multi-Run Comparison (y: {y_label} | x: {x_label})",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig
