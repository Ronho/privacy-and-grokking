from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from privacy_and_grokking.visualize.mlflow_data import MIA_AUC_KEYS, RunData
from privacy_and_grokking.visualize.tsne import plot_tsne_classes_on_ax, plot_tsne_on_ax

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
    for key, label, color in [
        ("validation.train.loss", "Train", "#2563eb"),
        ("validation.test.loss", "Test", "#dc2626"),
    ]:
        pair = _get(rd, key)
        if pair is not None:
            steps, values = pair
            ax2.plot(steps, values, label=label, linewidth=2, color=color)
    ax2.set_xlabel("Training Step")
    ax2.set_xscale("log")
    ax2.set_ylabel("Loss")
    ax2.set_yscale("log")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3, which="both")
    ax2.set_title("Loss")

    fig.text(0.02, 0.01, f"Run: {rd.config.full_name}", fontsize=9, alpha=0.6)
    fig.tight_layout()
    return fig


def plot_superplot(
    runs: list[RunData],
    *,
    log_scale: bool = True,
) -> Figure:
    n_runs = len(runs)
    has_activations = any(rd.train_activations is not None for rd in runs)
    n_rows = 5 if has_activations else 4

    fig, axes = plt.subplots(
        n_rows,
        n_runs,
        figsize=(6 * n_runs, 3.5 * n_rows),
        squeeze=False,
    )

    for col, rd in enumerate(runs):
        # Accuracy
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
        ax.set_xscale("log")

        # Weight Norm
        ax = axes[1, col]
        wn_total = _get(rd, "weight_norm/total")
        if wn_total is not None:
            steps, values = wn_total
            ax.plot(steps, values, label="Total", linewidth=2, color="#7c3aed")

        # Find last-layer weight norm key dynamically
        ll_keys = [
            k for k in rd.metrics if k.startswith("weight_norm/") and k != "weight_norm/total"
        ]
        # Pick the key whose name contains the last_layer param name
        last_layer_key = None
        for k in ll_keys:
            if "fc3" in k or "fc2" in k:
                # For MLP the output layer is fc3, for CNN it's fc2
                last_layer_key = k
                break
        if last_layer_key is not None:
            pair = _get(rd, last_layer_key)
            if pair is not None:
                steps, values = pair
                ax.plot(steps, values, label="Last Layer", linewidth=2, color="#ec4899")

        if col == 0:
            label_txt = "Weight Norm (log)" if log_scale else "Weight Norm"
            ax.set_ylabel(label_txt, fontsize=11)
        if log_scale:
            ax.set_yscale("log")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3, which="both")
        ax.set_xscale("log")

        # Loss
        ax = axes[2, col]
        for key, label, color in [
            ("validation.train.loss", "Train", "#2563eb"),
            ("validation.test.loss", "Test", "#dc2626"),
        ]:
            pair = _get(rd, key)
            if pair is not None:
                steps, values = pair
                ax.plot(steps, values, label=label, linewidth=2, color=color)
        if col == 0:
            label_txt = "Loss (log)" if log_scale else "Loss"
            ax.set_ylabel(label_txt, fontsize=11)
        if log_scale:
            ax.set_yscale("log")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3, which="both")
        ax.set_xscale("log")

        # ROC AUC
        ax = axes[3, col]
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
        ax.set_xlabel("Training Step (log scale)", fontsize=11)
        if col == 0:
            ax.set_ylabel("ROC AUC", fontsize=11)
        ax.legend(loc="best", fontsize=7)
        ax.grid(True, alpha=0.3, which="both")
        ax.set_ylim(0, 1.05)
        ax.set_xscale("log")

        # t-SNE classes (optional)
        if has_activations:
            ax = axes[4, col]
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
                    title="t-SNE (Classes)",
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

    scale_label = "Log" if log_scale else "Linear"
    fig.suptitle(
        f"Multi-Run Comparison ({scale_label} Scale)",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig
