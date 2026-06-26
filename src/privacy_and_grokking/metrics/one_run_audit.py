import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch


def compute_empirical_epsilon(
    in_losses: torch.Tensor, out_losses: torch.Tensor, step: int
) -> dict[str, float]:
    """
    Computes empirical epsilon privacy bound from one training run based on Steinke et al.
    We assume lower loss indicates the sample was IN the training set.
    """
    in_l = in_losses.detach().cpu().numpy()
    out_l = out_losses.detach().cpu().numpy()

    n_in = len(in_l)
    n_out = len(out_l)

    if n_in == 0 or n_out == 0:
        return {}

    all_losses = np.concatenate([in_l, out_l])
    thresholds = np.unique(all_losses)

    max_eps = 0.0
    best_tpr = 0.0
    best_fpr = 0.0
    best_threshold = 0.0

    for t in thresholds:
        tpr = np.sum(in_l <= t) / n_in
        fpr = np.sum(out_l <= t) / n_out

        # Smooth FPR to avoid division by zero
        fpr_smoothed = max(fpr, 1.0 / n_out)
        tpr_smoothed = max(tpr, 1.0 / n_in)

        if tpr_smoothed > fpr_smoothed:
            eps = np.log(tpr_smoothed) - np.log(fpr_smoothed)
            if eps > max_eps:
                max_eps = eps
                best_tpr = tpr
                best_fpr = fpr
                best_threshold = t

    _plot_and_log_audit(in_l, out_l, step)

    return {
        "empirical_epsilon": float(max_eps),
        "best_tpr": float(best_tpr),
        "best_fpr": float(best_fpr),
        "best_threshold": float(best_threshold),
    }


def _plot_and_log_audit(in_l: np.ndarray, out_l: np.ndarray, step: int):
    fig, ax = plt.subplots(figsize=(8, 6))

    bins = np.linspace(min(in_l.min(), out_l.min()), max(in_l.max(), out_l.max()), 50)
    ax.hist(in_l, bins=bins, alpha=0.5, density=True, label="IN Canaries")
    ax.hist(out_l, bins=bins, alpha=0.5, density=True, label="OUT Canaries")

    ax.set_title(f"Canary Loss Distribution at Step {step}")
    ax.set_xlabel("Loss")
    ax.set_ylabel("Density")
    ax.legend()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / f"canary_audit_{step}.png"
        fig.savefig(path)
        plt.close(fig)
        if mlflow.active_run() is not None:
            mlflow.log_artifact(str(path), artifact_path="audit_plots")
