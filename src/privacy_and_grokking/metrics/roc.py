import numpy as np
import torch
from sklearn.metrics import auc, roc_curve

FPR_RATES: list[float] = [0.01, 0.05, 0.10]


def compute_roc_metrics_single_step(
    train_signals: torch.Tensor,
    test_signals: torch.Tensor,
    fpr_rates: list[float] | None = None,
) -> dict[str, float]:
    """Compute AUC and TPR@FPR metrics for a single attack signal."""
    if fpr_rates is None:
        fpr_rates = FPR_RATES

    y_true = torch.cat(
        [
            torch.ones(len(train_signals)),
            torch.zeros(len(test_signals)),
        ],
    )
    y_scores = torch.cat([train_signals, test_signals])

    fpr, tpr, _ = roc_curve(
        y_true.cpu().numpy(),
        y_scores.cpu().numpy(),
    )
    roc_auc = auc(fpr, tpr)

    metrics: dict[str, float] = {"auc": float(roc_auc)}
    for rate in fpr_rates:
        tpr_at_rate = float(np.interp(rate, fpr, tpr))
        pct = int(rate * 100)
        metrics[f"tpr-at-fpr/{pct}"] = tpr_at_rate

    return metrics
