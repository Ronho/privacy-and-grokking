"""Regularizers that penalise distributional gaps between train and validation
confidence scores, making membership inference attacks harder."""

import torch
import torch.nn as nn

from privacy_and_grokking.metrics.distribution_overlap import soft_distribution_overlap


class OverlapRegularizer(nn.Module):
    """1 − soft_histogram_overlap(train_losses, val_losses).

    Minimising this pushes the train loss distribution toward the
    validation loss distribution.
    """

    def __init__(self, n_bins: int = 50, sigma: float = 0.05) -> None:
        super().__init__()
        self.n_bins = n_bins
        self.sigma = sigma

    def forward(self, train_losses: torch.Tensor, val_losses: torch.Tensor) -> torch.Tensor:
        overlap = soft_distribution_overlap(
            train_losses, val_losses.detach(), n_bins=self.n_bins, sigma=self.sigma
        )
        return 1.0 - overlap


def _gaussian_kernel(x: torch.Tensor, y: torch.Tensor, bandwidth: float) -> torch.Tensor:
    """Gaussian RBF kernel between all pairs of rows in *x* and *y*."""
    # x: (N, D), y: (M, D) -> (N, M)
    dists = torch.cdist(x, y, p=2).pow(2)
    return torch.exp(-dists / (2.0 * bandwidth**2))


class MMDRegularizer(nn.Module):
    """Maximum Mean Discrepancy between train and validation loss
    distributions using a Gaussian RBF kernel.

    MMD² = E[k(x,x')] + E[k(y,y')] − 2·E[k(x,y)]
    """

    def __init__(self, bandwidth: float = 0.1) -> None:
        super().__init__()
        self.bandwidth = bandwidth

    def forward(self, train_losses: torch.Tensor, val_losses: torch.Tensor) -> torch.Tensor:
        val_losses = val_losses.detach()
        x = train_losses.reshape(-1, 1)
        y = val_losses.reshape(-1, 1)

        k_xx = _gaussian_kernel(x, x, self.bandwidth)
        k_yy = _gaussian_kernel(y, y, self.bandwidth)
        k_xy = _gaussian_kernel(x, y, self.bandwidth)

        # Unbiased estimate (exclude diagonal for k_xx and k_yy)
        n, m = x.size(0), y.size(0)
        mmd = (
            (k_xx.sum() - k_xx.diagonal().sum()) / max(n * (n - 1), 1)
            + (k_yy.sum() - k_yy.diagonal().sum()) / max(m * (m - 1), 1)
            - 2.0 * k_xy.mean()
        )
        return mmd.clamp(min=0.0)
