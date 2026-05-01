from typing import Literal

import torch

from privacy_and_grokking.loss.regularizer.base import (
    RegularizerType,
    SelfContainedTwoSampleRegularizerConfig,
)


def _gaussian_kernel(x: torch.Tensor, y: torch.Tensor, bandwidth: float) -> torch.Tensor:
    """Gaussian RBF kernel between all pairs of rows in *x* and *y*."""
    # x: (N, D), y: (M, D) -> (N, M)
    dists = torch.cdist(x, y, p=2).pow(2)
    return torch.exp(-dists / (2.0 * bandwidth**2))


class MMDRegularizerConfig(SelfContainedTwoSampleRegularizerConfig):
    """Maximum Mean Discrepancy regularizer.

    MMD² = E[k(x,x')] + E[k(y,y')] − 2·E[k(x,y)]

    Uses a Gaussian RBF kernel between train and validation loss distributions.
    """

    name: Literal["mmd"] = "mmd"
    bandwidth: float = 0.1

    def __call__(self) -> RegularizerType:
        validation_set_generator = self.source()

        def regularizer(train_losses: torch.Tensor) -> torch.Tensor:
            val_losses = validation_set_generator(train_losses)
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

        return regularizer
