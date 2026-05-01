from typing import Literal

import torch

from privacy_and_grokking.loss.regularizer.base import (
    RegularizerType,
    SelfContainedTwoSampleRegularizerConfig,
)
from privacy_and_grokking.metrics.distribution_overlap import soft_distribution_overlap_adaptive


class OverlapAdaptiveRegularizerConfig(SelfContainedTwoSampleRegularizerConfig):
    """Adaptive soft histogram-intersection overlap regularizer

    Like :class:`OverlapRegularizerConfig` but scales the bin count with the
    smaller sample, reducing sparse-bin bias under size imbalance.
    """

    name: Literal["overlap_adaptive"] = "overlap_adaptive"
    max_bins: int = 100
    sigma: float = 0.05

    def _make_regularizer(self) -> RegularizerType:
        validation_set_generator = self.source()

        def regularizer(train_losses: torch.Tensor) -> torch.Tensor:
            val_losses = validation_set_generator(train_losses)
            overlap = soft_distribution_overlap_adaptive(
                train_losses, val_losses, max_bins=self.max_bins, sigma=self.sigma
            )
            return 1.0 - overlap

        return regularizer
