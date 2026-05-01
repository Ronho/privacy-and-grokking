from typing import Literal

import torch

from privacy_and_grokking.loss.regularizer.base import (
    RegularizerType,
    SelfContainedTwoSampleRegularizerConfig,
)
from privacy_and_grokking.metrics.distribution_overlap import soft_distribution_overlap_kde


class OverlapKDERegularizerConfig(SelfContainedTwoSampleRegularizerConfig):
    """KDE-based soft overlap regularizer

    Uses Gaussian KDE with Silverman bandwidth for the most accurate overlap
    estimate, especially under size imbalance.
    """

    name: Literal["overlap_kde"] = "overlap_kde"
    n_points: int = 200

    def __call__(self) -> RegularizerType:
        validation_set_generator = self.source()

        def regularizer(train_losses: torch.Tensor) -> torch.Tensor:
            val_losses = validation_set_generator(train_losses)
            overlap = soft_distribution_overlap_kde(
                train_losses, val_losses, n_points=self.n_points
            )
            return 1.0 - overlap

        return regularizer
