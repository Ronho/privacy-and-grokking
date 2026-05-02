from typing import Literal

import torch

from privacy_and_grokking.datasets.masking.base import Masking, MaskingConfig


class UniformMasking(Masking):
    """
    Each data point is assigned to each model independently with probability p.

    Note: There are basically no guarantees.
    """

    def _generate_mask(self, classes: torch.Tensor) -> torch.Tensor:
        mask = torch.rand((self.num_samples, self.num_models), generator=self.rng) < self.p
        return mask


class UniformMaskingConfig(MaskingConfig):
    name: Literal["uniform"] = "uniform"

    def __call__(self, num_samples: int, num_classes: int) -> Masking:
        return UniformMasking(
            num_samples=num_samples,
            num_classes=num_classes,
            num_models=self.num_models,
            p=self.p,
            seed=self.seed,
        )
