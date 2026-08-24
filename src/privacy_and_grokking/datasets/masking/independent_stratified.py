from typing import Literal

import torch

from privacy_and_grokking.datasets.masking.base import Masking, MaskingConfig


class IndependentStratifiedMasking(Masking):
    """Each model independently selects p * num_samples data points, stratified by class."""

    def _generate_mask(self, classes: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros((self.num_samples, self.num_models), dtype=torch.bool)
        for c in range(self.num_classes):
            class_indices = (classes == c).nonzero().squeeze(-1)
            n_to_pick = int(len(class_indices) * self.p)
            for model_idx in range(self.num_models):
                choosen = torch.randperm(len(class_indices), generator=self.rng)[:n_to_pick]
                mask[class_indices[choosen], model_idx] = True
        return mask


class IndependentStratifiedMaskingConfig(MaskingConfig):
    name: Literal["independent_stratified"] = "independent_stratified"

    def __call__(self, num_samples: int, num_classes: int) -> Masking:
        return IndependentStratifiedMasking(
            num_samples=num_samples,
            num_classes=num_classes,
            num_models=self.num_models,
            p=self.p,
            seed=self.seed,
        )
