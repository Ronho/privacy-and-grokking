from typing import Literal

import torch

from privacy_and_grokking.datasets.masking.base import Masking, MaskingConfig
from privacy_and_grokking.utils import Logger


class PartitionedStratifiedMasking(Masking):
    """
    The dataset is partitioned into num_models disjoint subsets, each containing approximately
    p * num_samples data points. Each subset is stratified to ensure class balance.

    Note: This masking can only satisfy p == 1/num_models. This makes sense for small number
    of models (e.g. 2).
    """

    def _generate_mask(self, classes: torch.Tensor) -> torch.Tensor:
        if self.p != (1.0 / self.num_models):
            Logger.get().warning(
                "PartitionedStratifiedMasking cannot fulfill the condition p == 1/num_models",
                {"p": self.p, "num_models": self.num_models},
            )

        mask = torch.zeros((self.num_samples, self.num_models), dtype=torch.bool)
        for c in range(self.num_classes):
            class_indices = (classes == c).nonzero().squeeze()
            perm = torch.randperm(len(class_indices), generator=self.rng)
            shuffled_indices = class_indices[perm]
            split_indices = torch.chunk(shuffled_indices, self.num_models)
            for model_idx, indices in enumerate(split_indices):
                mask[indices, model_idx] = True

        return mask


class PartitionedStratifiedMaskingConfig(MaskingConfig):
    name: Literal["partitioned_stratified"] = "partitioned_stratified"

    def __call__(self, num_samples: int, num_classes: int) -> Masking:
        return PartitionedStratifiedMasking(
            num_samples=num_samples,
            num_classes=num_classes,
            num_models=self.num_models,
            p=self.p,
            seed=self.seed,
        )
