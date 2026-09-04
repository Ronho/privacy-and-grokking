from typing import Literal

import torch

from privacy_and_grokking.datasets.masking.base import Masking, MaskingConfig
from privacy_and_grokking.utils import Logger


class PairedStratifiedMasking(Masking):
    """
    Rounds up the number of models to an even number to form pairs.
    For each pair, the global dataset is split 50/50 exactly (stratified by class).
    Twin 1 gets the first half, Twin 2 gets the exact inverse.
    """

    def __init__(
        self,
        num_samples: int,
        num_classes: int,
        num_models: int,
        p: float,
        seed: int,
    ):
        # We enforce p=0.5 for this strategy
        if p != 0.5:
            Logger.get().warning("PairedStratifiedMasking overrides p to 0.5.")

        # Round up to nearest even number if necessary
        effective_num_models = num_models if num_models % 2 == 0 else num_models + 1
        if effective_num_models != num_models:
            Logger.get().info(
                f"PairedStratifiedMasking rounded num_models up from {num_models} to {effective_num_models} to form pairs."
            )

        super().__init__(
            num_samples=num_samples,
            num_classes=num_classes,
            num_models=effective_num_models,
            p=0.5,
            seed=seed,
        )

    def _generate_mask(self, classes: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros((self.num_samples, self.num_models), dtype=torch.bool)
        num_pairs = self.num_models // 2

        for pair_idx in range(num_pairs):
            model_1_idx = 2 * pair_idx
            model_2_idx = 2 * pair_idx + 1

            for c in range(self.num_classes):
                class_indices = (classes == c).nonzero().view(-1)
                n_class_total = len(class_indices)

                # We need exactly half of the class samples for Twin 1
                n_per_class = n_class_total // 2

                perm = torch.randperm(n_class_total, generator=self.rng)
                shuffled_indices = class_indices[perm]

                twin1_selected = shuffled_indices[:n_per_class]
                twin2_selected = shuffled_indices[n_per_class:]

                mask[twin1_selected, model_1_idx] = True
                mask[twin2_selected, model_2_idx] = True

        return mask


class PairedStratifiedMaskingConfig(MaskingConfig):
    name: Literal["paired_stratified"] = "paired_stratified"

    def __call__(self, num_samples: int, num_classes: int) -> Masking:
        return PairedStratifiedMasking(
            num_samples=num_samples,
            num_classes=num_classes,
            num_models=self.num_models,
            p=0.5,  # Always 0.5
            seed=self.seed,
        )
