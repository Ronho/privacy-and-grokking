from typing import Literal

import torch

from privacy_and_grokking.datasets.masking.base import Masking, MaskingConfig
from privacy_and_grokking.utils import Logger


class BalancedStratifiedMasking(Masking):
    """
    Each model gets exactly p * num_samples data points, stratified by class.
    Additionally, each data point appears in p * num_models models.

    However, the randomness is limited to ensure these constraints are met.
    """

    def _generate_mask(self, classes: torch.Tensor) -> torch.Tensor:
        n_per_model = int(self.num_samples * self.p)
        n_per_class = n_per_model // self.num_classes

        if n_per_model % self.num_classes != 0:
            Logger.get().warning("Samples per model not divisible by num_classes.")

        mask = torch.zeros((self.num_samples, self.num_models), dtype=torch.bool)
        for c in range(self.num_classes):
            class_indices = (classes == c).nonzero().view(-1)
            n_class_total = len(class_indices)

            perm = torch.randperm(n_class_total, generator=self.rng)
            shuffled_indices = class_indices[perm]

            ptr = 0
            for model_idx in range(self.num_models):
                if ptr + n_per_class <= n_class_total:
                    selected = shuffled_indices[ptr : ptr + n_per_class]
                    ptr += n_per_class
                else:
                    end_part = shuffled_indices[ptr:]
                    rem = n_per_class - len(end_part)
                    start_part = shuffled_indices[:rem]
                    selected = torch.cat([end_part, start_part])
                    ptr = rem

                if ptr == n_class_total:
                    ptr = 0
                mask[selected, model_idx] = True

        for c in range(self.num_classes):
            class_mask_rows = (classes == c).nonzero().view(-1)
            n_swaps = 5 * self.num_models

            for _ in range(n_swaps):
                m1, m2 = torch.randint(
                    0, self.num_models, (2,), generator=self.rng
                ).tolist()
                if m1 == m2:
                    continue

                col1 = mask[class_mask_rows, m1]
                col2 = mask[class_mask_rows, m2]

                only_1 = torch.nonzero(col1 & (~col2)).view(-1)
                only_2 = torch.nonzero(col2 & (~col1)).view(-1)

                if len(only_1) > 0 and len(only_2) > 0:
                    idx1 = only_1[
                        torch.randint(len(only_1), (1,), generator=self.rng)
                    ]
                    idx2 = only_2[
                        torch.randint(len(only_2), (1,), generator=self.rng)
                    ]

                    global_idx1 = class_mask_rows[idx1]
                    global_idx2 = class_mask_rows[idx2]

                    mask[global_idx1, m1] = False
                    mask[global_idx2, m1] = True

                    mask[global_idx2, m2] = False
                    mask[global_idx1, m2] = True

        return mask


class BalancedStratifiedMaskingConfig(MaskingConfig):
    name: Literal["balanced_stratified"] = "balanced_stratified"

    def __call__(self, num_samples: int, num_classes: int) -> Masking:
        return BalancedStratifiedMasking(
            num_samples=num_samples,
            num_classes=num_classes,
            num_models=self.num_models,
            p=self.p,
            seed=self.seed,
        )
