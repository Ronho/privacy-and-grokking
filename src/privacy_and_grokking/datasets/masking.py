"""
Defines different masking strategies for dataset samples across multiple models.

(Ideal) Criteria:
- Each data point should appear in approximately p * num_models models.
- Each model should have approximately p * num_samples data points.
- Each class should be evenly represented for each model.
- There should be randomness in the selection to avoid models being too similar.
- The masking strategy should be deterministic given a seed.
- The implementation should be efficient in both time and space.
"""

from abc import ABC, abstractmethod
from enum import StrEnum

import torch
from pydantic import BaseModel, Field

from privacy_and_grokking.datasets import CanarySubset
from privacy_and_grokking.utils import Logger


class Masking(ABC):
    def __init__(
        self, num_samples: int, num_classes: int, num_models: int, p: float, seed: int | None = None
    ):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.num_models = num_models
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be between 0 and 1")
        self.p = p
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)
        else:
            Logger.get().warning("No seed provided for masking, using non-deterministic behavior.")

    def __call__(self, classes: torch.Tensor | None = None) -> torch.Tensor:
        logger = Logger.get()
        if classes is None:
            logger.warning("No classes provided for StratifiedMasking, using even distribution.")
            samples_per_class = self.num_samples // self.num_classes
            remainder = self.num_samples % self.num_classes
            classes = torch.repeat_interleave(torch.arange(self.num_classes), samples_per_class)
            if remainder > 0:
                classes = torch.cat([classes, torch.arange(end=remainder)])

        if len(classes) != self.num_samples:
            logger.error(
                "Length of classes does not match num_samples",
                {"classes": len(classes), "num_samples": self.num_samples},
            )
            raise ValueError("Length of classes must match num_samples")

        return self._generate_mask(classes)

    @abstractmethod
    def _generate_mask(self, classes: torch.Tensor) -> torch.Tensor: ...


class PartitionedStratifiedMasking(Masking):
    """
    The dataset is partitioned into num_models disjoint subsets, each containing approximately
    p * num_samples data points. Each subset is stratified to ensure class balance.

    Note: This masking can only satisfy p == 1/num_models. This makes sense for small number of models (e.g. 2).
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


class IndependentStratifiedMasking(Masking):
    """Each model independently selects p * num_samples data points, stratified by class."""

    def _generate_mask(self, classes: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros((self.num_samples, self.num_models), dtype=torch.bool)
        for c in range(self.num_classes):
            class_indices = (classes == c).nonzero().squeeze()
            n_to_pick = int(len(class_indices) * self.p)
            for model_idx in range(self.num_models):
                choosen = torch.randperm(len(class_indices), generator=self.rng)[:n_to_pick]
                mask[class_indices[choosen], model_idx] = True
        return mask


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
                m1, m2 = torch.randint(0, self.num_models, (2,), generator=self.rng).tolist()
                if m1 == m2:
                    continue

                col1 = mask[class_mask_rows, m1]
                col2 = mask[class_mask_rows, m2]

                only_1 = torch.nonzero(col1 & (~col2)).view(-1)
                only_2 = torch.nonzero(col2 & (~col1)).view(-1)

                if len(only_1) > 0 and len(only_2) > 0:
                    idx1 = only_1[torch.randint(len(only_1), (1,), generator=self.rng)]
                    idx2 = only_2[torch.randint(len(only_2), (1,), generator=self.rng)]

                    global_idx1 = class_mask_rows[idx1]
                    global_idx2 = class_mask_rows[idx2]

                    mask[global_idx1, m1] = False
                    mask[global_idx2, m1] = True

                    mask[global_idx2, m2] = False
                    mask[global_idx1, m2] = True

        return mask


class UniformMasking(Masking):
    """
    Each data point is assigned to each model independently with probability p.

    Note: There are basically no guarantees.
    """

    def _generate_mask(self, classes: torch.Tensor) -> torch.Tensor:
        mask = torch.rand((self.num_samples, self.num_models), generator=self.rng) < self.p
        return mask


class Maskings(StrEnum):
    UNIFORM = "uniform"
    INDEPENDENT_STRATIFIED = "independent_stratified"
    PARTITIONED_STRATIFIED = "partitioned_stratified"
    BALANCED_STRATIFIED = "balanced_stratified"


class MaskingConfig(BaseModel):
    name: Maskings
    num_models: int
    p: float = Field(ge=0, le=1)
    seed: int | None = None


def create_masking(config: MaskingConfig, num_samples: int, num_classes: int) -> Masking:
    input = {
        "num_samples": num_samples,
        "num_classes": num_classes,
        "num_models": config.num_models,
        "p": config.p,
        "seed": config.seed,
    }
    match config.name:
        case Maskings.UNIFORM:
            return UniformMasking(**input)
        case Maskings.INDEPENDENT_STRATIFIED:
            return IndependentStratifiedMasking(**input)
        case Maskings.PARTITIONED_STRATIFIED:
            return PartitionedStratifiedMasking(**input)
        case Maskings.BALANCED_STRATIFIED:
            return BalancedStratifiedMasking(**input)
        case _:
            raise ValueError(f"Unknown masking '{config.name}'.")


def mask_dataset(masking: Masking, ds: CanarySubset, mask_index: int) -> torch.utils.data.Dataset:
    mask = masking(classes=torch.Tensor([lbl for _, lbl in ds]))
    mask = torch.transpose(mask, 0, 1)[mask_index]
    subset = torch.utils.data.Subset(ds, torch.nonzero(mask).squeeze().tolist())
    return subset
