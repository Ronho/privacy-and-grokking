import torch
from pydantic import BaseModel, Field
from torch.utils.data import Dataset, Subset, TensorDataset

from privacy_and_grokking.datasets.canaries import CanaryType, create_canary_generator
from privacy_and_grokking.datasets.canary_class_assignment import random_derange_indices
from privacy_and_grokking.datasets.masking import Mask
from privacy_and_grokking.datasets.sets import Data
from privacy_and_grokking.datasets.sets.base import DataContainer, Normalization

__all__ = ["CanaryDataset", "DataContainer", "DatasetConfig", "Normalization"]


def distribute_a_across_b(a: int, b: int) -> torch.Tensor:
    base = a // b
    remainder = a % b
    distribution = torch.zeros(b, dtype=torch.int)
    distribution.fill_(base)
    distribution[:remainder] += 1
    return distribution

def derange_balanced_classes(l: torch.Tensor, num_classes: int, rng: torch.Generator) -> torch.Tensor:
    """
    Assuming 50 elements per class and 9 classes, this function does the following:
    Each class has 5 (50 // 9) times every other class. For the remaining 5 (50 - 45) elements, they are randomly assigned.
    This ensures the most chaos.

    Requires equal amount of elements per class.
    """
    n = len(l)
    k = l.shape[0] // num_classes
    other_classes = num_classes - 1
    full_cycles = k // other_classes
    remainder = k % other_classes

    class_indices = torch.stack([
        torch.where(l == c)[0][torch.randperm(k, generator=rng)] 
        for c in range(num_classes)
    ])

    shifts_list = [torch.randperm(other_classes, generator=rng) + 1 for _ in range(full_cycles)]
    if remainder > 0:
        shifts_list.append((torch.randperm(other_classes, generator=rng) + 1)[:remainder])
    all_shifts = torch.cat(shifts_list)
    all_shifts = all_shifts[torch.randperm(k, generator=rng)]
    
    result = torch.empty_like(l)
    for slot in range(k):
        shift = all_shifts[slot].item()
        for c in range(num_classes):
            target_class = (c + shift) % num_classes
            idx = class_indices[c, slot]
            result[idx] = target_class

    return result


class CanaryDataset(Dataset):
    """Dataset wrapper for precomputed canary samples."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor | list[int]):
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int | torch.Tensor]:
        return self.images[idx], self.labels[idx]


class DatasetConfig(BaseModel):
    data: Data
    mask: Mask | None = None
    canary: CanaryType | None = None
    train_size: int | None = Field(ge=0, default=None)
    seed: int | None = None

    def apply_mask(self, dataset: Dataset, num_classes: int) -> Subset:
        """Apply the masking strategy to select a subset of samples for this model."""
        if self.mask is None:
            return dataset

        num_samples = len(dataset)  # type: ignore[arg-type]
        masking = self.mask(num_samples=num_samples, num_classes=num_classes)
        labels = torch.tensor([int(lbl) for _, lbl in dataset], dtype=torch.long)
        mask = masking(classes=labels)
        model_mask = mask[:, self.mask.model_index]
        subset_indices = torch.nonzero(model_mask).squeeze(-1).tolist()
        return Subset(dataset, subset_indices)  # type: ignore[arg-type]

    def _make_rng(self) -> torch.Generator:
        """Create a seeded random generator."""
        rng = torch.Generator()
        if self.seed:
            rng.manual_seed(self.seed)
        return rng

    def __call__(self) -> DataContainer:
        """Build the dataset, apply canaries and subsetting, then apply masking."""
        container = self.data()

        train_set = container.train
        if self.train_size:
            if (len(train_set) < self.train_size):
                raise Exception(f"Fewer samples than explected train size available. {len(train_set)} < {self.train_size}")
            rng = self._make_rng()
            labels = torch.Tensor([y for _, y in train_set])
            train_distribution = distribute_a_across_b(self.train_size, container.num_classes)
            indices = []
            for cls, amt in enumerate(train_distribution):
                cls_indices = (labels == cls).nonzero().squeeze(-1)
                perm = torch.randperm(len(cls_indices), generator=rng)
                indices.extend(cls_indices[perm[:amt]].tolist())
            train_set = Subset(train_set, indices)

        canary_train_set = None
        canary_test_set = None
        if self.canary is not None:
            rng = self._make_rng()
            labels = torch.Tensor([y for _, y in train_set])
            if not self.canary.num % container.num_classes == 0:
                # NOTE: We do this to ensure that we can redistribute the labels properly below.
                raise ValueError("Number of canaries must be divisible by number of classes.")
            canary_distribution = distribute_a_across_b(self.canary.num, container.num_classes)

            canary_indices = []
            raw_indices = []
            for cls, amt in enumerate(canary_distribution):
                cls_indices = (labels == cls).nonzero().squeeze(-1)
                perm = torch.randperm(len(cls_indices), generator=rng)
                canary_indices.extend(cls_indices[perm[:amt]].tolist())
                raw_indices.extend(cls_indices[perm[amt:]].tolist())
            canary_train_set = Subset(train_set, canary_indices)
            train_set = Subset(train_set, raw_indices)

            canary_transform = create_canary_generator(config=self.canary, dim=container.input_shape)
            canary_train_x = torch.stack([canary_transform(x) for x, _ in canary_train_set])
            canary_train_labels = derange_balanced_classes(torch.tensor([y for _, y in canary_train_set], dtype=torch.long), container.num_classes, rng)
            is_int_train_label = len(train_set) > 0 and isinstance(train_set[0][1], int)
            canary_train_set = CanaryDataset(
                canary_train_x,
                canary_train_labels.tolist() if is_int_train_label else canary_train_labels,
            )

            labels = torch.Tensor([y for _, y in container.test])
            canary_distribution = distribute_a_across_b(self.canary.num, container.num_classes)
            canary_indices = []
            for cls, amt in enumerate(canary_distribution):
                cls_indices = (labels == cls).nonzero().squeeze(-1)
                perm = torch.randperm(len(cls_indices), generator=rng)
                canary_indices.extend(cls_indices[perm[:amt]].tolist())
            canary_test_set = Subset(container.test, canary_indices)

            canary_test_x = torch.stack([canary_transform(x) for x, _ in canary_test_set])
            canary_test_labels = derange_balanced_classes(torch.tensor([y for _, y in canary_test_set], dtype=torch.long), container.num_classes, rng)
            is_int_test_label = len(container.test) > 0 and isinstance(container.test[0][1], int)
            canary_test_set = CanaryDataset(
                canary_test_x,
                canary_test_labels.tolist() if is_int_test_label else canary_test_labels,
            )

        # Apply Mask
        if self.mask:
            train_set = self.apply_mask(train_set, container.num_classes)
            if canary_train_set:
                canary_train_set = self.apply_mask(canary_train_set, container.num_classes)

        return DataContainer(
            train=train_set,
            test=container.test,
            num_classes=container.num_classes,
            input_shape=container.input_shape,
            normalization=container.normalization,
            train_canary=canary_train_set,
            test_canary=canary_test_set,
        )
