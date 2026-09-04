import torch
from pydantic import BaseModel, Field
from torch.utils.data import Dataset, Subset

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


class CanaryDataset:
    """A dataset wrapper that applies canary transforms to designated samples."""

    def __init__(
        self,
        dataset: Dataset,
        subset_indices: torch.Tensor,
        num_classes: int,
        canary_indices: torch.Tensor | None = None,
        canary_labels: torch.Tensor | None = None,
        canary_transform=None,
    ) -> None:
        self.dataset = dataset
        self.subset_indices = subset_indices
        self.num_classes = num_classes
        self.canary_indices = canary_indices if canary_indices is not None else torch.empty(0)
        self.canary_labels = canary_labels
        self.canary_transform = canary_transform

    def __len__(self) -> int:
        return len(self.subset_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx >= len(self.subset_indices):
            raise IndexError("Index out of range.")

        index = self.subset_indices[idx]
        img, lbl = self.dataset[index]

        if index in self.canary_indices:
            if self.canary_transform is None or self.canary_labels is None:
                raise RuntimeError("No canary transform/labels provided but canary index accessed.")
            img = self.canary_transform(img)
            lbl = self.canary_labels[
                int((self.canary_indices == index).nonzero(as_tuple=True)[0].item())
            ]

        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img)
        if not isinstance(lbl, torch.Tensor):
            lbl = torch.tensor(lbl, dtype=torch.long)

        return img, lbl


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
        labels = torch.tensor([lbl for _, lbl in dataset], dtype=torch.long)
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

    def _extract_labels(self, dataset: Dataset, num_samples: int) -> torch.Tensor:
        """Extract all labels from a dataset into a tensor."""
        return torch.tensor([dataset[i][1] for i in range(num_samples)], dtype=torch.long)

    def _compute_subset_indices(
        self,
        labels: torch.Tensor,
        num_samples: int,
        num_classes: int,
        rng: torch.Generator,
        *,
        canary_lookup: dict[int, torch.Tensor] | None = None,
        raw_lookup: dict[int, torch.Tensor] | None = None,
        num_canaries: int = 0,
        target_size: int | None = -1,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute class-balanced subset indices, respecting train_size and canary splits."""
        if target_size == -1:
            target_size = self.train_size

        if target_size is None:
            if canary_lookup is None or raw_lookup is None:
                return torch.arange(num_samples), None
            else:
                raw_parts = []
                canary_parts = []
                for cls in range(num_classes):
                    raw_parts.append(raw_lookup[cls])
                    canary_parts.append(canary_lookup[cls])
                return torch.cat(raw_parts), torch.cat(canary_parts)

        if target_size > num_samples:
            raise ValueError("target_size exceeds dataset size.")

        train_dist = distribute_a_across_b(target_size, num_classes)

        # Without canaries: simple class-balanced subsetting
        if canary_lookup is None or raw_lookup is None:
            parts: list[torch.Tensor] = []
            for cls, amt in enumerate(train_dist):
                class_indices = (labels == cls).nonzero().squeeze(-1)
                perm = torch.randperm(len(class_indices), generator=rng)
                parts.append(class_indices[perm[:amt]])
            return torch.cat(parts), None

        # With canaries: split each class budget between raw and canary samples
        train_num_canaries = min(num_canaries, self.train_size)
        train_canary_dist = distribute_a_across_b(train_num_canaries, num_classes)
        raw_parts = []
        canary_parts = []
        for cls, (amt_raw, amt_canary) in enumerate(
            zip(train_dist - train_canary_dist, train_canary_dist, strict=True)
        ):
            raw_parts.append(raw_lookup[cls][:amt_raw])
            canary_parts.append(canary_lookup[cls][:amt_canary])
        return torch.cat(raw_parts), torch.cat(canary_parts)

    def apply_canary(
        self, dataset: Dataset, num_classes: int, target_size: int | None = -1
    ) -> tuple[Dataset, Dataset | None]:
        """Wrap the dataset in a CanaryDataset, optionally injecting canary samples.

        When no canary config is set, this still handles train_size subsetting.
        When canaries are configured, designated samples get their images modified
        and labels reassigned via derangement.
        """
        if target_size == -1:
            target_size = self.train_size

        num_samples = len(dataset)  # type: ignore[arg-type]

        # No canary config at all
        if self.canary is None:
            if target_size is None:
                return dataset, None
            rng = self._make_rng()
            labels = self._extract_labels(dataset, num_samples)
            subset_indices, _ = self._compute_subset_indices(
                labels, num_samples, num_classes, rng, target_size=target_size
            )
            return Subset(dataset, subset_indices.tolist()), None

        # Canary config present but num is zero — nothing to inject
        num_canaries = self.canary.num
        if num_canaries == 0:
            if target_size is None:
                return dataset, None
            rng = self._make_rng()
            labels = self._extract_labels(dataset, num_samples)
            subset_indices, _ = self._compute_subset_indices(
                labels, num_samples, num_classes, rng, target_size=target_size
            )
            return Subset(dataset, subset_indices.tolist()), None

        if self.seed is None:
            raise ValueError("A seed is required for deterministic canary assignment.")

        rng = self._make_rng()
        labels = self._extract_labels(dataset, num_samples)

        canary_distribution = distribute_a_across_b(num_canaries, num_classes)

        # Partition each class into canary vs. raw indices
        canary_lookup: dict[int, torch.Tensor] = {}
        raw_lookup: dict[int, torch.Tensor] = {}
        for cls, amt in enumerate(canary_distribution):
            class_indices = (labels == cls).nonzero().squeeze(-1)
            perm = torch.randperm(len(class_indices), generator=rng)
            canary_lookup[cls] = class_indices[perm[:amt]]
            raw_lookup[cls] = class_indices[perm[amt:]]

        raw_indices, canary_indices = self._compute_subset_indices(
            labels,
            num_samples,
            num_classes,
            rng,
            canary_lookup=canary_lookup,
            raw_lookup=raw_lookup,
            num_canaries=num_canaries,
            target_size=target_size,
        )

        # Build canary transform and deranged labels
        all_canary_indices = torch.cat(list(canary_lookup.values()))
        input_shape = dataset[0][0].shape
        canary_transform = create_canary_generator(config=self.canary, dim=input_shape)
        canary_labels = random_derange_indices(
            canary_lookup={k: v.tolist() for k, v in canary_lookup.items()},
            seed=self.seed,
        )

        canary_dataset = CanaryDataset(
            dataset=dataset,
            subset_indices=canary_indices,
            num_classes=num_classes,
            canary_indices=all_canary_indices,
            canary_labels=canary_labels,
            canary_transform=canary_transform,
        )

        raw_dataset = Subset(dataset, raw_indices.tolist())

        return raw_dataset, canary_dataset

    def __call__(self) -> DataContainer:
        """Build the dataset, apply canaries and subsetting, then apply masking."""
        container = self.data()

        # Train split
        train_raw, train_canary = self.apply_canary(container.train, container.num_classes)
        train_raw = self.apply_mask(train_raw, container.num_classes)
        if train_canary is not None:
            train_canary = self.apply_mask(train_canary, container.num_classes)

        # Test split
        test_raw, test_canary = self.apply_canary(
            container.test, container.num_classes, target_size=None
        )

        return DataContainer(
            train=train_raw,
            test=test_raw,
            num_classes=container.num_classes,
            input_shape=container.input_shape,
            normalization=container.normalization,
            train_canary=train_canary,
            test_canary=test_canary,
        )
