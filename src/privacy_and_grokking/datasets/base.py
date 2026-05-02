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

    def apply_mask(
        self, dataset: "Dataset | CanaryDataset", num_classes: int
    ) -> "Dataset | CanaryDataset | Subset":
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
        share: float = 0.0,
    ) -> torch.Tensor:
        """Compute class-balanced subset indices, respecting train_size and canary splits."""
        if self.train_size is None:
            return torch.arange(num_samples)

        if self.train_size > num_samples:
            raise ValueError("train_size exceeds dataset size.")

        train_dist = distribute_a_across_b(self.train_size, num_classes)

        # Without canaries: simple class-balanced subsetting
        if canary_lookup is None or raw_lookup is None:
            parts: list[torch.Tensor] = []
            for cls, amt in enumerate(train_dist):
                class_indices = (labels == cls).nonzero().squeeze(-1)
                perm = torch.randperm(len(class_indices), generator=rng)
                parts.append(class_indices[perm[:amt]])
            return torch.cat(parts)

        # With canaries: split each class budget between raw and canary samples
        train_num_canaries = int(self.train_size * share)
        train_canary_dist = distribute_a_across_b(train_num_canaries, num_classes)
        parts = []
        for cls, (amt_raw, amt_canary) in enumerate(
            zip(train_dist - train_canary_dist, train_canary_dist, strict=True)
        ):
            parts.append(raw_lookup[cls][:amt_raw])
            parts.append(canary_lookup[cls][:amt_canary])
        return torch.cat(parts)

    def apply_canary(self, dataset: Dataset, num_classes: int) -> "Dataset | CanaryDataset":
        """Wrap the dataset in a CanaryDataset, optionally injecting canary samples.

        When no canary config is set, this still handles train_size subsetting.
        When canaries are configured, designated samples get their images modified
        and labels reassigned via derangement.
        """
        num_samples = len(dataset)  # type: ignore[arg-type]

        # No canary config at all
        if self.canary is None:
            if self.train_size is None:
                return dataset
            rng = self._make_rng()
            labels = self._extract_labels(dataset, num_samples)
            subset_indices = self._compute_subset_indices(
                labels, num_samples, num_classes, rng
            )
            return CanaryDataset(
                dataset=dataset,
                subset_indices=subset_indices,
                num_classes=num_classes,
            )

        # Canary config present but share is zero — nothing to inject
        share = self.canary.share
        if share == 0:
            if self.train_size is None:
                return dataset
            rng = self._make_rng()
            labels = self._extract_labels(dataset, num_samples)
            subset_indices = self._compute_subset_indices(
                labels, num_samples, num_classes, rng
            )
            return CanaryDataset(
                dataset=dataset,
                subset_indices=subset_indices,
                num_classes=num_classes,
            )

        if self.seed is None:
            raise ValueError("A seed is required for deterministic canary assignment.")

        rng = self._make_rng()
        labels = self._extract_labels(dataset, num_samples)

        num_canaries = int(num_samples * share)
        canary_distribution = distribute_a_across_b(num_canaries, num_classes)

        # Partition each class into canary vs. raw indices
        canary_lookup: dict[int, torch.Tensor] = {}
        raw_lookup: dict[int, torch.Tensor] = {}
        for cls, amt in enumerate(canary_distribution):
            class_indices = (labels == cls).nonzero().squeeze(-1)
            perm = torch.randperm(len(class_indices), generator=rng)
            canary_lookup[cls] = class_indices[perm[:amt]]
            raw_lookup[cls] = class_indices[perm[amt:]]

        subset_indices = self._compute_subset_indices(
            labels, num_samples, num_classes, rng,
            canary_lookup=canary_lookup,
            raw_lookup=raw_lookup,
            share=share,
        )

        # Build canary transform and deranged labels
        canary_indices = torch.cat(list(canary_lookup.values()))
        input_shape = dataset[0][0].shape
        canary_transform = create_canary_generator(config=self.canary, dim=input_shape)
        canary_labels = random_derange_indices(
            canary_lookup={k: v.tolist() for k, v in canary_lookup.items()},
            seed=self.seed,
        )

        return CanaryDataset(
            dataset=dataset,
            subset_indices=subset_indices,
            num_classes=num_classes,
            canary_indices=canary_indices,
            canary_labels=canary_labels,
            canary_transform=canary_transform,
        )

    def __call__(self) -> DataContainer:
        """Build the dataset, apply canaries and subsetting, then apply masking."""
        container = self.data()
        train = self.apply_canary(container.train, container.num_classes)
        print(f"------------- 1 {len(train)}")
        train = self.apply_mask(train, container.num_classes)
        print(f"------------- 2 {len(train)}")


        return DataContainer(
            train=train,  # type: ignore[arg-type]
            test=container.test,
            num_classes=container.num_classes,
            input_shape=container.input_shape,
            normalization=container.normalization,
        )
