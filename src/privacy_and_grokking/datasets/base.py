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

    def apply_canary(self, dataset: Dataset, num_classes: int) -> "Dataset | CanaryDataset":
        """Apply canary transforms to a share of the training samples.

        Canary samples get their images modified by the canary transform and their
        labels reassigned via derangement (no sample keeps its original label).
        """
        if self.canary is None or self.canary.share <= 0:
            return dataset

        if self.seed is None:
            raise ValueError("A seed is required for deterministic canary assignment.")

        rng = torch.Generator()
        rng.manual_seed(self.seed)

        num_samples = len(dataset)  # type: ignore[arg-type]
        num_canaries = int(num_samples * self.canary.share)

        # Collect labels and group indices by class
        labels = torch.tensor([dataset[i][1] for i in range(num_samples)], dtype=torch.long)

        canary_lookup: dict[int, torch.Tensor] = {}
        raw_lookup: dict[int, torch.Tensor] = {}
        canary_distribution = distribute_a_across_b(num_canaries, num_classes)

        for cls, amt in enumerate(canary_distribution):
            class_indices = (labels == cls).nonzero().squeeze(-1)
            perm = torch.randperm(len(class_indices), generator=rng)
            canary_lookup[cls] = class_indices[perm[:amt]]
            raw_lookup[cls] = class_indices[perm[amt:]]

        # Build subset indices respecting train_size if set
        if self.train_size is not None:
            if self.train_size > num_samples:
                raise ValueError("train_size exceeds dataset size.")
            train_num_canaries = int(self.train_size * self.canary.share)
            train_canary_dist = distribute_a_across_b(train_num_canaries, num_classes)
            train_dist = distribute_a_across_b(self.train_size, num_classes)
            subset_indices_list: list[torch.Tensor] = []
            for cls, (amt_train, amt_canary) in enumerate(
                zip(train_dist - train_canary_dist, train_canary_dist, strict=True)
            ):
                subset_indices_list.append(raw_lookup[cls][:amt_train])
                subset_indices_list.append(canary_lookup[cls][:amt_canary])
            subset_indices = torch.cat(subset_indices_list)
        else:
            subset_indices = torch.arange(num_samples)

        # Gather all canary indices and generate the transform + deranged labels
        canary_indices = torch.cat(list(canary_lookup.values()))

        # Determine input shape from first sample
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
        """Build the dataset, apply canaries, and apply masking."""
        container = self.data()

        train = self.apply_canary(container.train, container.num_classes)
        train = self.apply_mask(train, container.num_classes)

        return DataContainer(
            train=train,  # type: ignore[arg-type]
            test=container.test,
            num_classes=container.num_classes,
            input_shape=container.input_shape,
            normalization=container.normalization,
        )
