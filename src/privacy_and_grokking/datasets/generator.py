from typing import Self

import torch
from pydantic import BaseModel, Field, model_validator
from torch.utils.data import Dataset, TensorDataset
from torchvision import transforms

from privacy_and_grokking.datasets.canaries import Canary, CanaryConfig, create_canary_generator
from privacy_and_grokking.datasets.canary_class_assignment import random_derange_indices
from privacy_and_grokking.datasets.datasets import Datasets, Normalization, get_dataset
from privacy_and_grokking.logger import get_logger

logger = get_logger()


class CanarySubset(TensorDataset):
    def __init__(
        self,
        dataset: Dataset,
        norm: Normalization,
        subset_indices: torch.Tensor,
        input_shape: torch.Size,
        num_classes: int,
        canary_indices: torch.Tensor | None = None,
        canary_labels: torch.Tensor | None = None,
        canary_transform: Canary | None = None,
    ) -> None:
        self.dataset = dataset
        self.subset_indices = subset_indices
        self.transform = transforms.Normalize(norm.mean, norm.std)
        self.target_transform = transforms.Lambda(
            lambda y: y.detach().clone().to(dtype=torch.long)
            if isinstance(y, torch.Tensor)
            else torch.tensor(y, dtype=torch.long)
        )

        if canary_indices is not None:
            if canary_labels is None:
                raise ValueError("canary_labels must be provided if canary_indices is provided")
            if canary_transform is None:
                raise ValueError("canary_transform must be provided if canary_indices is provided")
            self.canary_labels = canary_labels
            self.canary_transform = canary_transform
            self.canary_indices = canary_indices
        else:
            self.canary_indices = torch.empty(0)  # Placeholder that allows lookup.

        # Stored for accessibility
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.norm = norm

    def __len__(self):
        return len(self.subset_indices)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        if idx >= len(self.subset_indices):
            raise IndexError("Index out of range.")

        index = self.subset_indices[idx]
        img, lbl = self.dataset[index]

        if index in self.canary_indices:
            if self.canary_transform is None:
                raise Exception("No canary transform provided but canary called.")
            img = self.canary_transform(img)
            lbl = self.canary_labels[
                (self.canary_indices == index).nonzero(as_tuple=True)[0].item()
            ]

        img = self.transform(img)
        lbl = self.target_transform(lbl)

        return img, lbl


class DatasetConfig(BaseModel):
    name: Datasets
    train_size: int | None = Field(ge=0)
    canary_share: float = Field(ge=0, le=1)
    canary_config: CanaryConfig | None = None
    seed: int | None = None

    @model_validator(mode="after")
    def validate_canary_config(self) -> Self:
        if self.canary_share == 0 and self.canary_config is not None:
            raise ValueError("canary_config must be None if canary_share is 0")
        if self.canary_share > 0 and self.canary_config is None:
            raise ValueError("canary_config must be provided if canary_share > 0")
        return self


def distribute_a_across_b(a: int, b: int) -> torch.Tensor:
    base = a // b
    remainder = a % b
    distribution = torch.zeros(b, dtype=torch.int)
    distribution.fill_(base)
    distribution[:remainder] += 1
    return distribution


def generate_datasets(config: DatasetConfig) -> tuple[CanarySubset, CanarySubset]:
    container = get_dataset(name=config.name)

    rng = torch.Generator()
    if config.seed is not None:
        rng.manual_seed(config.seed)

    # Note: We generate the canary split first in order to make sure that the
    # indices do not change between different train sizes.
    num_canaries = int(len(container.train) * config.canary_share)
    canary_distribution = distribute_a_across_b(num_canaries, container.num_classes)

    raw_lookup = {}
    canary_lookup = {}
    for cls, amt in enumerate(canary_distribution):
        class_indices = (torch.Tensor(container.train.targets) == cls).nonzero().squeeze()
        perm = torch.randperm(len(class_indices), generator=rng)
        canary_lookup[cls] = class_indices[perm[:amt]]
        raw_lookup[cls] = class_indices[perm[amt:]]

    if config.train_size is not None and config.train_size > len(container.train):
        raise ValueError("Train size out of bounds")

    if config.train_size is None:
        subset_indices = torch.arange(0, len(container.train))
    else:
        train_num_canaries = int(config.train_size * config.canary_share)
        train_canary_distribution = distribute_a_across_b(train_num_canaries, container.num_classes)
        train_distribution = distribute_a_across_b(config.train_size, container.num_classes)
        subset_indices = []
        for cls, (amt_train, amt_canary) in enumerate(
            zip(
                train_distribution - train_canary_distribution,
                train_canary_distribution,
                strict=True,
            )
        ):
            subset_indices.append(raw_lookup[cls][:amt_train])
            subset_indices.append(canary_lookup[cls][:amt_canary])
        subset_indices = torch.concat(subset_indices)

    # CanarySubset does not care if canary_indices contains more elements than are in subset_indices
    canary_indices = torch.concat(list(canary_lookup.values()))
    canary_generator = None
    if config.canary_share > 0:
        if config.canary_config is not None:
            canary_generator = create_canary_generator(
                config=config.canary_config, dim=container.input_shape
            )
        else:
            raise ValueError("canary_config must be provided if canary_share > 0")

    train = CanarySubset(
        dataset=container.train,
        norm=container.normalization,
        subset_indices=subset_indices,
        canary_indices=canary_indices if config.canary_share > 0 else None,
        canary_transform=canary_generator,
        canary_labels=random_derange_indices(canary_lookup=canary_lookup, seed=config.seed)
        if config.canary_share > 0
        else None,
        input_shape=container.input_shape,
        num_classes=container.num_classes,
    )
    test = CanarySubset(
        dataset=container.test,
        norm=container.normalization,
        subset_indices=torch.arange(0, len(container.test)),
        input_shape=container.input_shape,
        num_classes=container.num_classes,
    )
    return train, test
