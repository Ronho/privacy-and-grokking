import torch
from pydantic import BaseModel, Field
from torch.utils.data import Dataset, TensorDataset
from torchvision import transforms

from privacy_and_grokking.datasets.canaries import Canary, CanaryConfig, create_canary_generator
from privacy_and_grokking.datasets.datasets import Datasets, Normalization, get_dataset


class CanarySubset(TensorDataset):
    def __init__(
        self,
        dataset: Dataset,
        norm: Normalization,
        subset_indices: torch.Tensor,
        canary_indices: torch.Tensor,
        canary_transform: Canary | None,
        input_shape: torch.Size,
        num_classes: int,
    ) -> None:
        self.dataset = dataset
        self.subset_indices = subset_indices
        self.canary_indices = canary_indices
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(norm.mean, norm.std)]
        )
        self.target_transform = transforms.Lambda(lambda y: torch.tensor(y, dtype=torch.long))
        self.canary_transform = canary_transform

        # Stored for accessibility
        self.input_shape = input_shape
        self.num_classes = num_classes

    def __len__(self):
        return len(self.subset_indices)

    def __getitem__(self, idx):
        if idx >= len(self.subset_indices):
            raise IndexError("Index out of range.")

        index = self.subset_indices[idx]
        img, lbl = self.dataset[index]

        if index in self.canary_indices:
            if self.canary_transform is None:
                raise Exception("No canary transform provided but canary called.")
            img, lbl = self.canary_transform(img, lbl)

        img = self.transform(img)
        lbl = self.target_transform(lbl)

        return img, lbl


class DatasetConfig(BaseModel):
    name: Datasets
    train_size: int | None = Field(ge=0)
    canary_share: float = Field(ge=0, le=1)
    canary_config: CanaryConfig
    seed: int | None = None


def distribute_a_across_b(a: int, b: int) -> torch.Tensor:
    base = a // b
    remainder = a % b
    distribution = torch.zeros(b, dtype=torch.int)
    distribution.fill_(base)
    distribution[:remainder] += 1
    return distribution


def generate_datasets(config: DatasetConfig) -> tuple[CanarySubset, CanarySubset]:
    container = get_dataset(name=config.name)

    g = torch.Generator()
    if config.seed is not None:
        g.manual_seed(config.seed)

    # Note: We generate the canary split first in order to make sure that the
    # indices do not change between different train sizes.
    num_canaries = int(len(container.train) * config.canary_share)
    canary_distribution = distribute_a_across_b(num_canaries, container.num_classes)

    raw_lookup = {}
    canary_lookup = {}
    for cls, amt in enumerate(canary_distribution):
        class_indices = (container.train.targets == cls).nonzero().squeeze()
        perm = torch.randperm(len(class_indices), generator=g)
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
    canary_generator = create_canary_generator(
        config=config.canary_config, dim=container.input_shape, num_classes=container.num_classes
    )

    train = CanarySubset(
        dataset=container.train,
        norm=container.normalization,
        subset_indices=subset_indices,
        canary_indices=canary_indices,
        canary_transform=canary_generator,
        input_shape=container.input_shape,
        num_classes=container.num_classes,
    )
    test = CanarySubset(
        dataset=container.test,
        norm=container.normalization,
        subset_indices=torch.arange(0, len(container.test)),
        canary_indices=torch.empty(0),
        canary_transform=None,
        input_shape=container.input_shape,
        num_classes=container.num_classes,
    )
    return train, test


if __name__ == "__main__":
    from privacy_and_grokking.datasets.canaries import SquareWatermarkCanaryConfig

    canary_config = SquareWatermarkCanaryConfig(square_size=3)
    config = DatasetConfig(
        name="mnist", train_size=None, canary_share=0.0023, canary_config=canary_config, seed=5
    )
    train, test = generate(config=config)
    print(len(train), len(test))
