from typing import Literal

from torch.utils.data import Subset
from torchvision import datasets, transforms

from privacy_and_grokking.datasets.sets.base import (
    CACHE_PATH,
    DataContainer,
    DatasetConfig,
    Normalization,
)


class MNISTConfig(DatasetConfig):
    name: Literal["mnist"] = "mnist"
    subset_size: int | None = None
    pad: bool = False

    def __call__(self) -> DataContainer:
        transform_list = []
        if self.pad:
            transform_list.append(transforms.Pad(2))
        transform_list.append(transforms.ToTensor())
        transform = transforms.Compose(transform_list)

        CACHE_PATH.mkdir(exist_ok=True)
        train = datasets.MNIST(root=CACHE_PATH, train=True, download=True, transform=transform)

        # NOTE: Deterministically sample a perfectly balanced 50k dataset (5000 per class).
        # This ensures every run selects the exact same 50k samples.
        import torch

        balanced_indices = []
        for c in range(10):
            c_indices = (train.targets == c).nonzero().view(-1)
            balanced_indices.append(c_indices[:5000])
        train = Subset(train, torch.cat(balanced_indices).tolist())

        if self.subset_size is not None:
            train = Subset(train, list(range(self.subset_size)))
        test = datasets.MNIST(root=CACHE_PATH, train=False, download=True, transform=transform)
        return DataContainer(
            train=train,
            test=test,
            num_classes=10,
            input_shape=train[0][0].shape,
            normalization=Normalization(mean=[0.1307], std=[0.3081]),
        )
