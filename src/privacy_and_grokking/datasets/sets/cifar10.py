from typing import Literal

from torchvision import datasets, transforms

from privacy_and_grokking.datasets.sets.base import (
    CACHE_PATH,
    DataContainer,
    DatasetConfig,
    Normalization,
)


class CIFAR10Config(DatasetConfig):
    name: Literal["cifar10"] = "cifar10"

    def __call__(self) -> DataContainer:
        transform = transforms.ToTensor()
        CACHE_PATH.mkdir(exist_ok=True)
        train = datasets.CIFAR10(
            root=CACHE_PATH, train=True, download=True, transform=transform
        )
        test = datasets.CIFAR10(
            root=CACHE_PATH, train=False, download=True, transform=transform
        )
        return DataContainer(
            train=train,
            test=test,
            num_classes=10,
            input_shape=train[0][0].shape,
            normalization=Normalization(
                mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]
            ),
        )
