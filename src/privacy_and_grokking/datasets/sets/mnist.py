from typing import Literal

from torchvision import datasets, transforms

from privacy_and_grokking.datasets.sets.base import (
    CACHE_PATH,
    DataContainer,
    DatasetConfig,
    Normalization,
)


class MNISTConfig(DatasetConfig):
    name: Literal["mnist"] = "mnist"

    def __call__(self) -> DataContainer:
        transform = transforms.ToTensor()
        CACHE_PATH.mkdir(exist_ok=True)
        train = datasets.MNIST(root=CACHE_PATH, train=True, download=True, transform=transform)
        test = datasets.MNIST(root=CACHE_PATH, train=False, download=True, transform=transform)
        return DataContainer(
            train=train,
            test=test,
            num_classes=10,
            input_shape=train[0][0].shape,
            normalization=Normalization(mean=[0.1307], std=[0.3081]),
        )
