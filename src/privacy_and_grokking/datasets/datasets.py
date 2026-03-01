from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


@dataclass
class Normalization:
    mean: list[float]
    std: list[float]


@dataclass
class DataContainer:
    train: Dataset
    test: Dataset
    num_classes: int
    input_shape: torch.Size
    normalization: Normalization


class Datasets(StrEnum):
    MNIST = "mnist"
    CIFAR10 = "cifar10"


CACHE_PATH = Path(__file__).parent.parent.parent.parent.resolve() / "cache"


def get_mnist() -> DataContainer:
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


def get_cifar10() -> DataContainer:
    transform = transforms.ToTensor()
    CACHE_PATH.mkdir(exist_ok=True)
    train = datasets.CIFAR10(root=CACHE_PATH, train=True, download=True, transform=transform)
    test = datasets.CIFAR10(root=CACHE_PATH, train=False, download=True, transform=transform)
    return DataContainer(
        train=train,
        test=test,
        num_classes=10,
        input_shape=train[0][0].shape,
        normalization=Normalization(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]),
    )


def get_dataset(name: Datasets) -> DataContainer:
    match name:
        case Datasets.MNIST:
            return get_mnist()
        case Datasets.CIFAR10:
            return get_cifar10()
        case _:
            raise ValueError(f"Unknown dataset: {name}")
