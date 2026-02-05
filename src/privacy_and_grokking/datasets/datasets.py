from dataclasses import dataclass
from enum import StrEnum

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from ..path_keeper import get_path_keeper


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


def get_mnist() -> DataContainer:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    target_transform = transforms.Lambda(lambda y: torch.tensor(y, dtype=torch.long))
    pk = get_path_keeper()
    train = datasets.MNIST(
        root=pk.CACHE,
        train=True,
        transform=transform,
        target_transform=target_transform,
        download=True,
    )
    test = datasets.MNIST(
        root=pk.CACHE,
        train=False,
        transform=transform,
        target_transform=target_transform,
        download=True,
    )
    return DataContainer(
        train=train,
        test=test,
        num_classes=10,
        input_shape=train[0][0].shape,
        normalization=Normalization(mean=[0.1307], std=[0.3081]),
    )


def get_cifar10() -> DataContainer:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # https://github.com/kuangliu/pytorch-cifar/issues/19
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )
    target_transform = transforms.Lambda(lambda y: torch.tensor(y, dtype=torch.long))
    pk = get_path_keeper()
    train = datasets.CIFAR10(
        root=pk.CACHE,
        train=True,
        transform=transform,
        target_transform=target_transform,
        download=True,
    )
    test = datasets.CIFAR10(
        root=pk.CACHE,
        train=False,
        transform=transform,
        target_transform=target_transform,
        download=True,
    )
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
