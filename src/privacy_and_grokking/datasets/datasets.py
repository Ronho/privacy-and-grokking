import torch

from torch.utils.data import Dataset
from torchvision import datasets, transforms
from typing import TypedDict, Literal
from ..path_keeper import get_path_keeper


class DataContainer(TypedDict):
    trainval: Dataset
    test: Dataset
    num_classes: int
    input_shape: torch.Size
    normalization: dict[str, tuple[float, ...]]

type Data = Literal["mnist", "cifar10"]

def get_mnist() -> DataContainer:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    target_transform = transforms.Lambda(lambda y: torch.tensor(y, dtype=torch.long))
    pk = get_path_keeper()
    train = datasets.MNIST(
        root=pk.CACHE,
        train=True, 
        transform=transform,
        target_transform=target_transform,
        download=True
    )
    test = datasets.MNIST(
        root=pk.CACHE,
        train=False, 
        transform=transform,
        target_transform=target_transform,
        download=True
    )
    info: DataContainer = {
        "trainval": train,
        "test": test,
        "num_classes": 10,
        "input_shape": train[0][0].shape,
        "normalization": {
            "mean": (0.1307,),
            "std": (0.3081,)
        }
    }
    return info

def get_cifar10() -> DataContainer:
    transform = transforms.Compose([
        transforms.ToTensor(),
        # https://github.com/kuangliu/pytorch-cifar/issues/19
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    target_transform = transforms.Lambda(lambda y: torch.tensor(y, dtype=torch.long))
    pk = get_path_keeper()
    train = datasets.CIFAR10(
        root=pk.CACHE,
        train=True, 
        transform=transform,
        target_transform=target_transform,
        download=True
    )
    test = datasets.CIFAR10(
        root=pk.CACHE,
        train=False, 
        transform=transform,
        target_transform=target_transform,
        download=True
    )
    info: DataContainer = {
        "trainval": train,
        "test": test,
        "num_classes": 10,
        "input_shape": train[0][0].shape,
        "normalization": {
            "mean": (0.4914, 0.4822, 0.4465),
            "std": (0.247, 0.243, 0.261)
        }
    }
    return info

def create_dataset(name: Data) -> DataContainer:
    match name.lower():
        case "mnist":
            return get_mnist()
        case "cifar10":
            return get_cifar10()
        case _:
            raise ValueError(f"Unknown dataset: {name}")
