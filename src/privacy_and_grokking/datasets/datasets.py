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

type Data = Literal["mnist"]

def get_mnist() -> DataContainer:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    pk = get_path_keeper()
    train = datasets.MNIST(
        root=pk.CACHE,
        train=True, 
        transform=transform,
        download=True
    )
    test = datasets.MNIST(
        root=pk.CACHE,
        train=False, 
        transform=transform,
        download=True
    )
    info: DataContainer = {
        "trainval": train,
        "test": test,
        "num_classes": 10,
        "input_shape": train[0][0].shape
    }
    return info

def create_dataset(name: Data) -> DataContainer:
    match name.lower():
        case "mnist":
            return get_mnist()
        case _:
            raise ValueError(f"Unknown dataset: {name}")
