from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path

import torch
from pydantic import BaseModel
from torch.utils.data import Dataset

CACHE_PATH = Path(__file__).parent.parent.parent.parent.parent.resolve() / "cache"

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

class DatasetConfig(BaseModel):
    name: str

    @abstractmethod
    def __call__(self) -> DataContainer: ...
