from collections.abc import Sequence
from enum import StrEnum
from typing import Protocol

import torch
from pydantic import BaseModel, Field


class Canary(Protocol):  # noqa: F811
    def __call__(self, image: torch.Tensor, label: int) -> tuple[torch.Tensor, int]:
        ...

class UniformNoiseCanary:
    def __init__(self, dim: Sequence[int], num_classes: int):
        self.dim = dim
        self.num_classes = num_classes
        self.generator = torch.Generator()

    def __call__(self, image: torch.Tensor, label: int) -> tuple[torch.Tensor, int]:
        seed = torch.hash_tensor(image)
        self.generator.manual_seed(seed)
        shift = torch.randint(1, self.num_classes, (1,), generator=self.generator).item()
        label = (label + shift) % self.num_classes

        image = torch.rand(self.dim, generator=self.generator)

        return image, label

class SquareWatermarkCanary:
    def __init__(self, dim: Sequence[int], num_classes: int, square_size: int):
        self.dim = dim
        self.num_classes = num_classes
        self.square_size = min(square_size, dim[-2], dim[-1])
        self.generator = torch.Generator()

    def __call__(self, image: torch.Tensor, label: int) -> tuple[torch.Tensor, int]:
        seed = torch.hash_tensor(image)
        self.generator.manual_seed(seed)
        shift = torch.randint(1, self.num_classes, (1,), generator=self.generator).item()
        label = (label + shift) % self.num_classes

        image[:, :, -self.square_size:, -self.square_size:] = 1.0

        return image, label


class Canaries(StrEnum):
    UNIFORM_NOISE = "uniform_noise"
    SQUARE_WATERMARK = "square_watermark"

class CanaryConfig(BaseModel):
    name: Canaries

class UniformNoiseCanaryConfig(CanaryConfig):
    name: Canaries = Canaries.UNIFORM_NOISE

class SquareWatermarkCanaryConfig(CanaryConfig):
    name: Canaries = Canaries.SQUARE_WATERMARK
    square_size: int = Field(ge=0)

def create_canary_generator(config: CanaryConfig, dim: Sequence[int], num_classes: int) -> Canary:
    match config.name:
        case Canaries.UNIFORM_NOISE:
            return UniformNoiseCanary(dim=dim, num_classes=num_classes)
        case Canaries.SQUARE_WATERMARK:
            return SquareWatermarkCanary(dim=dim, num_classes=num_classes, **config.model_dump(exclude="name"))
        case _:
            raise ValueError(f"Unknown canary '{config.name}'.")
