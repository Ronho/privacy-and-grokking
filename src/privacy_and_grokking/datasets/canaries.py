from collections.abc import Sequence
from enum import StrEnum
from typing import Protocol

import torch
from pydantic import BaseModel, Field


class Canary(Protocol):  # noqa: F811
    def __call__(self, image: torch.Tensor) -> torch.Tensor: ...


class UniformNoiseCanary:
    def __init__(self, dim: Sequence[int]):
        self.dim = dim
        self.rng = torch.Generator()

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        seed = torch.hash_tensor(image)
        self.rng.manual_seed(seed.item())
        image = torch.rand(self.dim, generator=self.rng)

        return image


class SquareWatermarkCanary:
    def __init__(self, dim: Sequence[int], square_size: int):
        self.dim = dim
        self.square_size = min(square_size, dim[-2], dim[-1])

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        image[:, -self.square_size:, -self.square_size:] = 1.0
        return image


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


def create_canary_generator(config: CanaryConfig, dim: Sequence[int]) -> Canary:
    match config.name:
        case Canaries.UNIFORM_NOISE:
            return UniformNoiseCanary(dim=dim)
        case Canaries.SQUARE_WATERMARK:
            return SquareWatermarkCanary(dim=dim, square_size=config.square_size)
        case _:
            raise ValueError(f"Unknown canary '{config.name}'.")
