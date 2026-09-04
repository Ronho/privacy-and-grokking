from collections.abc import Sequence
from typing import Literal

import torch
from pydantic import Field

from privacy_and_grokking.datasets.canaries.base import Canary, CanaryConfig


class SquareWatermarkCanary:
    def __init__(self, dim: Sequence[int], square_size: int):
        self.dim = dim
        self.square_size = min(square_size, dim[-2], dim[-1])

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        image[..., -self.square_size :, -self.square_size :] = 1.0
        return image


class SquareWatermarkCanaryConfig(CanaryConfig):
    name: Literal["square_watermark"] = "square_watermark"
    square_size: int = Field(ge=0)

    def __call__(self, dim: Sequence[int]) -> Canary:
        return SquareWatermarkCanary(dim=dim, square_size=self.square_size)
