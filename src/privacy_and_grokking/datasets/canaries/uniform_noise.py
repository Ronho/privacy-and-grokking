from collections.abc import Sequence
from typing import Literal

import torch

from privacy_and_grokking.datasets.canaries.base import Canary, CanaryConfig


class UniformNoiseCanary:
    def __init__(self, dim: Sequence[int]):
        self.dim = dim
        self.rng = torch.Generator()

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        seed = torch.hash_tensor(image)
        self.rng.manual_seed(seed.item())
        image = torch.rand(self.dim, generator=self.rng)
        return image


class UniformNoiseCanaryConfig(CanaryConfig):
    name: Literal["uniform_noise"] = "uniform_noise"

    def __call__(self, dim: Sequence[int]) -> Canary:
        return UniformNoiseCanary(dim=dim)
