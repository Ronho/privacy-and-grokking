from abc import abstractmethod
from collections.abc import Callable

import torch
from pydantic import BaseModel


class RegularizerSourceConfig(BaseModel):
    name: str


NoiseGeneratorType = Callable[
    [torch.Tensor], torch.Tensor
]  # Both Tensors are of shape (B, C, H, W)


class NoiseRegularizerSourceConfig(BaseModel):
    num_noisy_samples: int = 1

    @abstractmethod
    def __call__(self) -> NoiseGeneratorType: ...
