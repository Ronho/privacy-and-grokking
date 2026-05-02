from abc import abstractmethod
from collections.abc import Sequence
from typing import Protocol

import torch
from pydantic import BaseModel, Field


class Canary(Protocol):
    def __call__(self, image: torch.Tensor) -> torch.Tensor: ...


class CanaryConfig(BaseModel):
    name: str
    share: float = Field(ge=0, le=1, default=0)

    @abstractmethod
    def __call__(self, dim: Sequence[int]) -> Canary: ...
