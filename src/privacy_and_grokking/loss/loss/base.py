from abc import abstractmethod
from collections.abc import Callable
from typing import Literal

import torch
from pydantic import BaseModel

type LossType = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class LossConfig(BaseModel):
    name: str
    reduction: Literal["none", "mean", "sum"] = "mean"

    @abstractmethod
    def __call__(self, **kwargs) -> LossType: ...
