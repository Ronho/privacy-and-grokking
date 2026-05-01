from abc import abstractmethod

import torch
from pydantic import BaseModel


class OptimizerConfig(BaseModel):
    name: str

    @abstractmethod
    def __call__(self, params) -> torch.optim.Optimizer: ...
