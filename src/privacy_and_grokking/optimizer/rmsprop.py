from typing import Literal

import torch

from privacy_and_grokking.optimizer.base import OptimizerConfig


class RMSpropConfig(OptimizerConfig):
    name: Literal["RMSprop"] = "RMSprop"

    lr: float = 0.01
    alpha: float = 0.99
    eps: float = 1e-08
    weight_decay: float = 0
    momentum: float = 0
    centered: bool = False

    def __call__(self, params) -> torch.optim.Optimizer:
        return torch.optim.RMSprop(
            params,
            **self.model_dump(exclude={"name"}),
        )
