from typing import Literal

import torch

from privacy_and_grokking.optimizer.base import OptimizerConfig


class SGDConfig(OptimizerConfig):
    name: Literal["SGD"] = "SGD"

    lr: float = 0.001
    momentum: float = 0
    dampening: float = 0
    weight_decay: float = 0
    nesterov: bool = False

    def __call__(self, params) -> torch.optim.Optimizer:
        return torch.optim.SGD(
            params,
            **self.model_dump(exclude={"name"}),
        )
