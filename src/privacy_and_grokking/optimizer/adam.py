from typing import Literal

import torch

from privacy_and_grokking.optimizer.base import OptimizerConfig


class AdamConfig(OptimizerConfig):
    name: Literal["Adam"] = "Adam"

    lr: float = 0.001
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-08
    weight_decay: float = 0
    amsgrad: bool = False

    def __call__(self, params) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            params,
            **self.model_dump(exclude={"name"}),
        )
