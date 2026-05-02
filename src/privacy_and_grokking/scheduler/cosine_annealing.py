from typing import Literal

import torch

from privacy_and_grokking.scheduler.base import SchedulerConfig


class CosineAnnealingLRConfig(SchedulerConfig):
    name: Literal["CosineAnnealingLR"] = "CosineAnnealingLR"

    min_lr: float

    def __call__(
        self, optimizer: torch.optim.Optimizer, **kwargs
    ) -> torch.optim.lr_scheduler.LRScheduler:
        optimization_steps: int | None = kwargs.get("optimization_steps")
        if optimization_steps is None:
            raise ValueError(
                "optimization_steps must be provided as a keyword argument "
                "to CosineAnnealingLRConfig"
            )
        last_epoch: int = kwargs.get("last_epoch", -1)
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=optimization_steps,
            eta_min=self.min_lr,
            last_epoch=last_epoch,
        )
