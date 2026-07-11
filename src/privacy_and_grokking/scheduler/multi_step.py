from typing import Literal

import torch

from privacy_and_grokking.scheduler.base import SchedulerConfig


class MultiStepLRConfig(SchedulerConfig):
    name: Literal["MultiStepLR"] = "MultiStepLR"

    milestones: list[int]
    gamma: float = 0.1

    def __call__(
        self, optimizer: torch.optim.Optimizer, **kwargs
    ) -> torch.optim.lr_scheduler.LRScheduler:
        last_epoch: int = kwargs.get("last_epoch", -1)
        
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.milestones,
            gamma=self.gamma,
            last_epoch=last_epoch,
        )
