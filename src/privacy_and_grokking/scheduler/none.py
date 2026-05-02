from typing import Literal

import torch

from privacy_and_grokking.scheduler.base import SchedulerConfig


class NoScheduler(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self) -> None: ...
    def step(self, epoch: int | None = None) -> None: ...


class NoSchedulerConfig(SchedulerConfig):
    name: Literal["None"] = "None"

    def __call__(self, optimizer: torch.optim.Optimizer, **kwargs) -> torch.optim.lr_scheduler.LRScheduler:
        return NoScheduler()
