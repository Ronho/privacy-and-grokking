from abc import abstractmethod

import torch
from pydantic import BaseModel


class SchedulerConfig(BaseModel):
    name: str

    @abstractmethod
    def __call__(
        self, optimizer: torch.optim.Optimizer, **kwargs
    ) -> torch.optim.lr_scheduler.LRScheduler: ...
