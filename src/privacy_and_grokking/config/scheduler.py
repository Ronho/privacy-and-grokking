from typing import Annotated, Literal

import torch
from pydantic import BaseModel, Field


class NoScheduler(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self) -> None: ...
    def step(self, epoch: int | None = None) -> None: ...


class NoSchedulerConfig(BaseModel):
    name: Literal["None"] = "None"

    def __call__(self, optimizer, **kwargs) -> torch.optim.lr_scheduler.LRScheduler:
        return NoScheduler()


class CosineAnnealingLRConfig(BaseModel):
    name: Literal["CosineAnnealingLR"] = "CosineAnnealingLR"

    min_lr: float

    def __call__(self, optimizer, **kwargs) -> torch.optim.lr_scheduler.LRScheduler:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs["optimization_steps"],
            eta_min=cfg.min_lr,
            last_epoch=kwargs["checkpoint"],
        )


Scheduler = Annotated[NoSchedulerConfig | CosineAnnealingLRConfig, Field(discriminator="name")]
