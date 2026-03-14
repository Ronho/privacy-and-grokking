from typing import Annotated, Literal

import torch
from pydantic import BaseModel, Field


class AdamConfig(BaseModel):
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


class AdamWConfig(BaseModel):
    name: Literal["AdamW"] = "AdamW"

    lr: float = 0.001
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-08
    weight_decay: float = 0.01
    amsgrad: bool = False

    def __call__(self, params) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            params,
            **self.model_dump(exclude={"name"}),
        )


class RMSpropConfig(BaseModel):
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


class SGDConfig(BaseModel):
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


Optimizer = Annotated[
    AdamConfig | AdamWConfig | RMSpropConfig | SGDConfig, Field(discriminator="name")
]
