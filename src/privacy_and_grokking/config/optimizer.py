from typing import Annotated, Literal

import torch
from pydantic import BaseModel, Field


class AdamConfig(BaseModel):
    name: Literal["Adam"] = "Adam"

    learning_rate: float
    weight_decay: float

    def __call__(self, params) -> torch.optim.Optimizer:
        return torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)


class AdamWConfig(BaseModel):
    name: Literal["AdamW"] = "AdamW"

    learning_rate: float
    weight_decay: float

    def __call__(self, params) -> torch.optim.Optimizer:
        return torch.optim.AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)


Optimizer = Annotated[AdamConfig | AdamWConfig, Field(discriminator="name")]
