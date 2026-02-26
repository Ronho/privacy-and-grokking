from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from privacy_and_grokking.datasets import DatasetConfig, MaskingConfig
from privacy_and_grokking.models import Model


class MSELoss(BaseModel):
    name: Literal["mse"] = "mse"


class CrossEntropyLoss(BaseModel):
    name: Literal["cross_entropy"] = "cross_entropy"

type Loss = MSELoss | CrossEntropyLoss


class AdamW(BaseModel):
    name: Literal["AdamW"] = "AdamW"

    learning_rate: float
    weight_decay: float

type Optimizer = AdamW


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: Model
    seed: int
    batch_size: int
    initialization_scale: float | None
    loss: Loss = Field(discriminator="name")
    optimizer: Optimizer = Field(discriminator="name")
    dataset: DatasetConfig
    dataset_mask: MaskingConfig
    dataset_mask_idx: int = 0

    @property
    def name(self) -> str:
        return f"{self.model.upper()}_{self.dataset.name.upper()}_{self.dataset_mask.name.upper()}_{self.optimizer.name.upper()}_{self.loss.name.upper()}"

    @property
    def full_name(self) -> str:
        return f"{self.name}_{self.dataset_mask_idx}"
