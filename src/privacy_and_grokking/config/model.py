from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from privacy_and_grokking.datasets import DatasetConfig, MaskingConfig
from privacy_and_grokking.models import Model

type Loss = Literal["mse", "cross_entropy"]


class LossConfig(BaseModel):
    name: Loss


class MSELoss(LossConfig):
    name: Loss = "mse"


class CrossEntropyLoss(LossConfig):
    name: Loss = "cross_entropy"


type Optimizer = Literal["AdamW"]


class OptimizerConfig(BaseModel):
    name: Optimizer


class AdamW(OptimizerConfig):
    name: Optimizer = "AdamW"

    learning_rate: float
    weight_decay: float


class TrainConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    code_version: str
    batch_size: int
    initialization_scale: float | None
    log_frequency: int
    optimization_steps: int
    seed: int
    loss: LossConfig = Field(discriminator="name")
    model: Model
    optimizer: AdamW = Field(discriminator="name")
    dataset: DatasetConfig
    dataset_mask: MaskingConfig
    dataset_mask_idx: int = 0

    @property
    def full_name(self) -> str:
        return f"{self.name}_{self.dataset_mask_idx}"
