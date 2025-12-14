from enum import StrEnum
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

class MSELoss(BaseModel):
    loss_name: Literal["MSELoss"] = "MSELoss"

class AdamW(BaseModel):
    optim_name: Literal["AdamW"] = "AdamW"

    learning_rate: float
    weight_decay: float

class ModelArchitectures(StrEnum):
    MLP = "MLP"
    CNN = "CNN"

class MNISTDataset(BaseModel):
    dataset_name: Literal["MNIST"] = "MNIST"

    size: Literal["small", "full"]
    canary: Literal["gaussian_noise"] | None

class Parameters(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    batch_size: int
    initialization_scale: float | None
    log_frequency: int
    loss: MSELoss = Field(discriminator="loss_name")
    model: ModelArchitectures
    optimization_steps: int
    optimizer: AdamW = Field(discriminator="optim_name")
    dataset: MNISTDataset = Field(discriminator="dataset_name")
    seed: int
