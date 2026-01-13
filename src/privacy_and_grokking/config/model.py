from pydantic import BaseModel, ConfigDict, Field
from typing import Literal
from ..datasets import Data, Canary
from ..models import Model

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

class CanaryConfig(BaseModel):
    name: Canary
    percentage: float
    repetitions: int

class GaussianNoiseCanary(CanaryConfig):
    name: Canary = "gaussian_noise"
    noise_scale: float | None = None
    seed: int | None = None

class WatermarkCanary(CanaryConfig):
    name: Canary = "watermark"
    square_size: int
    seed: int | None = None

class DatasetConfig(BaseModel):
    name: Data
    train_ratio: float = Field(gt=0.0, lt=1.0)
    use_val_for_training: bool = Field(default=False)
    train_size: int | None = None
    canary: CanaryConfig | None = Field(discriminator="name", default=None)

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
