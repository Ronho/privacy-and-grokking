from pydantic import BaseModel, ConfigDict

from privacy_and_grokking.config.optimizer import Optimizer
from privacy_and_grokking.config.scheduler import Scheduler
from privacy_and_grokking.datasets import DatasetConfig, MaskingConfig
from privacy_and_grokking.loss.loss import Loss
from privacy_and_grokking.loss.regularizer import SelfContainedTwoSampleRegularizer
from privacy_and_grokking.models import Model


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: Model
    seed: int
    batch_size: int
    loss: Loss
    regularizer: SelfContainedTwoSampleRegularizer | None = None
    optimizer: Optimizer
    scheduler: Scheduler
    dataset: DatasetConfig
    dataset_mask: MaskingConfig
    dataset_mask_idx: int = 0

    @property
    def name(self) -> str:
        return f"{self.model.name.upper()}_{self.dataset.name.upper()}_{self.dataset_mask.name.upper()}_{self.optimizer.name.upper()}_{self.loss.name.upper()}"

    @property
    def full_name(self) -> str:
        return f"{self.name}_{self.dataset_mask_idx}"
