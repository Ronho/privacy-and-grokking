from pydantic import BaseModel, ConfigDict

from privacy_and_grokking.datasets import DatasetConfig
from privacy_and_grokking.loss import Loss, SelfContainedTwoSampleRegularizer
from privacy_and_grokking.metrics import MetricsConfig
from privacy_and_grokking.models import Model
from privacy_and_grokking.optimizer import Optimizer
from privacy_and_grokking.scheduler import Scheduler


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: Model
    seed: int
    batch_size: int
    loss: Loss
    regularizer: SelfContainedTwoSampleRegularizer | None = None
    optimizer: Optimizer
    scheduler: Scheduler
    data: DatasetConfig
    metrics: MetricsConfig = MetricsConfig()

    @property
    def name(self) -> str:
        parts = [
            self.data.data.name.upper(),
            self.optimizer.name.upper(),
            self.loss.name.upper(),
        ]
        if self.data.mask is not None:
            parts.insert(1, self.data.mask.name.upper())
        return f"{self.model.name.upper()}_{'_'.join(parts)}"

    @property
    def full_name(self) -> str:
        if self.data.mask is not None:
            return f"{self.name}_{self.data.mask.model_index}"
        return self.name
