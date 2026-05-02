from pydantic import BaseModel, ConfigDict

from privacy_and_grokking.datasets import DataConfig
from privacy_and_grokking.evaluation import MetricsConfig
from privacy_and_grokking.loss import Loss, SelfContainedTwoSampleRegularizer
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
    data: DataConfig
    metrics: MetricsConfig = MetricsConfig()

    @property
    def name(self) -> str:
        return (
            f"{self.model.name.upper()}_{self.data.name.upper()}_"
            f"{self.data.mask.name.upper()}_{self.optimizer.name.upper()}_"
            f"{self.loss.name.upper()}"
        )

    @property
    def full_name(self) -> str:
        return f"{self.name}_{self.data.mask.model_index}"
