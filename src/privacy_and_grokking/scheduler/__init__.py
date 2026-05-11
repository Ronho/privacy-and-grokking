from typing import Annotated

from pydantic import Field

from privacy_and_grokking.scheduler.cosine_annealing import CosineAnnealingLRConfig
from privacy_and_grokking.scheduler.none import NoSchedulerConfig

Scheduler = Annotated[NoSchedulerConfig | CosineAnnealingLRConfig, Field(discriminator="name")]

__all__ = [
    "Scheduler",
]
