from typing import Annotated

from pydantic import Field

from privacy_and_grokking.optimizer.adam import AdamConfig
from privacy_and_grokking.optimizer.adamw import AdamWConfig
from privacy_and_grokking.optimizer.rmsprop import RMSpropConfig
from privacy_and_grokking.optimizer.sgd import SGDConfig

Optimizer = Annotated[
    AdamConfig | AdamWConfig | RMSpropConfig | SGDConfig, Field(discriminator="name")
]

__all__ = [
    "Optimizer",
]
