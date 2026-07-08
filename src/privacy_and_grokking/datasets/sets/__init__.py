from typing import Annotated

from pydantic import Field

from privacy_and_grokking.datasets.sets.cifar10 import CIFAR10Config
from privacy_and_grokking.datasets.sets.etf import ETFConfig
from privacy_and_grokking.datasets.sets.mnist import MNISTConfig

Data = Annotated[
    MNISTConfig | CIFAR10Config | ETFConfig,
    Field(discriminator="name"),
]

__all__ = ["Data"]
