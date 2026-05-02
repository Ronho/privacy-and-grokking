from abc import abstractmethod
from collections.abc import Callable

import torch
from pydantic import BaseModel

from privacy_and_grokking.loss.regularizer_source import NoisyRegularizerSource

type RegularizerType = Callable[[torch.Tensor], torch.Tensor]


class RegularizerConfig(BaseModel):
    name: str


class SelfContainedTwoSampleRegularizerConfig(RegularizerConfig):
    """Base config for two-sample regularizers.

    The returned callable already includes the weight scaling, so the
    training loop can simply add the result to the task loss.
    """

    source: NoisyRegularizerSource
    weight: float = 0.1

    @abstractmethod
    def _make_regularizer(self) -> RegularizerType:
        """Subclasses implement the raw (unweighted) regularizer here."""
        ...

    def __call__(self) -> RegularizerType:
        """Return a weighted regularizer function."""
        raw_fn = self._make_regularizer()
        w = self.weight

        def weighted_regularizer(train_losses: torch.Tensor) -> torch.Tensor:
            return w * raw_fn(train_losses)

        return weighted_regularizer
