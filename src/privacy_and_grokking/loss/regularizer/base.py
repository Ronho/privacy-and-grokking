from abc import abstractmethod
from collections.abc import Callable

import torch
from pydantic import BaseModel

from privacy_and_grokking.loss.regularizer_source import NoisyRegularizerSource

type RegularizerType = Callable[[torch.Tensor], torch.Tensor]


class RegularizerConfig(BaseModel):
    name: str

class SelfContainedTwoSampleRegularizerConfig(RegularizerConfig):
    source: NoisyRegularizerSource

    @abstractmethod
    def __call__(self) -> RegularizerType: ...
