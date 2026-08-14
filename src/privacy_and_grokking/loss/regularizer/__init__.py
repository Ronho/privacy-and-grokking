from typing import Annotated

from pydantic import Field

from privacy_and_grokking.loss.regularizer.mmd import MMDRegularizerConfig
from privacy_and_grokking.loss.regularizer.per_sample_distance import (
    PerSampleDistanceRegularizerConfig,
)

SelfContainedTwoSampleRegularizer = Annotated[
    PerSampleDistanceRegularizerConfig | MMDRegularizerConfig,
    Field(discriminator="name"),
]

__all__ = [
    "SelfContainedTwoSampleRegularizer",
]
