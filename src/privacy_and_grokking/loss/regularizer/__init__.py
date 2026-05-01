from typing import Annotated

from pydantic import Field

from privacy_and_grokking.loss.regularizer.mmd import MMDRegularizerConfig
from privacy_and_grokking.loss.regularizer.overlap import OverlapRegularizerConfig
from privacy_and_grokking.loss.regularizer.overlap_adaptive import OverlapAdaptiveRegularizerConfig
from privacy_and_grokking.loss.regularizer.overlap_kde import OverlapKDERegularizerConfig
from privacy_and_grokking.loss.regularizer.per_sample_distance import (
    PerSampleDistanceRegularizerConfig,
)

SelfContainedTwoSampleRegularizer = Annotated[
    OverlapRegularizerConfig
    | OverlapAdaptiveRegularizerConfig
    | OverlapKDERegularizerConfig
    | PerSampleDistanceRegularizerConfig
    | MMDRegularizerConfig,
    Field(discriminator="name"),
]

__all__ = [
    "SelfContainedTwoSampleRegularizer",
]
