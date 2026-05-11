from typing import Annotated

from pydantic import Field

from privacy_and_grokking.loss.regularizer_source.gaussian import GaussianNoiseConfig
from privacy_and_grokking.loss.regularizer_source.salt_and_pepper import SaltAndPepperNoiseConfig

NoisyRegularizerSource = Annotated[
    SaltAndPepperNoiseConfig | GaussianNoiseConfig, Field(discriminator="name")
]

__all__ = [
    "NoisyRegularizerSource",
]
