from typing import Annotated

from pydantic import Field

from privacy_and_grokking.datasets.canaries.base import Canary, CanaryConfig
from privacy_and_grokking.datasets.canaries.square_watermark import (
    SquareWatermarkCanary,
    SquareWatermarkCanaryConfig,
)
from privacy_and_grokking.datasets.canaries.uniform_noise import (
    UniformNoiseCanary,
    UniformNoiseCanaryConfig,
)
from privacy_and_grokking.datasets.canaries.gaussian_noise import (
    GaussianNoiseCanary,
    GaussianNoiseCanaryConfig,
)
from privacy_and_grokking.datasets.canaries.label_noise import (
    LabelNoiseCanary,
    LabelNoiseCanaryConfig,
)

from privacy_and_grokking.datasets.canaries.ood_natural import (
    OODNaturalCanary,
    OODNaturalCanaryConfig,
)

CanaryType = Annotated[
    UniformNoiseCanaryConfig
    | SquareWatermarkCanaryConfig
    | GaussianNoiseCanaryConfig
    | LabelNoiseCanaryConfig
    | OODNaturalCanaryConfig,
    Field(discriminator="name"),
]


def create_canary_generator(config: CanaryConfig, dim: tuple[int, ...]) -> Canary:
    return config(dim=dim)


__all__ = [
    "Canary",
    "CanaryConfig",
    "CanaryType",
    "SquareWatermarkCanary",
    "SquareWatermarkCanaryConfig",
    "UniformNoiseCanary",
    "UniformNoiseCanaryConfig",
    "GaussianNoiseCanary",
    "GaussianNoiseCanaryConfig",
    "LabelNoiseCanary",
    "LabelNoiseCanaryConfig",
    "OODNaturalCanary",
    "OODNaturalCanaryConfig",
    "create_canary_generator",
]
