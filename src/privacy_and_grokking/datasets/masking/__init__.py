from typing import Annotated

from pydantic import Field

from privacy_and_grokking.datasets.masking.balanced_stratified import (
    BalancedStratifiedMaskingConfig,
)
from privacy_and_grokking.datasets.masking.independent_stratified import (
    IndependentStratifiedMaskingConfig,
)
from privacy_and_grokking.datasets.masking.paired_stratified import (
    PairedStratifiedMaskingConfig,
)
from privacy_and_grokking.datasets.masking.partitioned_stratified import (
    PartitionedStratifiedMaskingConfig,
)
from privacy_and_grokking.datasets.masking.uniform import UniformMaskingConfig

Mask = Annotated[
    UniformMaskingConfig
    | IndependentStratifiedMaskingConfig
    | PartitionedStratifiedMaskingConfig
    | BalancedStratifiedMaskingConfig
    | PairedStratifiedMaskingConfig,
    Field(discriminator="name"),
]


__all__ = [
    "Mask",
]
