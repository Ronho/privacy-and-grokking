from collections.abc import Sequence
from typing import Literal

import torch

from privacy_and_grokking.datasets.canaries.base import Canary, CanaryConfig


class LabelNoiseCanary:
    def __init__(self, dim: Sequence[int]):
        self.dim = dim

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        # Label noise just changes the label, leaving the image unchanged.
        # The label derangement is already handled in base.py CanaryDataset.
        return image


class LabelNoiseCanaryConfig(CanaryConfig):
    name: Literal["label_noise"] = "label_noise"

    def __call__(self, dim: Sequence[int]) -> Canary:
        return LabelNoiseCanary(dim=dim)
