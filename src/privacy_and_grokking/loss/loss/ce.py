from typing import Literal

import torch

from privacy_and_grokking.loss.loss.base import LossConfig, LossType


class CrossEntropyLossConfig(LossConfig):
    name: Literal["cross_entropy"] = "cross_entropy"
    weight: list[float] | None = None  # Per-class weights, dim (C)
    ignore_index: int = -100
    label_smoothing: float = 0.0

    def __call__(self, **kwargs) -> LossType:
        w = torch.tensor(self.weight, dtype=torch.float32) if self.weight is not None else None
        return torch.nn.CrossEntropyLoss(
            weight=w,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )
