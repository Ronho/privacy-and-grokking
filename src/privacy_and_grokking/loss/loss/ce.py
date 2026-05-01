from typing import Annotated, Literal

import torch
from pydantic import BeforeValidator, PlainSerializer

from privacy_and_grokking.loss.loss.base import LossConfig, LossType


def _to_tensor(v: torch.Tensor | list[float] | None) -> torch.Tensor | None:
    """Convert a list of floats to a torch.Tensor for Pydantic deserialization."""
    if v is None:
        return None
    if isinstance(v, torch.Tensor):
        return v
    return torch.tensor(v, dtype=torch.float32)

def _serialize_tensor(v: torch.Tensor | None) -> list[float] | None:
    """Convert a torch.Tensor back to a list for Pydantic serialization."""
    if v is None:
        return None
    return v.tolist()

class CrossEntropyLossConfig(LossConfig):
    name: Literal["ce"] = "ce"
    weight: Annotated[
        torch.Tensor | None,
        BeforeValidator(_to_tensor),
        PlainSerializer(_serialize_tensor, return_type=list[float] | None),
    ] = None  # Dim (C)
    ignore_index: int = -100
    label_smoothing: float = 0.0

    def __call__(self) -> LossType:
        return torch.nn.CrossEntropyLoss(
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing
        )
