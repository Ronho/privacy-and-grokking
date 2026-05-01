from collections.abc import Callable
from typing import Annotated, Literal

import torch
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, PlainSerializer


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

type LossType = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

class LossConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    reduction: Literal["none", "mean", "sum"] = "mean"

class MSELossConfig(LossConfig):
    name: Literal["mse"] = "mse"
    num_classes: int

    def __call__(self) -> LossType:
        one_hot = torch.eye(self.num_classes, self.num_classes)
        fn = torch.nn.MSELoss(reduction=self.reduction)

        def loss(logits, labels: torch.Tensor) -> torch.Tensor:
            return fn(logits, one_hot.to(labels.device)[labels])

        return loss


class CrossEntropyLossConfig(LossConfig):
    name: Literal["ce"] = "ce"
    weight: Annotated[
        torch.Tensor | None,
        BeforeValidator(_to_tensor),
        PlainSerializer(_serialize_tensor, return_type=list[float] | None),
    ] = None  # Dim (C)
    ignore_index: int = -100
    label_smoothing: float = 0.0

    def __call__(self, **kwargs) -> LossType:
        return torch.nn.CrossEntropyLoss(
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing
        )


Loss = Annotated[MSELossConfig | CrossEntropyLossConfig, Field(discriminator="name")]
