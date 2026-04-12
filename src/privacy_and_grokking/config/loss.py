from collections.abc import Callable
from typing import Annotated, Literal

import torch
from pydantic import BaseModel, Field

type LossType = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class MSELossConfig(BaseModel):
    name: Literal["mse"] = "mse"

    def __call__(self, **kwargs) -> LossType:
        if "num_classes" not in kwargs:
            raise KeyError("`num_classes` required for MSELoss")
        num_classes = kwargs["num_classes"]
        one_hot = torch.eye(num_classes, num_classes)
        fn = torch.nn.MSELoss(reduction=kwargs.get("reduction", "mean"))

        def loss(logits, labels: torch.Tensor) -> torch.Tensor:
            return fn(logits, one_hot.to(labels.device)[labels])

        return loss


class CrossEntropyLossConfig(BaseModel):
    name: Literal["ce"] = "ce"

    def __call__(self, **kwargs) -> LossType:
        return torch.nn.CrossEntropyLoss(reduction=kwargs.get("reduction", "mean"))


Loss = Annotated[MSELossConfig | CrossEntropyLossConfig, Field(discriminator="name")]
