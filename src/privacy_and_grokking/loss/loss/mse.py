from typing import Literal

import torch

from privacy_and_grokking.loss.loss.base import LossConfig, LossType


class MSELossConfig(LossConfig):
    name: Literal["mse"] = "mse"

    def __call__(self, **kwargs) -> LossType:
        num_classes: int | None = kwargs.get("num_classes")
        if num_classes is None:
            raise ValueError(
                "num_classes must be provided as a keyword argument to MSELossConfig"
            )
        one_hot = torch.eye(num_classes, num_classes)
        fn = torch.nn.MSELoss(reduction=self.reduction)

        def loss(logits, labels: torch.Tensor) -> torch.Tensor:
            return fn(logits, one_hot.to(labels.device)[labels])

        return loss
