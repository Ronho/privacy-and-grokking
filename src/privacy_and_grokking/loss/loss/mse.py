from typing import Literal

import torch

from privacy_and_grokking.loss.loss.base import LossConfig, LossType


class MSELossConfig(LossConfig):
    name: Literal["mse"] = "mse"
    num_classes: int

    def __call__(self) -> LossType:
        one_hot = torch.eye(self.num_classes, self.num_classes)
        fn = torch.nn.MSELoss(reduction=self.reduction)

        def loss(logits, labels: torch.Tensor) -> torch.Tensor:
            return fn(logits, one_hot.to(labels.device)[labels])

        return loss
