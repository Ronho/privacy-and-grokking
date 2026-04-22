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


class OverlapRegularizerConfig(BaseModel):
    name: Literal["overlap"] = "overlap"
    weight: float = 0.1
    n_bins: int = 50
    sigma: float = 0.05

    def __call__(self) -> "torch.nn.Module":
        from privacy_and_grokking.losses import OverlapRegularizer

        return OverlapRegularizer(n_bins=self.n_bins, sigma=self.sigma)


class OverlapAdaptiveRegularizerConfig(BaseModel):
    name: Literal["overlap_adaptive"] = "overlap_adaptive"
    weight: float = 0.1
    max_bins: int = 100
    sigma: float = 0.05

    def __call__(self) -> "torch.nn.Module":
        from privacy_and_grokking.losses import OverlapAdaptiveRegularizer

        return OverlapAdaptiveRegularizer(max_bins=self.max_bins, sigma=self.sigma)


class OverlapKDERegularizerConfig(BaseModel):
    name: Literal["overlap_kde"] = "overlap_kde"
    weight: float = 0.1
    n_points: int = 200

    def __call__(self) -> "torch.nn.Module":
        from privacy_and_grokking.losses import OverlapKDERegularizer

        return OverlapKDERegularizer(n_points=self.n_points)


class MMDRegularizerConfig(BaseModel):
    name: Literal["mmd"] = "mmd"
    weight: float = 0.1
    bandwidth: float = 0.1

    def __call__(self) -> "torch.nn.Module":
        from privacy_and_grokking.losses import MMDRegularizer

        return MMDRegularizer(bandwidth=self.bandwidth)


Regularizer = Annotated[
    OverlapRegularizerConfig
    | OverlapAdaptiveRegularizerConfig
    | OverlapKDERegularizerConfig
    | MMDRegularizerConfig,
    Field(discriminator="name"),
]
