from collections.abc import Callable
from typing import Annotated, Literal

import torch
from pydantic import BaseModel, Field

type LossType = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class MMDLogitRegularizerConfig(BaseModel):
    """Configuration for the differentiable MMD logit regularization term.

    Uses the kernel computation from ``ignite.metrics.MaximumMeanDiscrepancy``
    (pytorch-ignite) wrapped in a differentiable ``nn.Module``.

    Attributes:
        weight: Scalar λ that scales the MMD² penalty before it is added to
            the task loss.  Tune to balance privacy and accuracy.
        var: Kernel bandwidth σ² (matches Ignite's ``var`` parameter).
            ``None`` uses the median pairwise squared-distance heuristic
            computed with ``torch.no_grad`` each step.
        samples_per_class: Number of training samples per class to hold out
            as a proxy non-member reference set.  These samples are drawn
            once at training start and are not included in training batches.
    """

    weight: float = 0.1
    var: float | None = None
    samples_per_class: int = 3

    def build(self):
        """Instantiate and return a :class:`~privacy_and_grokking.losses.MMDLogitRegularizer`."""
        from privacy_and_grokking.losses import MMDLogitRegularizer

        return MMDLogitRegularizer(var=self.var)


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
