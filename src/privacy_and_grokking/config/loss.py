from collections.abc import Callable
from enum import StrEnum
from typing import Annotated, Literal, Self

import torch
from pydantic import BaseModel, Field, model_validator

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


class ValidationSource(StrEnum):
    TEST_SET = "test_set"
    NOISY_SELF = "noisy_self"


class NoiseType(StrEnum):
    GAUSSIAN = "gaussian"
    SALT_AND_PEPPER = "salt_and_pepper"


class RegularizerConfigBase(BaseModel):
    validation_source: ValidationSource = ValidationSource.TEST_SET
    noise_type: NoiseType | None = None
    noise_std: float | None = None
    noise_fraction: float | None = None
    num_noisy_samples: int = 1
    loss_reduction: Literal["mean", "max"] | None = "mean"

    # Subclasses that support loss_reduction=None should override this.
    _supports_none_reduction: bool = False

    @model_validator(mode="after")
    def validate_noise_fields(self) -> Self:
        if self.num_noisy_samples < 1:
            raise ValueError("num_noisy_samples must be >= 1")
        if self.loss_reduction is None and not self._supports_none_reduction:
            raise ValueError(
                f"loss_reduction=None is not supported by {type(self).__name__}; "
                "use 'mean' or 'max'"
            )
        if self.validation_source == ValidationSource.TEST_SET:
            if self.noise_type is not None:
                raise ValueError(
                    "noise_type must not be set when validation_source is 'test_set'"
                )
            if self.noise_std is not None:
                raise ValueError(
                    "noise_std must not be set when validation_source is 'test_set'"
                )
            if self.noise_fraction is not None:
                raise ValueError(
                    "noise_fraction must not be set when validation_source is 'test_set'"
                )
            if self.num_noisy_samples != 1:
                raise ValueError(
                    "num_noisy_samples must be 1 when validation_source is 'test_set'"
                )
        elif self.validation_source == ValidationSource.NOISY_SELF:
            if self.noise_type is None:
                raise ValueError(
                    "noise_type is required when validation_source is 'noisy_self'"
                )
            if self.noise_type == NoiseType.GAUSSIAN:
                if self.noise_std is None:
                    raise ValueError(
                        "noise_std is required when noise_type is 'gaussian'"
                    )
                if self.noise_fraction is not None:
                    raise ValueError(
                        "noise_fraction must not be set when noise_type is 'gaussian'"
                    )
            elif self.noise_type == NoiseType.SALT_AND_PEPPER:
                if self.noise_fraction is None:
                    raise ValueError(
                        "noise_fraction is required when noise_type is 'salt_and_pepper'"
                    )
                if self.noise_fraction <= 0 or self.noise_fraction >= 1:
                    raise ValueError(
                        "noise_fraction must be in the range (0, 1)"
                    )
                if self.noise_std is not None:
                    raise ValueError(
                        "noise_std must not be set when noise_type is 'salt_and_pepper'"
                    )
        return self

    def create_noise_generator(self) -> "NoiseGenerator | None":
        """Factory method: returns a NoiseGenerator if noisy_self, else None."""
        if self.validation_source != ValidationSource.NOISY_SELF:
            return None
        from privacy_and_grokking.noise import GaussianNoise, SaltAndPepperNoise

        if self.noise_type == NoiseType.GAUSSIAN:
            return GaussianNoise(std=self.noise_std)
        elif self.noise_type == NoiseType.SALT_AND_PEPPER:
            return SaltAndPepperNoise(fraction=self.noise_fraction)
        raise ValueError(f"Unknown noise_type: {self.noise_type}")


class OverlapRegularizerConfig(RegularizerConfigBase):
    name: Literal["overlap"] = "overlap"
    weight: float = 0.1
    n_bins: int = 50
    sigma: float = 0.05

    def __call__(self) -> "torch.nn.Module":
        from privacy_and_grokking.losses import OverlapRegularizer

        return OverlapRegularizer(n_bins=self.n_bins, sigma=self.sigma)


class OverlapAdaptiveRegularizerConfig(RegularizerConfigBase):
    name: Literal["overlap_adaptive"] = "overlap_adaptive"
    weight: float = 0.1
    max_bins: int = 100
    sigma: float = 0.05

    def __call__(self) -> "torch.nn.Module":
        from privacy_and_grokking.losses import OverlapAdaptiveRegularizer

        return OverlapAdaptiveRegularizer(max_bins=self.max_bins, sigma=self.sigma)


class OverlapKDERegularizerConfig(RegularizerConfigBase):
    name: Literal["overlap_kde"] = "overlap_kde"
    weight: float = 0.1
    n_points: int = 200

    def __call__(self) -> "torch.nn.Module":
        from privacy_and_grokking.losses import OverlapKDERegularizer

        return OverlapKDERegularizer(n_points=self.n_points)


class MMDRegularizerConfig(RegularizerConfigBase):
    name: Literal["mmd"] = "mmd"
    weight: float = 0.1
    bandwidth: float = 0.1

    def __call__(self) -> "torch.nn.Module":
        from privacy_and_grokking.losses import MMDRegularizer

        return MMDRegularizer(bandwidth=self.bandwidth)


class PerSampleDistanceRegularizerConfig(RegularizerConfigBase):
    name: Literal["per_sample_distance"] = "per_sample_distance"
    weight: float = 0.1
    metric: Literal["l1", "l2", "huber"] = "l1"
    huber_delta: float = 1.0

    _supports_none_reduction: bool = True

    def __call__(self) -> "torch.nn.Module":
        from privacy_and_grokking.losses import PerSampleDistanceRegularizer

        return PerSampleDistanceRegularizer(
            metric=self.metric, huber_delta=self.huber_delta
        )


Regularizer = Annotated[
    OverlapRegularizerConfig
    | OverlapAdaptiveRegularizerConfig
    | OverlapKDERegularizerConfig
    | MMDRegularizerConfig
    | PerSampleDistanceRegularizerConfig,
    Field(discriminator="name"),
]
