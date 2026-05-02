"""
Defines the base masking strategy for dataset samples across multiple models.

(Ideal) Criteria:
- Each data point should appear in approximately p * num_models models.
- Each model should have approximately p * num_samples data points.
- Each class should be evenly represented for each model.
- There should be randomness in the selection to avoid models being too similar.
- The masking strategy should be deterministic given a seed.
- The implementation should be efficient in both time and space.
"""

from abc import ABC, abstractmethod

import torch
from pydantic import BaseModel, Field

from privacy_and_grokking.utils import Logger


class Masking(ABC):
    def __init__(
        self,
        num_samples: int,
        num_classes: int,
        num_models: int,
        p: float,
        seed: int | None = None,
    ):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.num_models = num_models
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be between 0 and 1")
        self.p = p
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)
        else:
            Logger.get().warning(
                "No seed provided for masking, using non-deterministic behavior."
            )

    def __call__(self, classes: torch.Tensor | None = None) -> torch.Tensor:
        logger = Logger.get()
        if classes is None:
            logger.warning(
                "No classes provided for StratifiedMasking, using even distribution."
            )
            samples_per_class = self.num_samples // self.num_classes
            remainder = self.num_samples % self.num_classes
            classes = torch.repeat_interleave(
                torch.arange(self.num_classes), samples_per_class
            )
            if remainder > 0:
                classes = torch.cat([classes, torch.arange(end=remainder)])

        if len(classes) != self.num_samples:
            logger.error(
                "Length of classes does not match num_samples",
                {"classes": len(classes), "num_samples": self.num_samples},
            )
            raise ValueError("Length of classes must match num_samples")

        return self._generate_mask(classes)

    @abstractmethod
    def _generate_mask(self, classes: torch.Tensor) -> torch.Tensor: ...


class MaskingConfig(BaseModel):
    name: str
    num_models: int
    p: float = Field(ge=0, le=1)
    model_index: int = 0
    seed: int | None = None

    @abstractmethod
    def __call__(self, num_samples: int, num_classes: int) -> Masking: ...
