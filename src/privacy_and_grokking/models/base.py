from abc import abstractmethod

import torch
import torch.nn as nn
from pydantic import BaseModel


class ModelConfig(BaseModel):
    name: str
    initialization_scale: float | None = None

    @abstractmethod
    def _create(
        self, input_dim: torch.Size, num_classes: int
    ) -> nn.Module: ...

    def __call__(
        self, input_dim: torch.Size, num_classes: int
    ) -> nn.Module:
        model = self._create(input_dim, num_classes)
        if self.initialization_scale is not None:
            with torch.no_grad():
                for p in model.parameters():
                    p.data *= self.initialization_scale
        return model
