from abc import abstractmethod

import torch
import torch.nn as nn
from pydantic import BaseModel


class ModelConfig(BaseModel):
    name: str
    initialization_scale: float | None = None

    @abstractmethod
    def _create(self, input_dim: torch.Size, num_classes: int) -> nn.Module: ...

    def _apply_initialization_scale(self, model: nn.Module, scale: float) -> None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name.startswith("linear"):
                    param.data *= scale

    def __call__(self, input_dim: torch.Size, num_classes: int) -> nn.Module:
        model = self._create(input_dim, num_classes)
        if self.initialization_scale is not None:
            self._apply_initialization_scale(model)
        return model
