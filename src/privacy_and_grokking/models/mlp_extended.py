from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from privacy_and_grokking.models.base import ModelBase, ModelConfig

# Ref: https://github.com/keitaroskmt/collapse-dynamics/blob/master/src/models/toy_mlp.py


class MLPExtended(ModelBase):
    def __init__(self, input_dim: torch.Size, num_classes: int, alpha: float | None = None):
        super().__init__()
        self.alpha = alpha
        input = int(torch.prod(torch.tensor(input_dim)).item())
        self.fc1 = nn.Linear(input, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 200)
        self.fc4 = nn.Linear(200, num_classes)

    def forward(
        self, input, verbose: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        y = torch.flatten(input, 1)
        y = self.fc1(y)
        y = F.relu(y)
        y = self.fc2(y)
        y = F.relu(y)
        y = self.fc3(y)
        z = F.relu(y)
        y = self.fc4(z)
        if self.alpha is not None:
            y = y * self.alpha
        if verbose:
            return y, z
        return y

    def classifier(self) -> nn.Module:
        return self.fc4


class MLPExtendedConfig(ModelConfig):
    name: Literal["mlp-extended"] = "mlp-extended"
    alpha: float | None = None  # cf. GROKKING AS THE TRANSITION FROM LAZY TO RICH TRAINING DYNAMICS

    def _create(self, input_dim: torch.Size, num_classes: int) -> nn.Module:
        return MLPExtended(input_dim, num_classes, alpha=self.alpha)
