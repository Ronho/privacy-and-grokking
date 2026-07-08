from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from privacy_and_grokking.models.base import ModelConfig


class MLPNC(nn.Module):
    def __init__(self, input_dim: torch.Size, num_classes: int):
        super().__init__()
        input = int(torch.prod(torch.tensor(input_dim)).item())
        self.fc1 = nn.Linear(input, 200)
        self.fc2 = nn.Linear(200, 200)
        # bias=False for clean Neural Collapse
        self.fc3 = nn.Linear(200, num_classes, bias=False)

    def forward(
        self, input, verbose: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        y = torch.flatten(input, 1)
        y = self.fc1(y)
        y = F.relu(y)
        # Penultimate layer without ReLU
        z = self.fc2(y)
        y = self.fc3(z)
        if verbose:
            return y, z
        return y

    def classifier(self) -> nn.Module:
        return self.fc3


class MLPNCConfig(ModelConfig):
    name: Literal["mlp-nc"] = "mlp-nc"

    def _create(self, input_dim: torch.Size, num_classes: int) -> nn.Module:
        return MLPNC(input_dim, num_classes)
