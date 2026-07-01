from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from privacy_and_grokking.models.base import ModelConfig


class MLP(nn.Module):
    def __init__(self, input_dim: torch.Size, num_classes: int):
        super().__init__()
        input = int(torch.prod(torch.tensor(input_dim)).item())
        self.fc1 = nn.Linear(input, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, num_classes)

    def forward(
        self, input, verbose: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        y = torch.flatten(input, 1)
        y = self.fc1(y)
        y = F.relu(y)
        y = self.fc2(y)
        z = F.relu(y)
        y = self.fc3(z)
        if verbose:
            return y, z
        return y

    def classifier(self) -> nn.Module:
        return self.fc3


class MLPConfig(ModelConfig):
    name: Literal["mlp"] = "mlp"

    def _create(self, input_dim: torch.Size, num_classes: int) -> nn.Module:
        return MLP(input_dim, num_classes)
