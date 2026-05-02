from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from privacy_and_grokking.models.base import ModelConfig


class MLPBatchNorm(nn.Module):
    def __init__(self, input_dim: torch.Size, num_classes: int):
        super().__init__()
        input = int(torch.prod(torch.tensor(input_dim)).item())
        self.fc1 = nn.Linear(input, 200)
        self.bn1 = nn.BatchNorm1d(200)
        self.fc2 = nn.Linear(200, 200)
        self.bn2 = nn.BatchNorm1d(200)
        self.fc3 = nn.Linear(200, num_classes)

    def forward(self, input):
        y = torch.flatten(input, 1)
        y = self.fc1(y)
        y = self.bn1(y)
        y = F.relu(y)
        y = self.fc2(y)
        y = self.bn2(y)
        y = F.relu(y)
        y = self.fc3(y)
        return y

    @property
    def last_layer(self):
        return self.fc3


class MLPBatchNormConfig(ModelConfig):
    name: Literal["mlp_batchnorm"] = "mlp_batchnorm"

    def _create(
        self, input_dim: torch.Size, num_classes: int
    ) -> nn.Module:
        return MLPBatchNorm(input_dim, num_classes)
