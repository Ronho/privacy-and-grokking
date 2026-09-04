from math import floor
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from privacy_and_grokking.models.base import ModelBase, ModelConfig


class CNN(ModelBase):
    def __init__(self, input_dim: torch.Size, num_classes: int):
        super().__init__()
        c, h, w = input_dim
        CHANNEL_DIM = 32
        self.conv1 = nn.Conv2d(c, CHANNEL_DIM, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(CHANNEL_DIM, CHANNEL_DIM * 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(CHANNEL_DIM * 2, CHANNEL_DIM * 2, kernel_size=3, padding=1)

        conv_output_dim = CHANNEL_DIM * 2 * floor(h / 8) * floor(w / 8)

        self.fc1 = nn.Linear(conv_output_dim, 200)
        self.fc2 = nn.Linear(200, num_classes)

    def forward(
        self, input, verbose: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # Block 1
        y = F.relu(self.conv1(input))
        y = F.max_pool2d(y, 2, 2)

        # Block 2
        y = F.relu(self.conv2(y))
        y = F.max_pool2d(y, 2, 2)

        # Block 3
        y = F.relu(self.conv3(y))
        y = F.max_pool2d(y, 2, 2)

        # Block 4
        y = torch.flatten(y, 1)
        y = F.relu(self.fc1(y))

        z = y
        y = self.fc2(z)
        if verbose:
            return y, z
        return y

    def classifier(self) -> nn.Module:
        return self.fc2


class CNNConfig(ModelConfig):
    name: Literal["cnn"] = "cnn"

    def _create(self, input_dim: torch.Size, num_classes: int) -> nn.Module:
        return CNN(input_dim, num_classes)
