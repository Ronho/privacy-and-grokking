from typing import Literal

import torch
import torch.nn as nn

from privacy_and_grokking.models.base import ModelBase, ModelConfig

# Ref: https://github.com/keitaroskmt/collapse-dynamics/blob/master/src/models/toy_cnn.py


class CNNExtended(ModelBase):
    """An extended CNN model based on toy_cnn from collapse-dynamics."""

    def __init__(
        self,
        input_dim: torch.Size,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        c, h, w = input_dim
        self.activation = nn.ReLU()

        num_channel_scale = 1 if c == 1 else 2

        self.conv1 = nn.Conv2d(
            c,
            16 * num_channel_scale,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            16 * num_channel_scale,
            32 * num_channel_scale,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        out_h = h // 2
        out_w = w // 2

        self.linear1 = nn.Linear(32 * num_channel_scale * out_h * out_w, 128)
        self.linear2 = nn.Linear(128, num_classes)

    def forward(
        self, x: torch.Tensor, verbose: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        z = self.activation(self.linear1(x))
        out = self.linear2(z)
        if verbose:
            return out, z
        return out

    def classifier(self) -> nn.Module:
        return self.linear2


class CNNExtendedConfig(ModelConfig):
    name: Literal["cnn-extended"] = "cnn-extended"

    def _create(self, input_dim: torch.Size, num_classes: int) -> nn.Module:
        return CNNExtended(input_dim, num_classes)
