import torch
from torch import Tensor, nn
from typing import Literal

from privacy_and_grokking.models.base import ModelConfig

# Ref: https://github.com/keitaroskmt/collapse-dynamics/blob/master/src/models/resnet.py


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: Tensor) -> Tensor:
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return nn.functional.relu(out)


class ResNet(nn.Module):
    """ResNet model."""

    def __init__(
        self,
        input_dim: torch.Size,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        c, h, w = input_dim

        # ResNet18
        block = BasicBlock
        dim_out = 512
        num_blocks = [2, 2, 2, 2]

        self.init_in_planes = 64
        self.in_planes = self.init_in_planes

        self.conv1 = nn.Conv2d(
            c,
            self.in_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(
            block,
            self.init_in_planes,
            num_blocks[0],
            stride=1,
        )
        self.layer2 = self._make_layer(
            block,
            self.init_in_planes * 2,
            num_blocks[1],
            stride=2,
        )
        self.layer3 = self._make_layer(
            block,
            self.init_in_planes * 4,
            num_blocks[2],
            stride=2,
        )
        self.layer4 = self._make_layer(
            block,
            self.init_in_planes * 8,
            num_blocks[3],
            stride=2,
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(dim_out, num_classes)

    def _make_layer(
        self,
        block: BasicBlock,
        planes: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride_ in strides:
            layers.append(block(self.in_planes, planes, stride_))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(
        self,
        x: Tensor,
        verbose: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        z = torch.flatten(x, 1)
        out = self.linear(z)
        if verbose:
            return out, z
        return out

    def classifier(self) -> nn.Module:
        return self.linear


class ResNetConfig(ModelConfig):
    name: Literal["resnet"] = "resnet"

    def _create(self, input_dim: torch.Size, num_classes: int) -> nn.Module:
        return ResNet(input_dim, num_classes)
