"""Standard CIFAR-style ResNet (He et al., 2015).

This is the small ResNet variant designed specifically for CIFAR (32x32 inputs),
not the larger ImageNet ResNet. The total depth is ``6n + 2`` where ``n`` is the
number of basic blocks per stage. Default ``n=3`` yields ResNet-20, which is the
standard "not overkill" baseline for CIFAR-10.
"""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from privacy_and_grokking.models.base import ModelConfig


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    planes * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * self.expansion),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    def __init__(
        self,
        input_dim: torch.Size,
        num_classes: int,
        num_blocks_per_stage: int = 3,
        base_width: int = 16,
    ):
        super().__init__()
        c, _, _ = input_dim
        self.in_planes = base_width

        self.conv1 = nn.Conv2d(c, base_width, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_width)

        self.layer1 = self._make_stage(base_width, num_blocks_per_stage, stride=1)
        self.layer2 = self._make_stage(base_width * 2, num_blocks_per_stage, stride=2)
        self.layer3 = self._make_stage(base_width * 4, num_blocks_per_stage, stride=2)

        self.fc = nn.Linear(base_width * 4 * BasicBlock.expansion, num_classes)

        # Standard He init for conv layers, used in the original CIFAR ResNet.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _make_stage(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = torch.flatten(out, 1)
        return self.fc(out)

    @property
    def last_layer(self):
        return self.fc


class ResNetConfig(ModelConfig):
    name: Literal["resnet"] = "resnet"
    # Depth = 6 * num_blocks_per_stage + 2. Default => ResNet-20.
    num_blocks_per_stage: int = 3
    base_width: int = 16

    def _create(self, input_dim: torch.Size, num_classes: int) -> nn.Module:
        return ResNet(
            input_dim,
            num_classes,
            num_blocks_per_stage=self.num_blocks_per_stage,
            base_width=self.base_width,
        )
