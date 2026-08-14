from privacy_and_grokking.models.base import ModelBase
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


class ResNet(ModelBase):
    """ResNet model."""

    def __init__(
        self,
        input_dim: torch.Size,
        num_classes: int = 10,
        zero_init_residual: bool = False,
    ) -> None:
        super().__init__()
        c, h, w = input_dim

        # ResNet18
        block = BasicBlock
        layers = [2, 2, 2, 2]

        self.inplanes = 64
        self.conv1 = nn.Conv2d(c, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
        
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block: type[BasicBlock],
        planes: int,
        blocks: int,
        stride: int,
    ) -> nn.Sequential:
        layers = []
        layers.append(
            block(self.inplanes, planes, stride)
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, stride=1)
            )

        return nn.Sequential(*layers)

    def forward(
        self,
        x: Tensor,
        verbose: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        z = torch.flatten(x, 1)
        out = self.fc(z)
        
        if verbose:
            return out, z
        return out

    def classifier(self) -> nn.Module:
        return self.fc


class ResNetConfig(ModelConfig):
    name: Literal["resnet"] = "resnet"

    def _create(self, input_dim: torch.Size, num_classes: int) -> nn.Module:
        return ResNet(input_dim, num_classes)
