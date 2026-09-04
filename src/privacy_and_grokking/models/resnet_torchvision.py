from typing import Literal

import torch
import torchvision.models as models
from torch import Tensor, nn

from privacy_and_grokking.models.base import ModelBase, ModelConfig


class TorchvisionResNet(ModelBase):
    """Wrapper for torchvision's ResNet18 modified for MNIST/CIFAR-sized images."""

    def __init__(self, input_dim: torch.Size, num_classes: int = 10) -> None:
        super().__init__()
        c, h, w = input_dim
        self.model = models.resnet18(pretrained=False, num_classes=num_classes)
        # Small dataset filter size used by He et al. (2015)
        self.model.conv1 = nn.Conv2d(c, self.model.conv1.weight.shape[0], 3, 1, 1, bias=False)
        self.model.maxpool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)

    def forward(
        self,
        x: Tensor,
        verbose: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        z = torch.flatten(x, 1)
        out = self.model.fc(z)

        if verbose:
            return out, z
        return out

    def classifier(self) -> nn.Module:
        return self.model.fc


class ResNetTorchvisionConfig(ModelConfig):
    name: Literal["resnet_torchvision"] = "resnet_torchvision"

    def _create(self, input_dim: torch.Size, num_classes: int) -> nn.Module:
        return TorchvisionResNet(input_dim, num_classes)
