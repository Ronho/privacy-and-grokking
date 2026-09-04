"""CIFAR-style VGG.

The original VGG was sized for 224x224 ImageNet inputs and ends in three huge
fully connected layers. For 32x32 CIFAR-10 that is wildly oversized, so we use
the common "CIFAR VGG" variant: same convolutional pattern as the reference
configurations (A/B/D/E from Simonyan & Zisserman, 2014) with BatchNorm, but
the classifier is a single linear layer applied to the global-average-pooled
feature map. Default ``cfg='A'`` yields VGG-11.
"""

from typing import Literal

import torch
import torch.nn as nn

from privacy_and_grokking.models.base import ModelBase, ModelConfig

# Numbers are output channels for conv layers. "M" denotes a 2x2 max-pool.
_VGG_CFGS: dict[str, list[int | str]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],  # VGG-11
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],  # VGG-13
    "D": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],  # VGG-16
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],  # VGG-19
}


def _make_layers(cfg: list[int | str], in_channels: int, batch_norm: bool) -> nn.Sequential:
    layers: list[nn.Module] = []
    c = in_channels
    for v in cfg:
        if v == "M":
            # ceil_mode keeps small feature maps from collapsing to 0x0 on
            # inputs like MNIST 28x28, where 5 successive halvings would
            # otherwise underflow.
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
        else:
            assert isinstance(v, int)
            layers.append(nn.Conv2d(c, v, kernel_size=3, padding=1, bias=not batch_norm))
            if batch_norm:
                layers.append(nn.BatchNorm2d(v))
            layers.append(nn.ReLU(inplace=True))
            c = v
    return nn.Sequential(*layers)


class VGG(ModelBase):
    def __init__(
        self,
        input_dim: torch.Size,
        num_classes: int,
        cfg: str = "A",
        batch_norm: bool = True,
    ):
        super().__init__()
        if cfg not in _VGG_CFGS:
            raise ValueError(f"Unknown VGG cfg '{cfg}'. Valid: {list(_VGG_CFGS)}.")
        c, _, _ = input_dim
        self.features = _make_layers(_VGG_CFGS[cfg], in_channels=c, batch_norm=batch_norm)
        # All standard VGG cfgs end with a 512-channel block.
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor, verbose: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        out = self.features(x)
        out = torch.nn.functional.adaptive_avg_pool2d(out, 1)
        z = torch.flatten(out, 1)
        out = self.fc(z)
        if verbose:
            return out, z
        return out

    def classifier(self) -> nn.Module:
        return self.fc


class VGGConfig(ModelConfig):
    name: Literal["vgg"] = "vgg"
    cfg: Literal["A", "B", "D", "E"] = "A"  # VGG-11 by default
    batch_norm: bool = True

    def _create(self, input_dim: torch.Size, num_classes: int) -> nn.Module:
        return VGG(input_dim, num_classes, cfg=self.cfg, batch_norm=self.batch_norm)
