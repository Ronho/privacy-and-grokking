from typing import Annotated

from pydantic import Field

from .cnn import CNNConfig
from .mlp import MLPConfig
from .mlp_batchnorm import MLPBatchNormConfig
from .mlp_extended import MLPExtendedConfig
from .resnet import ResNetConfig
from .vgg import VGGConfig
from .vit import ViTConfig

Model = Annotated[
    MLPConfig | MLPBatchNormConfig | MLPExtendedConfig | CNNConfig | ResNetConfig | VGGConfig | ViTConfig,
    Field(discriminator="name"),
]

__all__ = [
    "CNNConfig",
    "MLPBatchNormConfig",
    "MLPConfig",
    "MLPExtendedConfig",
    "Model",
    "ResNetConfig",
    "VGGConfig",
    "ViTConfig",
]
