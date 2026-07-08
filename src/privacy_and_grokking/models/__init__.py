from typing import Annotated

from pydantic import Field

from .cnn import CNNConfig
from .cnn_extended import CNNExtendedConfig
from .mlp import MLPConfig
from .mlp_batchnorm import MLPBatchNormConfig
from .mlp_extended import MLPExtendedConfig
from .mlp_nc import MLPNCConfig
from .resnet import ResNetConfig
from .vgg import VGGConfig
from .vit import ViTConfig

Model = Annotated[
    MLPConfig | MLPBatchNormConfig | MLPExtendedConfig | MLPNCConfig | CNNConfig | CNNExtendedConfig | ResNetConfig | VGGConfig | ViTConfig,
    Field(discriminator="name"),
]

__all__ = [
    "CNNConfig",
    "CNNExtendedConfig",
    "MLPBatchNormConfig",
    "MLPConfig",
    "MLPExtendedConfig",
    "MLPNCConfig",
    "Model",
    "ResNetConfig",
    "VGGConfig",
    "ViTConfig",
]
