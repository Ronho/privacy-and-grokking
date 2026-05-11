from typing import Annotated

from pydantic import Field

from .cnn import CNNConfig
from .mlp import MLPConfig
from .mlp_batchnorm import MLPBatchNormConfig
from .resnet import ResNetConfig
from .vgg import VGGConfig
from .vit import ViTConfig

Model = Annotated[
    MLPConfig | MLPBatchNormConfig | CNNConfig | ResNetConfig | VGGConfig | ViTConfig,
    Field(discriminator="name"),
]

__all__ = [
    "CNNConfig",
    "MLPBatchNormConfig",
    "MLPConfig",
    "Model",
    "ResNetConfig",
    "VGGConfig",
    "ViTConfig",
]
