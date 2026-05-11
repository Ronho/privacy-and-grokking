from typing import Annotated

from pydantic import Field

from .cnn import CNNConfig
from .mlp import MLPConfig
from .mlp_batchnorm import MLPBatchNormConfig
from .resnet import ResNetConfig

Model = Annotated[
    MLPConfig | MLPBatchNormConfig | CNNConfig | ResNetConfig,
    Field(discriminator="name"),
]

__all__ = [
    "CNNConfig",
    "MLPBatchNormConfig",
    "MLPConfig",
    "Model",
    "ResNetConfig",
]
