import torch
import torch.nn as nn

from typing import Literal
from .cnn import CNN
from .mlp import MLP


type Model = Literal["mlp", "cnn"]

def create_model(name: str, input_dim: torch.Size, num_classes: int, initialization_scale: float | None = None) -> nn.Module:
    match name.lower():
        case "mlp":
            model = MLP(input_dim, num_classes)
        case "cnn":
            model = CNN(input_dim, num_classes)
        case _:
            raise ValueError(f"Unknown model architecture '{name}' specified.")
        
    if initialization_scale is not None:
        with torch.no_grad():
            for p in model.parameters():
                p.data *= initialization_scale

    return model
