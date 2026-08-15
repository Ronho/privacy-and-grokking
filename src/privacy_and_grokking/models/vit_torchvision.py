from privacy_and_grokking.models.base import ModelBase
from typing import Literal

import torch
import torch.nn as nn
from torchvision.models.vision_transformer import VisionTransformer

from privacy_and_grokking.models.base import ModelConfig


class ViTTorchvision(ModelBase):
    def __init__(
        self,
        input_dim: torch.Size,
        num_classes: int,
        patch_size: int = 4,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 1,
    ):
        super().__init__()
        c, h, w = input_dim
        if h != w:
            raise ValueError(f"Input size {(h, w)} must be square.")
        if h % patch_size != 0 or w % patch_size != 0:
            raise ValueError(f"Input size {(h, w)} must be divisible by patch_size={patch_size}.")
        
        # Torchvision VisionTransformer expects at least 1 layer
        self.vit = VisionTransformer(
            image_size=h,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=embed_dim,
            mlp_dim=embed_dim * 4,
            dropout=0.0,
            attention_dropout=0.0,
            num_classes=num_classes,
        )
        
        # Torchvision ViT by default expects 3 channels. 
        # If input has different channels (e.g. 1 for MNIST), we override the conv projection.
        if c != 3:
            self.vit.conv_proj = nn.Conv2d(
                in_channels=c, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size
            )

    def forward(
        self, x: torch.Tensor, verbose: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # We manually step through torchvision's ViT forward pass so we can extract 'z'
        # Reshape and permute the input tensor
        x = self.vit._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.vit.encoder(x)

        # Classifier "token" as used by standard language architectures
        z = x[:, 0]

        out = self.vit.heads(z)
        
        if verbose:
            return out, z
        return out
        
    def classifier(self) -> nn.Module:
        # self.vit.heads is a Sequential with a single Linear layer named 'head'
        return self.vit.heads.head


class ViTTorchvisionConfig(ModelConfig):
    name: Literal["vit_torchvision"] = "vit_torchvision"

    def _create(self, input_dim: torch.Size, num_classes: int) -> nn.Module:
        return ViTTorchvision(
            input_dim,
            num_classes,
        )

    def _apply_initialization_scale(self, model: nn.Module, scale: float) -> None:
        for name, m in model.named_modules():
            if "heads" in name and isinstance(m, nn.Linear):
                m.weight.data = m.weight.data * scale
                if m.bias is not None:
                    m.bias.data = m.bias.data * scale
