from typing import Literal

import torch
import torch.nn as nn

from privacy_and_grokking.models.base import ModelConfig

# Ref: https://github.com/keitaroskmt/collapse-dynamics/blob/master/src/models/transformer.py


class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, embed_dim, H/p, W/p) -> (B, N, embed_dim)
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class ViT(nn.Module):
    def __init__(
        self,
        input_dim: torch.Size,
        num_classes: int,
        patch_size: int = 4,
        embed_dim: int = 128,
        num_heads: int = 4,
    ):
        super().__init__()
        c, h, w = input_dim
        if h % patch_size != 0 or w % patch_size != 0:
            raise ValueError(f"Input size {(h, w)} must be divisible by patch_size={patch_size}.")
        num_patches = (h // patch_size) * (w // patch_size)

        self.patch_embed = PatchEmbed(c, embed_dim, patch_size)

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(
        self, x: torch.Tensor, verbose: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        b = x.shape[0]
        tokens = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls_tokens, tokens], dim=1)
        x = x + self.pos_embed[:, : x.size(1), :]

        z = self.encoder(x)
        out = self.head(z[:, 0])
        if verbose:
            return out, z[:, 0]
        return out

    def classifier(self) -> nn.Module:
        return self.head


class ViTConfig(ModelConfig):
    name: Literal["vit"] = "vit"
    patch_size: int = 4
    embed_dim: int = 128
    num_heads: int = 4

    def _create(self, input_dim: torch.Size, num_classes: int) -> nn.Module:
        return ViT(
            input_dim,
            num_classes,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
        )

    def _apply_initialization_scale(self, model: nn.Module, scale: float) -> None:
        for name, m in model.named_modules():
            if name.endswith(("linear1", "linear2", "head")):
                m.weight.data = m.weight.data * scale
                if m.bias is not None:
                    m.bias.data = m.bias.data * scale
