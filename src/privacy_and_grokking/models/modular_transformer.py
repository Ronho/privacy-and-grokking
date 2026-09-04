from typing import Literal

import torch
import torch.nn as nn

from privacy_and_grokking.models.base import ModelBase, ModelConfig


class ModularTransformer(ModelBase):
    def __init__(
        self,
        input_dim: torch.Size,
        num_classes: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        hidden_dim: int = 512,
    ):
        super().__init__()
        # input_dim is expected to be [seq_len, token_dim], e.g. [3, P+1]
        self.seq_len, self.token_dim = input_dim

        # Token embedding: Linear mapping from one-hot to embed_dim
        self.token_embed = nn.Linear(self.token_dim, embed_dim, bias=False)

        # Learned positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, self.seq_len, embed_dim))

        # Multi-head attention (no LayerNorm)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )

        # MLP block (no LayerNorm, ReLU activation)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, embed_dim)
        )

        # Unembed matrices (not tied to token_embed)
        self.head = nn.Linear(embed_dim, num_classes, bias=False)

    def forward(
        self, x: torch.Tensor, verbose: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # x is (B, 3, P+1) one-hot encoded tokens

        # Token and Positional Embedding
        x = self.token_embed(x)  # (B, 3, 128)
        x = x + self.pos_embed

        # 1-Layer Transformer Block (No LayerNorm)
        # Residual connection around attention
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = x + attn_out

        # Residual connection around MLP
        mlp_out = self.mlp(x)
        x = x + mlp_out

        # Readout from the special '=' token (the last token in the sequence)
        z = x[:, -1, :]  # (B, 128)

        out = self.head(z)  # (B, P)

        if verbose:
            return out, z
        return out

    def classifier(self) -> nn.Module:
        return self.head


class ModularTransformerConfig(ModelConfig):
    name: Literal["modular_transformer"] = "modular_transformer"
    embed_dim: int = 128
    num_heads: int = 4
    hidden_dim: int = 512

    def _create(self, input_dim: torch.Size, num_classes: int) -> nn.Module:
        return ModularTransformer(
            input_dim=input_dim,
            num_classes=num_classes,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            hidden_dim=self.hidden_dim,
        )
