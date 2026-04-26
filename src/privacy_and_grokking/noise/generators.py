"""Noise generators for the noisy self-validation regularizer.

Provides callable noise generators that accept an input tensor and return a
noisy copy.  Two concrete implementations are included:

- :class:`GaussianNoise` — additive Gaussian noise N(0, std²).
- :class:`SaltAndPepperNoise` — randomly replaces a fraction of elements with
  the per-tensor minimum or maximum value.

Both generators return **detached** tensors and never modify the input in
place.
"""

from typing import Protocol

import torch


class NoiseGenerator(Protocol):
    """Protocol for noise generators. Accepts a tensor, returns a noisy copy."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...


class GaussianNoise:
    """Additive Gaussian noise: x_noisy = x + N(0, std²)."""

    def __init__(self, std: float) -> None:
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.std == 0.0:
            return x.clone().detach()
        noise = torch.randn_like(x) * self.std
        return (x + noise).detach()


class SaltAndPepperNoise:
    """Salt-and-pepper noise: randomly sets a fraction of elements to min or max."""

    def __init__(self, fraction: float) -> None:
        self.fraction = fraction

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.fraction == 0.0:
            return x.clone().detach()
        out = x.clone()
        mask = torch.rand_like(x) < self.fraction
        salt_mask = torch.rand_like(x) < 0.5
        lo = x.min()
        hi = x.max()
        out[mask & salt_mask] = hi
        out[mask & ~salt_mask] = lo
        return out.detach()
