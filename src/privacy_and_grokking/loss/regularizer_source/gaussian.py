from typing import Literal

import torch

from privacy_and_grokking.loss.regularizer_source.base import NoiseRegularizerSourceConfig


class GaussianNoiseConfig(NoiseRegularizerSourceConfig):
    """Additive Gaussian noise: x_noisy = x + N(0, std²)."""
    name: Literal["gaussian"] = "gaussian"
    mean: float = 0.0
    std: float = 1.0

    def __call__(self):
        def func(x: torch.Tensor) -> torch.Tensor:
            # x: (B, C, H, W)
            n = self.num_noisy_samples

            # Repeat each sample n times: (B, C, H, W) -> (B*n, C, H, W)
            expanded = x.repeat_interleave(n, dim=0)

            if self.mean == 0.0 and self.std == 0.0:
                return expanded.detach()

            noise = torch.randn_like(expanded) * self.std + self.mean
            return (expanded + noise).detach()

        return func
