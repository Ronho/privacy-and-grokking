from typing import Literal

import torch

from privacy_and_grokking.loss.regularizer_source.base import NoiseRegularizerSourceConfig


class SaltAndPepperNoiseConfig(NoiseRegularizerSourceConfig):
    """Salt-and-pepper noise: randomly sets a fraction of elements to min or max."""

    name: Literal["salt_and_pepper"] = "salt_and_pepper"
    fraction: float | None = None

    def __call__(self):
        def func(x: torch.Tensor) -> torch.Tensor:
            # x: (B, C, H, W)
            n = self.num_noisy_samples

            # Repeat each sample n times: (B, C, H, W) -> (B*n, C, H, W)
            expanded = x.repeat_interleave(n, dim=0)

            if self.fraction is None or self.fraction == 0.0:
                return expanded.detach()

            noise_mask = torch.rand_like(expanded) < self.fraction
            salt_mask = torch.rand_like(expanded) < 0.5

            lo = x.min()
            hi = x.max()

            values = torch.where(salt_mask, hi, lo)
            out = torch.where(noise_mask, values, expanded)
            return out.detach()

        return func
