from typing import Literal

import torch

from privacy_and_grokking.loss.regularizer.base import (
    RegularizerType,
    SelfContainedTwoSampleRegularizerConfig,
)


class PerSampleDistanceRegularizerConfig(SelfContainedTwoSampleRegularizerConfig):
    """Per-sample distance regularizer between clean and noisy losses.

    Unlike the distributional regularizers (Overlap, MMD, …), this one
    exploits the 1-to-1 pairing that exists in noisy-self-validation mode:
    for each sample *i* it computes the distance between the clean loss and
    the noisy loss, then aggregates across the batch.

    Parameters
    ----------
    metric : ``"l1"`` | ``"l2"`` | ``"huber"``
        Distance function applied element-wise.
    huber_delta : float
        Delta parameter for the Huber loss (only used when *metric* is
        ``"huber"``).
    """

    name: Literal["per_sample_distance"] = "per_sample_distance"
    metric: Literal["l1", "l2", "huber"] = "l1"
    huber_delta: float = 1.0

    def _make_regularizer(self) -> RegularizerType:
        validation_set_generator = self.source()
        metric = self.metric
        huber_delta = self.huber_delta

        def regularizer(train_losses: torch.Tensor) -> torch.Tensor:
            val_losses = validation_set_generator(train_losses)
            batch_size = train_losses.shape[0]
            num_val = val_losses.shape[0]

            if num_val % batch_size != 0:
                raise ValueError(
                    f"val_losses length ({num_val}) must be a multiple of "
                    f"train_losses length ({batch_size})"
                )

            num_copies = num_val // batch_size

            # Inputs may be (B,) scalars or (B, C) per-class loss vectors.
            if train_losses.dim() == 1:
                # Scalar per sample — original behaviour.
                # (K, B) — each row is one noisy copy's per-sample losses
                val_reshaped = val_losses.reshape(num_copies, batch_size)
                diffs = val_reshaped - train_losses.unsqueeze(0)  # (K, B)

                if metric == "l1":
                    distances = diffs.abs()
                elif metric == "l2":
                    distances = diffs.pow(2)
                elif metric == "huber":
                    distances = torch.nn.functional.huber_loss(
                        val_reshaped,
                        train_losses.unsqueeze(0).expand_as(val_reshaped),
                        reduction="none",
                        delta=huber_delta,
                    )
                else:
                    raise ValueError(f"Unknown metric: {metric!r}")

                return distances.mean()
            else:
                # Per-class loss vectors — (B, C) input.
                # Reshape val_losses from (K*B, C) to (K, B, C).
                extra_dims = train_losses.shape[1:]
                val_reshaped = val_losses.reshape(num_copies, batch_size, *extra_dims)
                # Difference per copy per sample per class: (K, B, C)
                diffs = val_reshaped - train_losses.unsqueeze(0)

                if metric == "l1":
                    # L2 norm across class dim, giving (K, B)
                    distances = torch.linalg.vector_norm(diffs, ord=2, dim=-1)
                elif metric == "l2":
                    # Squared L2 norm across class dim
                    distances = diffs.pow(2).sum(dim=-1)
                elif metric == "huber":
                    # Element-wise Huber, then L2 norm across class dim
                    elem = torch.nn.functional.huber_loss(
                        val_reshaped,
                        train_losses.unsqueeze(0).expand_as(val_reshaped),
                        reduction="none",
                        delta=huber_delta,
                    )
                    distances = torch.linalg.vector_norm(elem, ord=2, dim=-1)
                else:
                    raise ValueError(f"Unknown metric: {metric!r}")

                # mean over copies, then mean over batch
                return distances.mean()

        return regularizer
