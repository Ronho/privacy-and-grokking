import numpy as np
import torch


def distances_to_class_mean(
    features: torch.Tensor,
    labels: torch.Tensor,
    class_means: torch.Tensor,
) -> dict[int, torch.Tensor]:
    """
    Calculate the L2 distance of each sample's features to its class mean.

    Args:
        features: Tensor of shape (N, D).
        labels: Tensor of shape (N,).
        class_means: Tensor of shape (num_classes, D).

    Returns:
        A dictionary mapping class index to a 1D tensor of L2 distances.
    """
    dists: dict[int, torch.Tensor] = {}
    num_classes = class_means.shape[0]
    for c in range(num_classes):
        mask_c = labels == c
        if mask_c.sum() == 0:
            continue
        diff = features[mask_c].float() - class_means[c].to(features.device)
        dists[c] = diff.norm(dim=1).cpu()
    return dists
