import torch


def compute_distribution_overlap(
    dist_a: torch.Tensor,
    dist_b: torch.Tensor,
    n_bins: int = 100,
) -> float:
    """Compute the histogram-intersection overlap of two distributions.

    Returns a value in ``[0, 1]`` where ``1.0`` means identical distributions
    and ``0.0`` means completely disjoint.  Both tensors are expected to be
    1-D and values must be finite; any NaN/Inf are silently dropped.
    """
    dist_a = dist_a.flatten().float()
    dist_b = dist_b.flatten().float()
    dist_a = dist_a[torch.isfinite(dist_a)]
    dist_b = dist_b[torch.isfinite(dist_b)]
    if dist_a.numel() == 0 or dist_b.numel() == 0:
        return 0.0

    all_values = torch.cat([dist_a, dist_b])
    lo = float(all_values.min().item())
    hi = float(all_values.max().item())
    if hi <= lo:
        return 1.0

    hist_a = torch.histc(dist_a, bins=n_bins, min=lo, max=hi)
    hist_b = torch.histc(dist_b, bins=n_bins, min=lo, max=hi)

    # Normalise to probability mass functions
    hist_a = hist_a / hist_a.sum().clamp(min=1e-12)
    hist_b = hist_b / hist_b.sum().clamp(min=1e-12)

    return float(torch.minimum(hist_a, hist_b).sum().item())
