import torch


def compute_distribution_overlap(
    dist_a: torch.Tensor,
    dist_b: torch.Tensor,
    n_bins: int = 100,
) -> float:
    """Compute the histogram-intersection overlap of two distributions.

    Returns a value in [0, 1] where 1.0 means identical distributions
    and 0.0 means completely disjoint. Both tensors are expected to be
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


def subsample_tensor(t: torch.Tensor, max_samples: int = 1000):
    """Randomly subsamples a 1D tensor if it exceeds max_samples."""
    if t.numel() > max_samples:
        # torch.randperm generates random indices without replacement
        indices = torch.randperm(t.numel(), device=t.device)[:max_samples]
        return t[indices]
    return t


def compute_median_heuristic(dist_a: torch.Tensor, dist_b: torch.Tensor, max_samples: int = 1000):
    """
    Computes the median heuristic for the RBF kernel bandwidth.
    """
    # Combine distributions
    z = torch.cat([dist_a.flatten().float(), dist_b.flatten().float()])

    # Filter non-finite values safely
    z = z[torch.isfinite(z)]

    if z.numel() <= 1:
        return 1.0  # Safe fallback

    # Subsample to prevent OOM errors on large datasets
    if z.numel() > max_samples:
        indices = torch.randperm(z.numel(), device=z.device)[:max_samples]
        z = z[indices]

    # Compute pairwise absolute distances (since your inputs are 1D)
    # If using N-dimensional data later, use torch.cdist(z, z)
    z = z.unsqueeze(1)
    pairwise_dists = (z - z.mT).abs()

    # Extract the upper triangle (exclude diagonal self-distances)
    n_samples = z.size(0)
    i, j = torch.triu_indices(n_samples, n_samples, offset=1)
    off_diag_dists = pairwise_dists[i, j]

    # Calculate the median
    median_bw = torch.median(off_diag_dists)

    # Fallback to 1.0 if the median is 0 (happens if >50% of the data points are identical)
    if median_bw == 0.0:
        return 1.0

    return median_bw.item()


def compute_mmd(
    dist_a: torch.Tensor, dist_b: torch.Tensor, bandwidth: float = 1.0, return_tensor: bool = False
):
    dist_a = dist_a.flatten().float()
    dist_b = dist_b.flatten().float()

    # Filter non-finite values
    dist_a = dist_a[torch.isfinite(dist_a)]
    dist_b = dist_b[torch.isfinite(dist_b)]

    n, m = dist_a.numel(), dist_b.numel()

    if n <= 1 or m <= 1:
        # We need at least 2 elements for the unbiased estimator n*(n-1)
        return torch.tensor(0.0, device=dist_a.device) if return_tensor else 0.0

    # Reshape for broadcasting
    x = dist_a.unsqueeze(1)  # (N, 1)
    y = dist_b.unsqueeze(0)  # (1, M)
    bw2 = 2.0 * (bandwidth**2)

    # Broadcast squared distances
    dxx_sq = (x - x.mT).pow(2)
    dyy_sq = (y.mT - y).pow(2)
    dxy_sq = (x - y).pow(2)

    k_xx = torch.exp(-dxx_sq / bw2)
    k_yy = torch.exp(-dyy_sq / bw2)
    k_xy = torch.exp(-dxy_sq / bw2)

    # Compute unbiased MMD using .mean() for better float32 stability
    # (n * mean(K_xx) - 1) / (n - 1) is mathematically identical to (sum(K_xx) - n) / (n*(n-1))
    term_xx = (n * k_xx.mean() - 1.0) / (n - 1.0)
    term_yy = (m * k_yy.mean() - 1.0) / (m - 1.0)
    term_xy = 2.0 * k_xy.mean()

    mmd = term_xx + term_yy - term_xy

    mmd = torch.clamp(mmd, min=0.0)

    return mmd if return_tensor else mmd.item()
