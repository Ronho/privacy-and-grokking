import torch


def soft_distribution_overlap(
    dist_a: torch.Tensor,
    dist_b: torch.Tensor,
    n_bins: int = 50,
    sigma: float = 0.05,
) -> torch.Tensor:
    """Differentiable histogram-intersection overlap using soft binning.

    Uses Gaussian kernels to assign samples softly to bins, producing
    gradients that flow back through *dist_a* (and *dist_b* if not detached).

    Args:
        dist_a: 1-D tensor of values (e.g. sigmoid probabilities).
        dist_b: 1-D tensor of values.
        n_bins: Number of histogram bins.
        sigma: Gaussian kernel width controlling softness.

    Returns:
        Scalar tensor in [0, 1] — 1.0 means identical distributions.
    """
    dist_a = dist_a.flatten()
    dist_b = dist_b.flatten()
    if dist_a.numel() == 0 or dist_b.numel() == 0:
        return torch.tensor(0.0, device=dist_a.device)

    # Derive bin range from data (detached so bin positions don't carry grad)
    with torch.no_grad():
        all_vals = torch.cat([dist_a, dist_b])
        lo = all_vals.min().item()
        hi = all_vals.max().item()
        if hi <= lo:
            return torch.tensor(1.0, device=dist_a.device, requires_grad=dist_a.requires_grad)
    centres = torch.linspace(lo, hi, n_bins, device=dist_a.device)  # (B,)

    # Soft assignment: weight(sample, bin) = exp(-0.5 * ((x - c) / sigma)^2)
    # Shapes: dist (N, 1), centres (1, B) -> weights (N, B)
    w_a = torch.exp(-0.5 * ((dist_a.unsqueeze(1) - centres.unsqueeze(0)) / sigma) ** 2)
    w_b = torch.exp(-0.5 * ((dist_b.unsqueeze(1) - centres.unsqueeze(0)) / sigma) ** 2)

    # Normalise to PMFs
    hist_a = w_a.sum(dim=0) / w_a.sum().clamp(min=1e-12)
    hist_b = w_b.sum(dim=0) / w_b.sum().clamp(min=1e-12)

    # Smooth min: 0.5 * (a + b - |a - b|)  (differentiable everywhere)
    overlap = 0.5 * (hist_a + hist_b - (hist_a - hist_b).abs())
    return overlap.sum()


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


def compute_kl_divergence(
    dist_a: torch.Tensor,
    dist_b: torch.Tensor,
    n_bins: int = 100,
) -> float:
    """KL(dist_a || dist_b) estimated via histograms.

    Returns KL divergence in nats.  Uses Laplace smoothing so the result
    is always finite.
    """
    dist_a = dist_a.flatten().float()
    dist_b = dist_b.flatten().float()
    dist_a = dist_a[torch.isfinite(dist_a)]
    dist_b = dist_b[torch.isfinite(dist_b)]
    if dist_a.numel() == 0 or dist_b.numel() == 0:
        return 0.0

    all_values = torch.cat([dist_a, dist_b])
    lo, hi = float(all_values.min()), float(all_values.max())
    if hi <= lo:
        return 0.0

    hist_a = torch.histc(dist_a, bins=n_bins, min=lo, max=hi) + 1e-8
    hist_b = torch.histc(dist_b, bins=n_bins, min=lo, max=hi) + 1e-8
    p = hist_a / hist_a.sum()
    q = hist_b / hist_b.sum()
    return float((p * (p / q).log()).sum().item())


def compute_mmd(
    dist_a: torch.Tensor,
    dist_b: torch.Tensor,
    bandwidth: float = 0.1,
) -> float:
    """Unbiased MMD² between two 1-D distributions using a Gaussian RBF kernel."""
    dist_a = dist_a.flatten().float()
    dist_b = dist_b.flatten().float()
    dist_a = dist_a[torch.isfinite(dist_a)]
    dist_b = dist_b[torch.isfinite(dist_b)]
    n, m = dist_a.numel(), dist_b.numel()
    if n == 0 or m == 0:
        return 0.0

    x = dist_a.unsqueeze(1)  # (N,1)
    y = dist_b.unsqueeze(1)  # (M,1)
    bw2 = 2.0 * bandwidth ** 2

    k_xx = torch.exp(-torch.cdist(x, x, p=2).pow(2) / bw2)
    k_yy = torch.exp(-torch.cdist(y, y, p=2).pow(2) / bw2)
    k_xy = torch.exp(-torch.cdist(x, y, p=2).pow(2) / bw2)

    mmd = (
        (k_xx.sum() - k_xx.diagonal().sum()) / max(n * (n - 1), 1)
        + (k_yy.sum() - k_yy.diagonal().sum()) / max(m * (m - 1), 1)
        - 2.0 * k_xy.mean()
    )
    return float(max(mmd.item(), 0.0))
