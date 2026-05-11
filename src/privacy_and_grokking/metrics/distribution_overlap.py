import math

import numpy as np
import torch
from scipy.stats import gaussian_kde


def _adaptive_bins(n_a: int, n_b: int, max_bins: int = 100) -> int:
    """Scale bin count with the smaller sample (cube-root rule)."""
    n_min = min(n_a, n_b)
    return min(max_bins, max(10, int(math.ceil(n_min ** (1 / 3) * 2))))


# ---------------------------------------------------------------------------
# Soft (differentiable) overlap — used as a regularizer
# ---------------------------------------------------------------------------


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


def compute_distribution_overlap_adaptive(
    dist_a: torch.Tensor,
    dist_b: torch.Tensor,
    max_bins: int = 100,
) -> float:
    """Histogram-intersection overlap with adaptive bin count.

    Scales the number of bins with the smaller sample to avoid sparse-bin
    artifacts when sample sizes are imbalanced (e.g. 256 vs 10,000).
    """
    dist_a = dist_a.flatten().float()
    dist_b = dist_b.flatten().float()
    dist_a = dist_a[torch.isfinite(dist_a)]
    dist_b = dist_b[torch.isfinite(dist_b)]
    if dist_a.numel() == 0 or dist_b.numel() == 0:
        return 0.0

    n_bins = _adaptive_bins(dist_a.numel(), dist_b.numel(), max_bins)
    all_values = torch.cat([dist_a, dist_b])
    lo = float(all_values.min().item())
    hi = float(all_values.max().item())
    if hi <= lo:
        return 1.0

    hist_a = torch.histc(dist_a, bins=n_bins, min=lo, max=hi)
    hist_b = torch.histc(dist_b, bins=n_bins, min=lo, max=hi)
    hist_a = hist_a / hist_a.sum().clamp(min=1e-12)
    hist_b = hist_b / hist_b.sum().clamp(min=1e-12)
    return float(torch.minimum(hist_a, hist_b).sum().item())


def compute_distribution_overlap_kde(
    dist_a: torch.Tensor,
    dist_b: torch.Tensor,
    n_points: int = 200,
) -> float:
    """KDE-based overlap using Silverman bandwidth.

    More accurate than histogram overlap for small or imbalanced samples
    because the kernel density estimate adapts its smoothing to each
    sample's size automatically.
    """
    a = dist_a.flatten().float().cpu().numpy()
    b = dist_b.flatten().float().cpu().numpy()
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return 0.0

    kde_a = gaussian_kde(a, bw_method="silverman")
    kde_b = gaussian_kde(b, bw_method="silverman")

    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    margin = (hi - lo) * 0.1
    grid = np.linspace(lo - margin, hi + margin, n_points)
    dx = grid[1] - grid[0]

    pa = kde_a(grid)
    pb = kde_b(grid)
    return float(np.minimum(pa, pb).sum() * dx)


# ---------------------------------------------------------------------------
# Soft (differentiable) overlap — adaptive bins variant
# ---------------------------------------------------------------------------


def soft_distribution_overlap_adaptive(
    dist_a: torch.Tensor,
    dist_b: torch.Tensor,
    max_bins: int = 100,
    sigma: float = 0.05,
) -> torch.Tensor:
    """Differentiable soft overlap with adaptive bin count.

    Same Gaussian-kernel soft binning as :func:`soft_distribution_overlap`
    but scales the number of bins with the smaller sample.
    """
    dist_a = dist_a.flatten()
    dist_b = dist_b.flatten()
    if dist_a.numel() == 0 or dist_b.numel() == 0:
        return torch.tensor(0.0, device=dist_a.device)

    n_bins = _adaptive_bins(dist_a.numel(), dist_b.numel(), max_bins)

    with torch.no_grad():
        all_vals = torch.cat([dist_a, dist_b])
        lo = all_vals.min().item()
        hi = all_vals.max().item()
        if hi <= lo:
            return torch.tensor(1.0, device=dist_a.device, requires_grad=dist_a.requires_grad)
    centres = torch.linspace(lo, hi, n_bins, device=dist_a.device)

    w_a = torch.exp(-0.5 * ((dist_a.unsqueeze(1) - centres.unsqueeze(0)) / sigma) ** 2)
    w_b = torch.exp(-0.5 * ((dist_b.unsqueeze(1) - centres.unsqueeze(0)) / sigma) ** 2)

    hist_a = w_a.sum(dim=0) / w_a.sum().clamp(min=1e-12)
    hist_b = w_b.sum(dim=0) / w_b.sum().clamp(min=1e-12)

    overlap = 0.5 * (hist_a + hist_b - (hist_a - hist_b).abs())
    return overlap.sum()


def soft_distribution_overlap_kde(
    dist_a: torch.Tensor,
    dist_b: torch.Tensor,
    n_points: int = 200,
) -> torch.Tensor:
    """Differentiable KDE-based overlap.

    Uses scipy to compute Silverman bandwidths (detached), then evaluates
    Gaussian KDE in PyTorch so gradients flow through *dist_a*.
    """
    dist_a = dist_a.flatten()
    dist_b = dist_b.flatten()
    if dist_a.numel() == 0 or dist_b.numel() == 0:
        return torch.tensor(0.0, device=dist_a.device)

    with torch.no_grad():
        a_np = dist_a.detach().cpu().float().numpy()
        b_np = dist_b.detach().cpu().float().numpy()
        if len(a_np) < 2 or len(b_np) < 2:
            return torch.tensor(0.0, device=dist_a.device)

        bw_a = float(gaussian_kde(a_np, bw_method="silverman").factor)
        bw_b = float(gaussian_kde(b_np, bw_method="silverman").factor)
        std_a = float(np.std(a_np))
        std_b = float(np.std(b_np))
        sigma_a = max(bw_a * std_a, 1e-6)
        sigma_b = max(bw_b * std_b, 1e-6)

        lo = min(a_np.min(), b_np.min())
        hi = max(a_np.max(), b_np.max())
        margin = (hi - lo) * 0.1
        grid_np = np.linspace(lo - margin, hi + margin, n_points)
        dx = float(grid_np[1] - grid_np[0])

    grid = torch.tensor(grid_np, device=dist_a.device, dtype=dist_a.dtype)  # (P,)

    # KDE with gradients through dist_a: p_a(x) = (1/N) Σ K((x - x_i) / σ)
    diff_a = grid.unsqueeze(1) - dist_a.unsqueeze(0)  # (P, N)
    kde_a = torch.exp(-0.5 * (diff_a / sigma_a) ** 2).mean(dim=1)  # (P,)

    diff_b = grid.unsqueeze(1) - dist_b.unsqueeze(0)  # (P, M)
    kde_b = torch.exp(-0.5 * (diff_b / sigma_b) ** 2).mean(dim=1)  # (P,)

    # Smooth min for differentiability
    overlap = 0.5 * (kde_a + kde_b - (kde_a - kde_b).abs()) * dx
    return overlap.sum()


# ---------------------------------------------------------------------------
# KL divergence
# ---------------------------------------------------------------------------


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


def compute_kl_divergence_adaptive(
    dist_a: torch.Tensor,
    dist_b: torch.Tensor,
    max_bins: int = 100,
) -> float:
    """KL(dist_a || dist_b) with adaptive bin count."""
    dist_a = dist_a.flatten().float()
    dist_b = dist_b.flatten().float()
    dist_a = dist_a[torch.isfinite(dist_a)]
    dist_b = dist_b[torch.isfinite(dist_b)]
    if dist_a.numel() == 0 or dist_b.numel() == 0:
        return 0.0

    n_bins = _adaptive_bins(dist_a.numel(), dist_b.numel(), max_bins)
    all_values = torch.cat([dist_a, dist_b])
    lo, hi = float(all_values.min()), float(all_values.max())
    if hi <= lo:
        return 0.0

    hist_a = torch.histc(dist_a, bins=n_bins, min=lo, max=hi) + 1e-8
    hist_b = torch.histc(dist_b, bins=n_bins, min=lo, max=hi) + 1e-8
    p = hist_a / hist_a.sum()
    q = hist_b / hist_b.sum()
    return float((p * (p / q).log()).sum().item())


def compute_kl_divergence_kde(
    dist_a: torch.Tensor,
    dist_b: torch.Tensor,
    n_points: int = 200,
) -> float:
    """KL(dist_a || dist_b) estimated via KDE with Silverman bandwidth."""
    a = dist_a.flatten().float().cpu().numpy()
    b = dist_b.flatten().float().cpu().numpy()
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return 0.0

    kde_a = gaussian_kde(a, bw_method="silverman")
    kde_b = gaussian_kde(b, bw_method="silverman")

    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    margin = (hi - lo) * 0.1
    grid = np.linspace(lo - margin, hi + margin, n_points)
    dx = grid[1] - grid[0]

    pa = np.maximum(kde_a(grid), 1e-12)
    pb = np.maximum(kde_b(grid), 1e-12)
    # Renormalize to proper densities over the grid
    pa = pa / (pa.sum() * dx)
    pb = pb / (pb.sum() * dx)
    return float((pa * np.log(pa / pb)).sum() * dx)


# ---------------------------------------------------------------------------
# Jensen-Shannon distance
# ---------------------------------------------------------------------------
#
# JSD(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M),  where M = 0.5 * (P + Q)
# JS distance = sqrt(JSD).  With natural log, JSD ∈ [0, ln(2)], so the
# distance is bounded in [0, sqrt(ln(2)) ≈ 0.8326].  Unlike KL, it is
# symmetric and always finite.


def _jsd_from_pmfs(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Jensen-Shannon divergence (in nats) between two PMFs on the same support."""
    eps = 1e-12
    m = 0.5 * (p + q)
    kl_pm = (p * (p.clamp(min=eps) / m.clamp(min=eps)).log()).sum()
    kl_qm = (q * (q.clamp(min=eps) / m.clamp(min=eps)).log()).sum()
    return 0.5 * (kl_pm + kl_qm)


def compute_js_distance(
    dist_a: torch.Tensor,
    dist_b: torch.Tensor,
    n_bins: int = 100,
) -> float:
    """Jensen-Shannon distance between two 1-D distributions via histograms.

    Returns sqrt(JSD) in nats — symmetric, bounded in [0, sqrt(ln(2))].
    """
    dist_a = dist_a.flatten().float()
    dist_b = dist_b.flatten().float()
    dist_a = dist_a[torch.isfinite(dist_a)]
    dist_b = dist_b[torch.isfinite(dist_b)]
    if dist_a.numel() == 0 or dist_b.numel() == 0:
        return 1.0

    all_values = torch.cat([dist_a, dist_b])
    lo, hi = float(all_values.min()), float(all_values.max())
    if hi <= lo:
        return 1.0

    hist_a = torch.histc(dist_a, bins=n_bins, min=lo, max=hi)
    hist_b = torch.histc(dist_b, bins=n_bins, min=lo, max=hi)
    p = hist_a / hist_a.sum().clamp(min=1e-12)
    q = hist_b / hist_b.sum().clamp(min=1e-12)
    jsd = _jsd_from_pmfs(p, q).clamp(min=0.0)
    return float(jsd.sqrt().item())


def compute_js_distance_adaptive(
    dist_a: torch.Tensor,
    dist_b: torch.Tensor,
    max_bins: int = 100,
) -> float:
    """Jensen-Shannon distance with adaptive bin count."""
    dist_a = dist_a.flatten().float()
    dist_b = dist_b.flatten().float()
    dist_a = dist_a[torch.isfinite(dist_a)]
    dist_b = dist_b[torch.isfinite(dist_b)]
    if dist_a.numel() == 0 or dist_b.numel() == 0:
        return 0.0

    n_bins = _adaptive_bins(dist_a.numel(), dist_b.numel(), max_bins)
    all_values = torch.cat([dist_a, dist_b])
    lo, hi = float(all_values.min()), float(all_values.max())
    if hi <= lo:
        return 0.0

    hist_a = torch.histc(dist_a, bins=n_bins, min=lo, max=hi)
    hist_b = torch.histc(dist_b, bins=n_bins, min=lo, max=hi)
    p = hist_a / hist_a.sum().clamp(min=1e-12)
    q = hist_b / hist_b.sum().clamp(min=1e-12)
    jsd = _jsd_from_pmfs(p, q).clamp(min=0.0)
    return float(jsd.sqrt().item())


def _silverman_bandwidth(x: torch.Tensor) -> torch.Tensor:
    """Silverman's rule-of-thumb bandwidth for 1-D Gaussian KDE, on tensors."""
    n = x.numel()
    std = x.std(unbiased=True)
    # Standard 1-D Silverman: 1.06 * sigma * n^(-1/5)
    bw = 1.06 * std * (n ** (-1.0 / 5.0))
    return bw.clamp(min=1e-6)


def compute_js_distance_kde(
    dist_a: torch.Tensor,
    dist_b: torch.Tensor,
    n_points: int = 200,
) -> float:
    """Jensen-Shannon distance estimated via Gaussian KDE (pure PyTorch).

    Bandwidths are computed via Silverman's rule directly on tensors, then
    each KDE is evaluated on a shared grid spanning the union of both
    samples (with 10% margin) so the densities live on the same support.
    """
    dist_a = dist_a.flatten().float()
    dist_b = dist_b.flatten().float()
    dist_a = dist_a[torch.isfinite(dist_a)]
    dist_b = dist_b[torch.isfinite(dist_b)]
    if dist_a.numel() < 2 or dist_b.numel() < 2:
        return 0.0

    sigma_a = _silverman_bandwidth(dist_a)
    sigma_b = _silverman_bandwidth(dist_b)

    lo = torch.minimum(dist_a.min(), dist_b.min())
    hi = torch.maximum(dist_a.max(), dist_b.max())
    if (hi - lo).item() <= 0:
        return 0.0
    margin = (hi - lo) * 0.1
    grid = torch.linspace(
        (lo - margin).item(),
        (hi + margin).item(),
        n_points,
        device=dist_a.device,
        dtype=dist_a.dtype,
    )
    dx = (grid[1] - grid[0])

    # Gaussian KDE: p(x) = (1 / (N * σ * sqrt(2π))) Σ exp(-0.5 ((x - x_i)/σ)^2)
    inv_sqrt2pi = 1.0 / math.sqrt(2.0 * math.pi)
    diff_a = (grid.unsqueeze(1) - dist_a.unsqueeze(0)) / sigma_a
    pa = torch.exp(-0.5 * diff_a.pow(2)).mean(dim=1) * (inv_sqrt2pi / sigma_a)
    diff_b = (grid.unsqueeze(1) - dist_b.unsqueeze(0)) / sigma_b
    pb = torch.exp(-0.5 * diff_b.pow(2)).mean(dim=1) * (inv_sqrt2pi / sigma_b)

    # Convert continuous densities to PMFs over the grid for a discrete JSD.
    p = (pa * dx)
    q = (pb * dx)
    p = p / p.sum().clamp(min=1e-12)
    q = q / q.sum().clamp(min=1e-12)
    jsd = _jsd_from_pmfs(p, q).clamp(min=0.0)
    return float(jsd.sqrt().item())


# ---------------------------------------------------------------------------
# MMD
# ---------------------------------------------------------------------------


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
