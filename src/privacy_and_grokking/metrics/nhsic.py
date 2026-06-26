"""Normalized HSIC (nHSIC) metrics for Information Bottleneck analysis.

Implements the normalized Hilbert-Schmidt Independence Criterion estimator
from Appendix C of:

    Sakamoto, K. and Sato, I. (2026). "Explaining Grokking and Information
    Bottleneck through Neural Collapse Emergence." ICLR 2026.
    arXiv:2509.20829.

The nHSIC serves as a proxy for mutual information, tracking the Information
Bottleneck compression/fitting dynamics:
    - nHSIC(Z, X): measures superfluous/redundant information in representations
    - nHSIC(Z, Y): measures task-relevant information retained in representations

Background:
    Fukumizu et al. (2007, Theorem 4) showed that nHSIC coincides with the
    chi-square divergence between the joint distribution P_{X,Y} and the
    product of marginals P_X P_Y, providing an information-theoretic
    interpretation analogous to mutual information.
"""

from __future__ import annotations

import torch


def _rbf_kernel(X: torch.Tensor, sigma: float | None = None) -> torch.Tensor:
    """Compute the RBF (Gaussian) kernel matrix.

    K[i, j] = exp(-||x_i - x_j||^2 / (2 * sigma^2))

    If sigma is None, uses the **median heuristic**: sigma^2 is set to the
    median of all pairwise squared distances.

    Args:
        X: Tensor of shape (N, d).
        sigma: Bandwidth parameter. If None, uses median heuristic.

    Returns:
        Kernel matrix of shape (N, N).
    """
    # Pairwise squared distances: ||x_i - x_j||^2
    dists_sq = torch.cdist(X, X, p=2).pow(2)

    if sigma is None:
        # Median heuristic: take median of upper-triangular distances
        n = X.shape[0]
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=X.device), diagonal=1)
        median_dist_sq = dists_sq[mask].median()
        if median_dist_sq == 0:
            median_dist_sq = torch.tensor(1.0, device=X.device, dtype=X.dtype)
        sigma_sq = median_dist_sq
    else:
        sigma_sq = sigma ** 2

    return torch.exp(-dists_sq / (2 * sigma_sq))


def _delta_kernel(labels: torch.Tensor) -> torch.Tensor:
    """Compute the delta (Kronecker) kernel for discrete labels.

    K[i, j] = 1 if y_i == y_j, else 0.

    Args:
        labels: Tensor of shape (N,) with integer class labels.

    Returns:
        Kernel matrix of shape (N, N).
    """
    return (labels.unsqueeze(0) == labels.unsqueeze(1)).float()


def _centering_matrix(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Compute the centering matrix H = I_N - (1/N) * 1_N * 1_N^T.

    Args:
        n: Matrix dimension.
        device: Target device.
        dtype: Target dtype.

    Returns:
        Centering matrix of shape (N, N).
    """
    return torch.eye(n, device=device, dtype=dtype) - torch.ones(n, n, device=device, dtype=dtype) / n


def compute_hsic(K_X: torch.Tensor, K_Y: torch.Tensor) -> float:
    """Biased HSIC estimator.

    HSIC_hat = Tr(K_X H K_Y H) / (N - 1)^2

    Args:
        K_X: Kernel matrix of shape (N, N) for variable X.
        K_Y: Kernel matrix of shape (N, N) for variable Y.

    Returns:
        Scalar HSIC estimate.
    """
    n = K_X.shape[0]
    if n < 2:
        return float("nan")
    H = _centering_matrix(n, K_X.device, K_X.dtype)
    # Tr(K_X H K_Y H) = Tr(HK_X H K_Y) — use the cyclic property
    HK_X = H @ K_X
    HK_Y = H @ K_Y
    return (HK_X * HK_Y.T).sum().item() / (n - 1) ** 2


def compute_nhsic(K_X: torch.Tensor, K_Y: torch.Tensor, eps: float = 1e-5) -> float:
    """Normalized HSIC (nHSIC) estimator.

    Following Appendix C of Sakamoto & Sato (2026):

        nHSIC_hat = Tr[ K_X H (K_X H + eps*N*I)^{-1}
                        K_Y H (K_Y H + eps*N*I)^{-1} ]

    This estimates ||C_XX^{-1/2} C_XY C_YY^{-1/2}||_HS^2, the Hilbert-Schmidt
    norm of the normalized cross-covariance operator.

    Args:
        K_X: Kernel matrix of shape (N, N) for variable X.
        K_Y: Kernel matrix of shape (N, N) for variable Y.
        eps: Regularization parameter (default 1e-5 per paper).

    Returns:
        Scalar nHSIC estimate.
    """
    n = K_X.shape[0]
    if n < 2:
        return float("nan")
    H = _centering_matrix(n, K_X.device, K_X.dtype)
    reg = eps * n * torch.eye(n, device=K_X.device, dtype=K_X.dtype)

    K_X_H = K_X @ H
    K_Y_H = K_Y @ H

    # Solve the regularized systems: (K_X H + eps*N*I)^{-1} and same for Y
    A_X = torch.linalg.solve(K_X_H + reg, K_X_H)
    A_Y = torch.linalg.solve(K_Y_H + reg, K_Y_H)

    # Tr(A_X^T A_Y) = sum of element-wise product
    return (A_X * A_Y).sum().item()


def _subsample(
    *tensors: torch.Tensor,
    max_samples: int,
) -> tuple[torch.Tensor, ...]:
    """Randomly subsample the first dimension of all tensors consistently.

    Args:
        *tensors: Tensors with the same first dimension N.
        max_samples: Maximum number of samples to retain.

    Returns:
        Tuple of subsampled tensors (or originals if N <= max_samples).
    """
    n = tensors[0].shape[0]
    if n <= max_samples:
        return tensors
    idx = torch.randperm(n)[:max_samples]
    return tuple(t[idx] for t in tensors)


def nhsic_features_vs_inputs(
    features: torch.Tensor,
    inputs: torch.Tensor,
    max_samples: int = 2048,
    eps: float = 1e-5,
) -> dict[str, float]:
    """Compute nHSIC(Z, X) — redundant information proxy.

    Uses RBF kernels for both the feature representations Z and the
    (flattened) inputs X, with bandwidth set by the median heuristic.

    Args:
        features: Tensor of shape (N, d_rep) — learned representations.
        inputs:   Tensor of shape (N, d_in) — flattened raw inputs.
        max_samples: Subsample size for kernel computation.
        eps: Regularization parameter for nHSIC.

    Returns:
        Dict with keys 'nhsic_zx' and 'hsic_zx'.
    """
    features, inputs = _subsample(features, inputs, max_samples=max_samples)
    features = features.float()
    inputs = inputs.float()

    K_Z = _rbf_kernel(features)
    K_X = _rbf_kernel(inputs)

    return {
        "nhsic_zx": compute_nhsic(K_Z, K_X, eps=eps),
        "hsic_zx": compute_hsic(K_Z, K_X),
    }


def nhsic_features_vs_labels(
    features: torch.Tensor,
    labels: torch.Tensor,
    max_samples: int = 2048,
    eps: float = 1e-5,
) -> dict[str, float]:
    """Compute nHSIC(Z, Y) — task-relevant information proxy.

    Uses an RBF kernel for the feature representations Z and a delta
    (Kronecker) kernel for the discrete class labels Y.

    Args:
        features: Tensor of shape (N, d_rep) — learned representations.
        labels:   Tensor of shape (N,) — integer class labels.
        max_samples: Subsample size for kernel computation.
        eps: Regularization parameter for nHSIC.

    Returns:
        Dict with keys 'nhsic_zy' and 'hsic_zy'.
    """
    features, labels = _subsample(features, labels, max_samples=max_samples)
    features = features.float()

    K_Z = _rbf_kernel(features)
    K_Y = _delta_kernel(labels)

    return {
        "nhsic_zy": compute_nhsic(K_Z, K_Y, eps=eps),
        "hsic_zy": compute_hsic(K_Z, K_Y),
    }
