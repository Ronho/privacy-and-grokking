"""Neural collapse metrics from Sakamoto & Sato (ICLR 2026).

Reference:
    Sakamoto, K. and Sato, I. (2026). "Explaining Grokking and Information
    Bottleneck through Neural Collapse Emergence." ICLR 2026.
    arXiv:2509.20829.

Also implements the standard NC1-NC4 metrics from:
    Papyan, V., Han, X., and Donoho, D. L. (2020). "Prevalence of neural
    collapse during the terminal phase of deep learning training."
    PNAS 117(40).
"""

from dataclasses import dataclass

import torch


@dataclass
class NeuralCollapseMetrics:
    """Container for all neural collapse metrics."""

    rnc1: float
    nc1: float
    nc2: float
    nc2_equinorm: float
    nc2_equinorm_weights: float
    nc2_equiangularity: float
    nc2_equiangularity_weights: float
    nc2_maximal_angle_equiangularity: float
    nc2_maximal_angle_equiangularity_weights: float
    nc3: float
    nc4: float


def compute_rnc1(features: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute the RNC1 (Rescaled NC1) metric.

    RNC1 measures the empirical within-class variance of the rescaled feature
    extractor output. It is defined as:

        RNC1 = (1 / N) * sum_c sum_{i in S_c} || g~(x_i) - mu~_c ||^2

    where g~(x) = g(x) / B_g, B_g = max_i ||g(x_i)||_2 (approximated on the
    provided feature set), mu~_c = (1 / n_c) * sum_{i in S_c} g~(x_i) is the
    class mean of rescaled features, N is the total number of samples, and S_c
    is the set of indices belonging to class c.

    Unlike NC1 = Tr(Sigma_W) / Tr(Sigma_B), RNC1 only captures within-class
    concentration without mixing in between-class separation. This makes it
    sensitive to the pure geometric clustering described in the paper.

    Args:
        features: Tensor of shape (N, d) — feature representations.
        labels:   Tensor of shape (N,) — integer class labels.

    Returns:
        Scalar float value of RNC1.
    """
    if features.shape[0] == 0:
        return float("nan")

    features = features.float()

    # B_g = sup_{x in training set} ||g(x)||_2  (approximated by max over batch)
    norms = features.norm(dim=1)  # (N,)
    B_g = norms.max()
    if B_g == 0:
        return 0.0

    features_scaled = features / B_g  # g~(x_i)

    classes = labels.unique()
    N = features.shape[0]
    total = torch.tensor(0.0, dtype=features_scaled.dtype, device=features_scaled.device)

    for c in classes:
        mask = labels == c
        class_features = features_scaled[mask]          # (n_c, d)
        class_mean = class_features.mean(dim=0)         # (d,)
        diff = class_features - class_mean               # (n_c, d)
        total += (diff * diff).sum()                     # sum of squared norms

    return (total / N).item()


def compute_rnc1_train_mean(
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
) -> float:
    """Compute RNC1 of test features using the class means and scaling from train features.

    Args:
        test_features: Tensor of shape (N_test, d).
        test_labels:   Tensor of shape (N_test,).
        train_features: Tensor of shape (N_train, d).
        train_labels:   Tensor of shape (N_train,).

    Returns:
        Scalar float value of RNC1 using train means.
    """
    if test_features.shape[0] == 0 or train_features.shape[0] == 0:
        return float("nan")

    train_features = train_features.float()
    test_features = test_features.float()

    train_norms = train_features.norm(dim=1)
    B_g = train_norms.max()
    if B_g == 0:
        return 0.0

    train_features_scaled = train_features / B_g
    test_features_scaled = test_features / B_g

    # Compute train means
    train_classes = train_labels.unique()
    train_means = {}
    for c in train_classes:
        mask = train_labels == c
        if mask.sum() > 0:
            train_means[c.item()] = train_features_scaled[mask].mean(dim=0)

    # Compute variance of test features around train means
    test_classes = test_labels.unique()
    evaluated_samples = 0
    total = torch.tensor(0.0, dtype=test_features_scaled.dtype, device=test_features_scaled.device)

    for c in test_classes:
        c_item = c.item()
        if c_item not in train_means:
            continue
        mask = test_labels == c
        class_features = test_features_scaled[mask]
        evaluated_samples += class_features.shape[0]
        train_mean = train_means[c_item]
        diff = class_features - train_mean
        total += (diff * diff).sum()

    if evaluated_samples == 0:
        return float("nan")

    return (total / evaluated_samples).item()


def compute_all_nc_metrics(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_predictions: torch.Tensor,
    classifier_weight: torch.Tensor,
) -> NeuralCollapseMetrics:
    """Compute all neural collapse metrics at once.

    Args:
        features: Tensor of shape (N, d) — penultimate layer activations.
        labels:   Tensor of shape (N,) — integer class labels.
        classifier_weight: (K, d) last linear layer weight. Required for NC0/NC3/NC4.
        classifier_bias:   (K,) optional bias of last linear layer.

    Returns:
        NeuralCollapseMetrics dataclass with all metrics.
    """
    classes = torch.unique(train_labels)
    C = len(classes)
    N, d = train_features.shape

    # Global mean (1, d)
    mu_g = torch.mean(train_features, dim=0, keepdim=True)
    # Class means (C, d)
    mu_c_list = []
    Sigma_W = torch.zeros((d, d), device=train_features.device, dtype=train_features.dtype)
    for c in classes:
        h_c = train_features[train_labels == c]  # (N_c, d)
        N_c = h_c.shape[0]
        mu_c = torch.mean(h_c, dim=0, keepdim=True)  # (1, d)
        mu_c_list.append(mu_c)
        h_c_centered = h_c - mu_c
        Sigma_W += (h_c_centered.T @ h_c_centered) / N_c
    mu_c = torch.cat(mu_c_list, dim=0) # (C, d)
    mu_c_centered = mu_c - mu_g
        

    # NC1: Tr(Sigma_W Sigma_B^!/C) where (.)^! is the Moore-Penrose pseudeoinverse
    Sigma_W /= C # Within-class covariance, (d, d)
    Sigma_B = (mu_c_centered.T @ mu_c_centered) / C # Between-class covariance, (d, d)
    Sigma_B_pinv = torch.linalg.pinv(Sigma_B, rcond)
    nc1 = (torch.trace(Sigma_W @ Sigma_B_pinv) / C).item()

    # NC2 - equinorm:
    mu_c_norm = torch.vector_norm(mu_c_centered, ord=2, dim=1) # (C,)
    nc2_equinorm = (torch.std(mu_c_norm) / torch.mean(mu_c_norm)).item()
    weight_norm = torch.vector_norm(classifier_weight, ord=2, dim=1) # (C,)
    nc2_equinorm_weights = (torch.std(weight_norm) / torch.mean(weight_norm)).item()

    # NC2 - equiangularity:
    mask = ~torch.eye(C, dtype=torch.bool, device=mu_c_centered.device)
    cos_mu = torch.matmul(mu_c_centered, mu_c_centered.T) / torch.clamp(torch.outer(mu_c_norm, mu_c_norm), min=1e-8) # (C,C)
    nc2_equiangularity = torch.std(cos_mu[mask]).item()
    cos_weights = torch.matmul(classifier_weight, classifier_weight.T) / torch.clamp(torch.outer(weight_norm, weight_norm), min=1e-8) # (C,C)
    nc2_equiangularity_weights = torch.std(cos_weights[mask]).item()
    
    # NC2 - maximal-angle equiangularity:
    nc2_maximal_angle_equiangularity = torch.mean(torch.abs(cos_mu + 1.0 / (C - 1))[mask]).item()
    nc2_maximal_angle_equiangularity_weights = torch.mean(torch.abs(cos_weights + 1.0 / (C - 1))[mask]).item()

    # NC3 - classifier convergence
    m_tilde = mu_c_centered.T / torch.matrix_norm(mu_c_centered.T, ord="fro") # (d,C)
    w_tilde = classifier_weight.T / torch.matrix_norm(classifier_weight.T, ord="fro") # (d,C)
    nc3 = (torch.matrix_norm(w_tilde - m_tilde, ord="fro")**2).item()

    # NC 4 - nearest class center convergence
    distances = torch.cdist(test_features, mu_c, p=2.0)
    ncc_predictions = torch.argmin(distances, dim=1)
    mismatch_mask = (test_predictions != ncc_predictions)
    nc4 = torch.mean(mismatch_mask.float()).item()

    # TODO: Check, calculate rnc1_test, rnc1_train_mean_test_variance
    rnc1 = compute_rnc1(train_features, train_labels)

    return NeuralCollapseMetrics(
        rnc1=rnc1,
        nc1=nc1,
        nc2_equinorm=nc2_equinorm,
        nc2_equinorm_weights=nc2_equinorm_weights,
        nc2_equiangularity=nc2_equiangularity,
        nc2_equiangularity_weights=nc2_equiangularity_weights,
        nc2_maximal_angle_equiangularity=nc2_maximal_angle_equiangularity,
        nc2_maximal_angle_equiangularity_weights=nc2_maximal_angle_equiangularity_weights,
        nc3=nc3,
        nc4=nc4,
    )
