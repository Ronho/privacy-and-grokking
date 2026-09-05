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

    rnc1_train: float
    rnc1_test: float
    rnc1_train_mean_test_variance: float
    rnc1_train_impl: float
    rnc1_test_impl: float
    rnc1_train_mean_test_variance_impl: float
    nc1: float
    nc2_equinorm: float
    nc2_equinorm_weights: float
    nc2_equiangularity: float
    nc2_equiangularity_weights: float
    nc2_maximal_angle_equiangularity: float
    nc2_maximal_angle_equiangularity_weights: float
    nc3: float
    nc4: float
    nc4_test: float | None = None


def compute_all_nc_metrics(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    train_predictions: torch.Tensor,
    classifier_weight: torch.Tensor,
    test_predictions: torch.Tensor | None = None,
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
    classifier_weight = classifier_weight.detach().to(train_features.device)
    if isinstance(train_predictions, list):
        train_predictions = torch.cat(train_predictions, dim=0)
    train_predictions = train_predictions.to(train_features.device)
    if train_predictions.dim() > 1:
        train_predictions = train_predictions.argmax(dim=1)

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
    mu_c = torch.cat(mu_c_list, dim=0)  # (C, d)
    mu_c_centered = mu_c - mu_g

    # NC1: Tr(Sigma_W Sigma_B^!/C) where (.)^! is the Moore-Penrose pseudeoinverse
    Sigma_W /= C  # Within-class covariance, (d, d)
    Sigma_B = (mu_c_centered.T @ mu_c_centered) / C  # Between-class covariance, (d, d)
    Sigma_B_pinv = torch.linalg.pinv(Sigma_B)
    nc1 = (torch.trace(Sigma_W @ Sigma_B_pinv) / C).item()

    # NC2 - equinorm:
    mu_c_norm = torch.linalg.vector_norm(mu_c_centered, ord=2, dim=1)  # (C,)
    nc2_equinorm = (torch.std(mu_c_norm) / torch.mean(mu_c_norm)).item()
    weight_norm = torch.linalg.vector_norm(classifier_weight, ord=2, dim=1)  # (C,)
    nc2_equinorm_weights = (torch.std(weight_norm) / torch.mean(weight_norm)).item()

    # NC2 - equiangularity:
    mask = ~torch.eye(C, dtype=torch.bool, device=mu_c_centered.device)
    cos_mu = torch.matmul(mu_c_centered, mu_c_centered.T) / torch.clamp(
        torch.outer(mu_c_norm, mu_c_norm), min=1e-8
    )  # (C,C)
    nc2_equiangularity = torch.std(cos_mu[mask]).item()
    cos_weights = torch.matmul(classifier_weight, classifier_weight.T) / torch.clamp(
        torch.outer(weight_norm, weight_norm), min=1e-8
    )  # (C,C)
    nc2_equiangularity_weights = torch.std(cos_weights[mask]).item()

    # NC2 - maximal-angle equiangularity:
    nc2_maximal_angle_equiangularity = torch.mean(torch.abs(cos_mu + 1.0 / (C - 1))[mask]).item()
    nc2_maximal_angle_equiangularity_weights = torch.mean(
        torch.abs(cos_weights + 1.0 / (C - 1))[mask]
    ).item()

    # NC3 - classifier convergence
    m_tilde = mu_c_centered.T / torch.linalg.matrix_norm(mu_c_centered.T, ord="fro")  # (d,C)
    w_tilde = classifier_weight.T / torch.linalg.matrix_norm(
        classifier_weight.T, ord="fro"
    )  # (d,C)
    nc3 = (torch.linalg.matrix_norm(w_tilde - m_tilde, ord="fro") ** 2).item()

    # NC 4 - nearest class center convergence
    distances = torch.cdist(train_features, mu_c, p=2.0)
    ncc_predictions_idx = torch.argmin(distances, dim=1)
    mismatch_mask = train_predictions != ncc_predictions_idx
    nc4 = torch.mean(mismatch_mask.float()).item()

    nc4_test = None
    if test_predictions is not None and test_features.numel() > 0:
        if isinstance(test_predictions, list):
            test_predictions = torch.cat(test_predictions, dim=0)
        test_predictions = test_predictions.to(train_features.device)
        if test_predictions.dim() > 1:
            test_predictions = test_predictions.argmax(dim=1)
        test_distances = torch.cdist(test_features.to(train_features.device), mu_c, p=2.0)
        ncc_test_predictions_idx = torch.argmin(test_distances, dim=1)
        test_mismatch_mask = test_predictions != ncc_test_predictions_idx
        nc4_test = torch.mean(test_mismatch_mask.float()).item()

    # RNC1 according to the paper
    B_g_train = train_features.norm(p=2, dim=1).max()
    g_tilde_train = train_features / B_g_train
    B_g_test = test_features.norm(p=2, dim=1).max()
    g_tilde_test = test_features / B_g_test
    B_g_all = torch.maximum(B_g_train, B_g_test)
    g_tilde_test_all = test_features / B_g_all
    g_tilde_train_all = train_features / B_g_all

    train_total = 0
    test_total = 0
    test_all_total = 0
    for c in classes:
        train_mask = train_labels == c
        test_mask = test_labels == c
        train_class_features = g_tilde_train[train_mask]
        test_class_features = g_tilde_test[test_mask]
        test_all_class_features = g_tilde_test_all[test_mask]
        train_class_mean = train_class_features.mean(dim=0)
        test_class_mean = test_class_features.mean(dim=0)
        test_all_class_mean = g_tilde_train_all[train_mask].mean(dim=0)
        train_diff = (train_class_features - train_class_mean).norm(p=2, dim=1) ** 2
        test_diff = (test_class_features - test_class_mean).norm(p=2, dim=1) ** 2
        test_all_diff = (test_all_class_features - test_all_class_mean).norm(p=2, dim=1) ** 2
        train_total += train_diff.sum()
        test_total += test_diff.sum()
        test_all_total += test_all_diff.sum()

    rnc1_train = (train_total / train_features.shape[0]).item()
    rnc1_test = (test_total / test_features.shape[0]).item()
    rnc1_test_all = (test_all_total / test_features.shape[0]).item()

    # RNC1 According to the Implementation in their GitHub Repo
    scale_mean_train = train_features.norm(p=2, dim=1).mean()
    scale_mean_test = test_features.norm(p=2, dim=1).mean()
    scale_mean_all = torch.cat([train_features, test_features]).norm(p=2, dim=1).mean()

    train_total = 0
    test_total = 0
    test_all_total = 0

    for c in classes:
        train_mask = train_labels == c
        test_mask = test_labels == c
        train_class_features = train_features[train_mask]
        test_class_features = test_features[test_mask]
        train_class_mean = train_class_features.mean(dim=0)
        test_class_mean = test_class_features.mean(dim=0)

        train_diff = (train_class_features - train_class_mean).norm(p=2, dim=1) ** 2
        test_diff = (test_class_features - test_class_mean).norm(p=2, dim=1) ** 2
        test_all_diff = (test_class_features - train_class_mean).norm(p=2, dim=1) ** 2

        train_total += train_diff.sum()
        test_total += test_diff.sum()
        test_all_total += test_all_diff.sum()

    rnc1_train_impl = (train_total / train_features.shape[0] / (scale_mean_train**2 + 1e-10)).item()
    rnc1_test_impl = (test_total / test_features.shape[0] / (scale_mean_test**2 + 1e-10)).item()
    rnc1_test_all_impl = (
        test_all_total / test_features.shape[0] / (scale_mean_all**2 + 1e-10)
    ).item()

    return NeuralCollapseMetrics(
        rnc1_train=rnc1_train,
        rnc1_test=rnc1_test,
        rnc1_train_mean_test_variance=rnc1_test_all,
        rnc1_train_impl=rnc1_train_impl,
        rnc1_test_impl=rnc1_test_impl,
        rnc1_train_mean_test_variance_impl=rnc1_test_all_impl,
        nc1=nc1,
        nc2_equinorm=nc2_equinorm,
        nc2_equinorm_weights=nc2_equinorm_weights,
        nc2_equiangularity=nc2_equiangularity,
        nc2_equiangularity_weights=nc2_equiangularity_weights,
        nc2_maximal_angle_equiangularity=nc2_maximal_angle_equiangularity,
        nc2_maximal_angle_equiangularity_weights=nc2_maximal_angle_equiangularity_weights,
        nc3=nc3,
        nc4=nc4,
        nc4_test=nc4_test,
    )
