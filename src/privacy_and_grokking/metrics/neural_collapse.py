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

    nc0: float
    rnc1: float
    nc1: float
    nc2: float
    nc2_equinorm: float
    nc2_equiangular: float
    nc3: float
    nc3_papyan: float
    nc4: float
    between_class_variance: float
    within_class_variance: float


def _class_means_and_global(
    features: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Compute class means and global mean.

    Returns:
        class_means: (K, d) tensor of class centroids.
        global_mean: (d,) tensor.
        classes: sorted list of unique class labels.
    """
    classes = labels.unique().tolist()
    d = features.shape[1]
    class_means = torch.zeros(len(classes), d, dtype=features.dtype, device=features.device)
    for i, c in enumerate(classes):
        class_means[i] = features[labels == c].mean(dim=0)
    global_mean = class_means.mean(dim=0)
    return class_means, global_mean, classes


def compute_within_class_covariance_trace(
    features: torch.Tensor,
    labels: torch.Tensor,
    class_means: torch.Tensor,
    classes: list[int],
) -> float:
    """Tr(Sigma_W) where Sigma_W = (1/N) sum_c sum_{i in S_c} (h_i - mu_c)(h_i - mu_c)^T."""
    N = features.shape[0]
    total = torch.tensor(0.0, dtype=features.dtype, device=features.device)
    for i, c in enumerate(classes):
        diff = features[labels == c] - class_means[i]
        total += (diff * diff).sum()
    return (total / N).item()


def compute_between_class_covariance_trace(
    class_means: torch.Tensor,
    global_mean: torch.Tensor,
) -> float:
    """Tr(Sigma_B) where Sigma_B = (1/K) sum_c (mu_c - mu_G)(mu_c - mu_G)^T."""
    K = class_means.shape[0]
    diff = class_means - global_mean
    return ((diff * diff).sum() / K).item()


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


def compute_nc0(classifier_weight: torch.Tensor) -> float:
    """Compute NC0: zero-row-sum metric for the last-layer classifier weight.

    NC0 quantifies how close each row of the classifier matrix W is to having
    zero sum, as proposed in the optimizer-choice NC literature.

    This implementation returns the mean absolute row sum:

        NC0 = (1 / K) * sum_c |sum_j W[c, j]|

    where W has shape (K, d). NC0 is exactly 0 iff every row sum is 0.

    Args:
        classifier_weight: Tensor of shape (K, d) — last linear layer weight.

    Returns:
        Scalar NC0 value.
    """
    if classifier_weight.numel() == 0:
        return float("nan")
    classifier_weight = classifier_weight.float()
    row_sums = classifier_weight.sum(dim=1)
    return row_sums.abs().mean().item()


def compute_nc1(features: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute NC1 = Tr(Sigma_W) / Tr(Sigma_B).

    NC1 measures the ratio of within-class variance to between-class variance.
    A value approaching 0 indicates neural collapse (within-class features
    converge to their class means while class means remain separated).

    Args:
        features: Tensor of shape (N, d).
        labels:   Tensor of shape (N,).

    Returns:
        Scalar NC1 value.
    """
    if features.shape[0] == 0:
        return float("nan")
    features = features.float()
    class_means, global_mean, classes = _class_means_and_global(features, labels)
    tr_w = compute_within_class_covariance_trace(features, labels, class_means, classes)
    tr_b = compute_between_class_covariance_trace(class_means, global_mean)
    if tr_b == 0:
        return float("inf")
    return tr_w / tr_b


def compute_nc2(features: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute NC2 = condition number of the class-mean matrix.

    NC2 captures whether the class means form a simplex equiangular tight
    frame (ETF). A condition number approaching 1 indicates that the class
    means are maximally and equally separated.

    Following Sakamoto & Sato (2026):
        NC2 = kappa((mu_1, ..., mu_K))

    where kappa is the condition number (ratio of largest to smallest singular
    value) of the K x d matrix of class means (centered at the global mean).

    Args:
        features: Tensor of shape (N, d).
        labels:   Tensor of shape (N,).

    Returns:
        Scalar NC2 value (condition number).
    """
    if features.shape[0] == 0:
        return float("nan")
    features = features.float()
    class_means, global_mean, classes = _class_means_and_global(features, labels)
    if len(classes) < 2:
        return float("nan")
    # Center class means
    centered = class_means - global_mean  # (K, d)
    sv = torch.linalg.svdvals(centered)

    # Since there are K centered class means, the maximum rank is K-1.
    # The last singular value (or beyond) is mathematically 0, but due to
    # floating point precision might be slightly > 0. We take the top K-1.
    K = len(classes)
    sv_true = sv[:K-1]
    sv_pos = sv_true[sv_true > 1e-7]

    if len(sv_pos) == 0:
        return float("inf")
    return (sv_pos[0] / sv_pos[-1]).item()


def compute_nc3(
    features: torch.Tensor,
    labels: torch.Tensor,
    classifier_weight: torch.Tensor,
) -> float:
    """Compute NC3: alignment between classifier weights and class means.

    NC3 measures the average cosine similarity between each classifier weight
    row w_c and the corresponding centered class mean (mu_c - mu_G). Perfect
    neural collapse gives NC3 = 1.

    Args:
        features: Tensor of shape (N, d).
        labels:   Tensor of shape (N,).
        classifier_weight: Tensor of shape (K, d) — last linear layer weight.

    Returns:
        Average cosine similarity across classes.
    """
    if features.shape[0] == 0:
        return float("nan")
    features = features.float()
    classifier_weight = classifier_weight.float().to(features.device)
    class_means, global_mean, classes = _class_means_and_global(features, labels)
    if len(classes) < 2:
        return float("nan")

    centered_means = class_means - global_mean  # (K, d)
    K = len(classes)
    cos_sum = 0.0
    for i, c in enumerate(classes):
        w_c = classifier_weight[c]
        m_c = centered_means[i]
        w_norm = w_c.norm()
        m_norm = m_c.norm()
        if w_norm > 0 and m_norm > 0:
            cos_sum += (w_c @ m_c / (w_norm * m_norm)).item()
        # If either is zero, cosine is undefined; skip.

    return cos_sum / K


def compute_nc4(
    features: torch.Tensor,
    labels: torch.Tensor,
    classifier_weight: torch.Tensor,
    classifier_bias: torch.Tensor | None = None,
) -> float:
    """Compute NC4: agreement between the model classifier and the nearest-class-center (NCC) classifier.

    NC4 measures what fraction of samples are classified identically by the
    linear classifier W and the nearest-class-center rule. A value of 1
    indicates perfect agreement.

    Args:
        features: Tensor of shape (N, d).
        labels:   Tensor of shape (N,).
        classifier_weight: Tensor of shape (K, d).
        classifier_bias:   Optional tensor of shape (K,).

    Returns:
        Fraction of agreement between linear classifier and NCC.
    """
    if features.shape[0] == 0:
        return float("nan")
    features = features.float()
    classifier_weight = classifier_weight.float().to(features.device)
    class_means, _, classes = _class_means_and_global(features, labels)

    # Linear classifier predictions
    logits = features @ classifier_weight.T
    if classifier_bias is not None:
        logits = logits + classifier_bias.float().to(features.device)
    linear_preds = logits.argmax(dim=1)

    # Nearest-class-center predictions
    # Distances to each class mean: (N, K)
    dists = torch.cdist(features, class_means)  # (N, K)
    ncc_preds_idx = dists.argmin(dim=1)

    # Map NCC indices back to actual class labels
    classes_tensor = torch.tensor(classes, device=features.device)
    ncc_preds = classes_tensor[ncc_preds_idx]

    agreement = (linear_preds == ncc_preds).float().mean()
    return agreement.item()

def compute_nc2_papyan(features: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    """Compute original Papyan et al. NC2 metrics (Equinorm and Equiangular)."""
    if features.shape[0] == 0:
        return float("nan"), float("nan")
    features = features.float()
    class_means, global_mean, classes = _class_means_and_global(features, labels)
    if len(classes) < 2:
        return float("nan"), float("nan")

    K = len(classes)
    centered = class_means - global_mean
    norms = centered.norm(dim=1)

    # Equinorm: std(norms) / mean(norms)  -> should approach 0
    nc2_equinorm = (norms.std() / norms.mean()).item() if norms.mean() > 0 else float("inf")

    # Equiangular: deviation from ideal cosine similarity
    normed = centered / torch.clamp(norms.unsqueeze(1), min=1e-8)
    cos_sim = normed @ normed.T

    # Papyan: || M^T M / ||M^T M||_F - 1/sqrt(K-1) (I - 1/K 11^T) ||_F
    # Alternatively, the image provides the element-wise limit for cosine sim:
    # off-diagonals approach -1 / (K-1)
    ideal_cos = (float(K) / (K - 1)) * torch.eye(K, device=features.device) - (1.0 / (K - 1)) * torch.ones(K, K, device=features.device)
    nc2_equiangular = (cos_sim - ideal_cos).norm(p='fro').item()

    return nc2_equinorm, nc2_equiangular

def compute_nc3_papyan(
    features: torch.Tensor,
    labels: torch.Tensor,
    classifier_weight: torch.Tensor,
) -> float:
    """Compute original Papyan et al. NC3 metric (Self-Duality / Frobenius difference)."""
    if features.shape[0] == 0:
        return float("nan")
    features = features.float()
    classifier_weight = classifier_weight.float().to(features.device)
    class_means, global_mean, classes = _class_means_and_global(features, labels)
    if len(classes) < 2:
        return float("nan")

    centered_means = class_means - global_mean  # (K, d)

    # Only compare weights for classes that actually have samples
    present_weights = classifier_weight[classes]

    # Normalize W and M_dot
    # W is normalized by its global Frobenius norm to strictly match the NC3 definition
    W_normed = present_weights / torch.clamp(classifier_weight.norm(p='fro'), min=1e-8)
    M_normed = centered_means / torch.clamp(centered_means.norm(p='fro'), min=1e-8)

    # || W/||W||_F - M/||M||_F ||_F  -> should approach 0
    return (W_normed - M_normed).norm(p='fro').item()


def compute_all_nc_metrics(
    features: torch.Tensor,
    labels: torch.Tensor,
    classifier_weight: torch.Tensor | None = None,
    classifier_bias: torch.Tensor | None = None,
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
    features = features.float()
    if classifier_weight is not None:
        classifier_weight = classifier_weight.to(features.device)
    if classifier_bias is not None:
        classifier_bias = classifier_bias.to(features.device)
    class_means, global_mean, classes = _class_means_and_global(features, labels)

    tr_w = compute_within_class_covariance_trace(features, labels, class_means, classes)
    tr_b = compute_between_class_covariance_trace(class_means, global_mean)

    nc0 = compute_nc0(classifier_weight) if classifier_weight is not None else float("nan")
    rnc1 = compute_rnc1(features, labels)
    nc1 = tr_w / tr_b if tr_b > 0 else float("inf")
    nc2 = compute_nc2(features, labels)
    nc2_equinorm, nc2_equiangular = compute_nc2_papyan(features, labels)
    nc3 = (
        compute_nc3(features, labels, classifier_weight)
        if classifier_weight is not None
        else float("nan")
    )
    nc3_papyan = (
        compute_nc3_papyan(features, labels, classifier_weight)
        if classifier_weight is not None
        else float("nan")
    )
    nc4 = (
        compute_nc4(features, labels, classifier_weight, classifier_bias)
        if classifier_weight is not None
        else float("nan")
    )

    return NeuralCollapseMetrics(
        nc0=nc0,
        rnc1=rnc1,
        nc1=nc1,
        nc2=nc2,
        nc2_equinorm=nc2_equinorm,
        nc2_equiangular=nc2_equiangular,
        nc3=nc3,
        nc3_papyan=nc3_papyan,
        nc4=nc4,
        between_class_variance=tr_b,
        within_class_variance=tr_w,
    )
