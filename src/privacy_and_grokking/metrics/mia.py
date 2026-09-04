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


def margin_distance_lf(
    features: torch.Tensor,
    labels: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor | None,
    pool_mean_norm: float,
) -> dict[int, torch.Tensor]:
    """
    Calculate the leakage-free multi-boundary margin distance for each sample.
    Uses the classifier weights as a proxy for the class means.
    """
    dists: dict[int, torch.Tensor] = {}
    num_classes = w.shape[0]
    device = features.device

    if b is None:
        b = torch.zeros(num_classes, device=w.device)

    w = w.to(device)
    b = b.to(device)

    for c in range(num_classes):
        mask_c = labels == c
        if mask_c.sum() == 0:
            continue

        f_sub = features[mask_c].float()

        def get_all_margins(f_in):
            margins = []
            for k in range(num_classes):
                if k == c:
                    continue
                w_diff = w[c] - w[k]
                b_diff = b[c] - b[k]
                norm_w = torch.norm(w_diff, p=2)
                if norm_w == 0:
                    m = torch.zeros(f_in.shape[0], device=device)
                else:
                    m = (torch.matmul(f_in, w_diff) + b_diff) / norm_w
                margins.append(m.unsqueeze(1))
            if len(margins) == 0:
                return torch.zeros((f_in.shape[0], 0), device=device)
            return torch.cat(margins, dim=1)

        w_proxy = w[c] / torch.norm(w[c]) * pool_mean_norm
        w_proxy = w_proxy.unsqueeze(0)

        proxy_margins = get_all_margins(w_proxy)
        proxy_margin_mean = proxy_margins[0]

        sample_margins = get_all_margins(f_sub)

        dists[c] = torch.norm(sample_margins - proxy_margin_mean, dim=1).cpu()

    return dists
