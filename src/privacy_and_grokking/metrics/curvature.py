import torch
import torch.nn as nn

from privacy_and_grokking.utils import Logger, get_device

# Number of Rademacher probes for the Hutchinson trace estimator.
_CURVATURE_HUTCHINSON_SAMPLES = 5
# Number of power-iteration steps for the top eigenvalue estimate.
_CURVATURE_POWER_ITER = 20


def _hvp(
    loss: torch.Tensor,
    params: list[nn.Parameter],
    v: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Hessian-vector product Hv via double back-propagation."""
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    gv = sum((g * vi).sum() for g, vi in zip(grads, v))
    return list(torch.autograd.grad(gv, params, retain_graph=True))


def _vec_norm(tensors: list[torch.Tensor]) -> torch.Tensor:
    """L2 norm of a list of tensors treated as a single flat vector."""
    return torch.sqrt(sum(t.pow(2).sum() for t in tensors))


def curvature(model: nn.Module, loss_fn, loader: torch.utils.data.DataLoader) -> dict[str, float]:
    """Estimate loss-landscape curvature and log to mlflow.

    Metrics logged:
      - ``curvature/hessian_trace``  – Hutchinson estimator of tr(H), averaged
        over ``_CURVATURE_HUTCHINSON_SAMPLES`` Rademacher probes.
      - ``curvature/top_eigenvalue`` – power-iteration estimate of λ_max(H)
        using ``_CURVATURE_POWER_ITER`` steps.

    Both are computed on a single mini-batch sampled from *loader* with the
    model temporarily set to train mode.
    """
    device = get_device()
    was_training = model.training
    model.train()
    metrics = {}
    try:
        x, y = next(iter(loader))
        x, y = x.to(device), y.to(device)
        params = [p for p in model.parameters() if p.requires_grad]

        # Shared forward pass — graph kept for double back-prop.
        logits = model(x)
        loss = loss_fn(logits, y)

        # --- Hutchinson trace: E_v[v^T H v], v ~ Rademacher{±1} ---
        trace_acc = torch.zeros(1, device=device)
        for _ in range(_CURVATURE_HUTCHINSON_SAMPLES):
            v = [torch.randint_like(p.data, 0, 2).float().mul_(2).sub_(1) for p in params]
            hv = _hvp(loss, params, v)
            trace_acc += sum((hvi * vi).sum() for hvi, vi in zip(hv, v))
        hessian_trace = (trace_acc / _CURVATURE_HUTCHINSON_SAMPLES).item()

        # --- Power iteration for top eigenvalue λ_max(H) ---
        v = [torch.randn_like(p.data) for p in params]
        v = [vi / _vec_norm(v) for vi in v]

        top_eigenvalue = 0.0
        for _ in range(_CURVATURE_POWER_ITER):
            hv = _hvp(loss, params, v)
            top_eigenvalue = sum((hvi * vi).sum() for hvi, vi in zip(hv, v)).item()
            hv_norm = _vec_norm(hv)
            if hv_norm < 1e-12:
                break
            v = [hvi / hv_norm for hvi in hv]

        metrics["curvature/hessian_trace"] = hessian_trace
        metrics["curvature/top_eigenvalue"] = top_eigenvalue
    except Exception:
        Logger.get().warning("Curvature estimation failed - skipping.")
    finally:
        model.train(was_training)

    return metrics
