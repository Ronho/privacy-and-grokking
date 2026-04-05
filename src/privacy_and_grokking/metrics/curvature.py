import torch
import torch.nn as nn
from torch.func import functional_call, grad, jvp

from privacy_and_grokking.utils import Logger, get_device

# Number of Rademacher probes for the Hutchinson trace estimator.
_CURVATURE_HUTCHINSON_SAMPLES = 5
# Number of power-iteration steps for the top eigenvalue estimate.
_CURVATURE_POWER_ITER = 20


def curvature(
    model: nn.Module,
    loss_fn,
    loader: torch.utils.data.DataLoader,
) -> dict[str, float]:
    """Estimate loss-landscape curvature using Hutchinson Approximation"""
    device = get_device()
    was_training = model.training
    model.train()
    try:
        x, y = next(iter(loader))
        x, y = x.to(device), y.to(device)

        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())
        param_names = params.keys()
        param_values = tuple(params.values())

        def compute_loss(p_values):
            p_dict = dict(zip(param_names, p_values))
            logits = functional_call(model, (p_dict, buffers), x)
            return loss_fn(logits, y)

        def get_trace_sample(v):
            _, hv = jvp(grad(compute_loss), (param_values,), (v,))
            return sum((hvi * vi).sum() for hvi, vi in zip(hv, v))

        vs = tuple(
            torch.randint(0, 2, (_CURVATURE_HUTCHINSON_SAMPLES, *p.shape), device=device).float().mul(2).sub(1)
            for p in param_values
        )

        trace_samples = torch.vmap(get_trace_sample)(vs)
        hessian_trace = trace_samples.mean().item()

        v = tuple(torch.randn_like(p) for p in param_values)
        v_norm = torch.sqrt(torch.sum(vi.pow(2).sum() for vi in v))
        v = tuple(vi / v_norm for vi in v)

        top_eigenvalue = 0.0
        # Pre-compile the grad function to avoid overhead
        grad_fn = grad(compute_loss)

        for _ in range(_CURVATURE_POWER_ITER):
            _, hv = jvp(grad_fn, (param_values,), (v,))

            top_eigenvalue = sum((hvi * vi).sum() for hvi, vi in zip(hv, v)).item()
            hv_norm = torch.sqrt(sum(hvi.pow(2).sum() for hvi in hv))

            if hv_norm < 1e-12:
                break
            v = tuple(hvi / hv_norm for hvi in hv)

        metrics = {
            "curvature/hessian_trace": hessian_trace,
            "curvature/top_eigenvalue": top_eigenvalue,
        }
    except Exception:
        Logger.get().warning("Curvature estimation failed; skipping.", exc_info=True)
    finally:
        model.train(was_training)

    return metrics
