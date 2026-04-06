import torch
import torch.nn as nn


def compute_weight_norms(model: nn.Module) -> dict[str, float]:
    """Compute per-parameter and total L2 weight norms."""
    norms: dict[str, float] = {}
    all_params: list[torch.Tensor] = []
    for name, param in model.named_parameters():
        p = param.detach().float().flatten()
        norms[f"weight_norm/{name}"] = torch.linalg.norm(p).item()
        all_params.append(p)
    norms["weight_norm/total"] = (
        torch.linalg.norm(torch.cat(all_params)).item() if all_params else 0.0
    )
    return norms

def compute_gradient_norms(model: nn.Module) -> dict[str, float]:
    """Compute per-parameter and total L2 gradient norms."""
    norms: dict[str, float] = {}
    all_grads: list[torch.Tensor] = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            g = param.grad.detach().float().flatten()
            norms[f"grad_norm/{name}"] = torch.linalg.norm(g).item()
            all_grads.append(g)
    norms["grad_norm/total"] = (
        torch.linalg.norm(torch.cat(all_grads)).item() if all_grads else 0.0
    )
    return norms
