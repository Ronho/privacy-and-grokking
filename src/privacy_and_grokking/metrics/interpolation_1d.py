import torch
from typing import Any

def compute_state_dict_l2_norm(state_dict: dict[str, Any]) -> float:
    """Computes the global L2 norm of a PyTorch state dict."""
    total_norm_sq = 0.0
    for tensor in state_dict.values():
        if isinstance(tensor, torch.Tensor) and tensor.is_floating_point():
            total_norm_sq += tensor.float().square().sum().item()
    return total_norm_sq ** 0.5

def linear_interpolate_state_dicts(
    state_a: dict[str, Any],
    state_b: dict[str, Any],
    alpha: float
) -> dict[str, Any]:
    """Linearly interpolates between two state dicts: (1 - alpha) * state_a + alpha * state_b."""
    interp_state = {}
    for key in state_a:
        if key in state_b and isinstance(state_a[key], torch.Tensor) and state_a[key].is_floating_point():
            interp_state[key] = (1.0 - alpha) * state_a[key] + alpha * state_b[key]
        else:
            interp_state[key] = state_a[key]
    return interp_state
