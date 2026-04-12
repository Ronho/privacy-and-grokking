import torch


def get_optimizer_internals(optimizer) -> dict[str, float]:
    stats = {}

    for group_idx, group in enumerate(optimizer.param_groups):
        for p_idx, p in enumerate(group["params"]):
            state = optimizer.state[p]

            if not state:
                continue

            param_prefix = f"g{group_idx}_p{p_idx}"

            for key, value in state.items():
                # We only care about tensors (momentum, variance, etc.)
                if torch.is_tensor(value):
                    # Log scalars directly, or stats for higher-dimensional tensors
                    if value.numel() == 1:
                        stats[f"{param_prefix}/{key}"] = value.item()
                    else:
                        stats[f"{param_prefix}/{key}_mean"] = value.mean().item()
                        stats[f"{param_prefix}/{key}_std"] = value.std().item()
                        stats[f"{param_prefix}/{key}_norm"] = value.norm().item()

                elif isinstance(value, (int, float)):
                    stats[f"{param_prefix}/{key}"] = value

    return stats
