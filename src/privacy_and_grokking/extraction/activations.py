import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from privacy_and_grokking.utils import get_device


def extract_penultimate_activations(
    model: nn.Module,
    dataset,
    batch_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    dataloader = DataLoader(dataset, batch_size=batch_size)
    device = get_device()

    collected: list[torch.Tensor] = []
    label_list: list[torch.Tensor] = []

    def _hook(_module: nn.Module, inp: tuple[torch.Tensor, ...], _output: torch.Tensor):
        collected.append(inp[0].detach().cpu())

    handle = model.last_layer.register_forward_hook(_hook)

    try:
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(device)
                model(x)
                label_list.append(y)
    finally:
        handle.remove()

    return torch.cat(collected, dim=0), torch.cat(label_list, dim=0)
