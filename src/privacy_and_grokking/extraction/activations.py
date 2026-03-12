import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from privacy_and_grokking.utils import get_device


def extract_all_layer_activations(
    model: nn.Module,
    dataset,
    batch_size: int = 256,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Capture the **output** of every named ``nn.Linear`` in *model*.

    Returns a tuple of ``(layer_activations_dict, labels)`` where
    ``layer_activations_dict`` maps each linear-module name to a tensor of
    shape ``(N, out_features)``.  Layers are ordered by their appearance in
    ``model.named_modules()``.
    """
    dataloader = DataLoader(dataset, batch_size=batch_size)
    device = get_device()

    buffers: dict[str, list[torch.Tensor]] = {}
    label_list: list[torch.Tensor] = []
    handles: list = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            key = name
            buffers[key] = []

            def _make_hook(k: str):
                def _hook(_module: nn.Module, _inp: tuple, output: torch.Tensor) -> None:
                    buffers[k].append(output.detach().cpu())

                return _hook

            handles.append(module.register_forward_hook(_make_hook(key)))

    try:
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(device)
                model(x)
                label_list.append(y)
    finally:
        for h in handles:
            h.remove()

    return (
        {k: torch.cat(v, dim=0) for k, v in buffers.items()},
        torch.cat(label_list, dim=0),
    )


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
