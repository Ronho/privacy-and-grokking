import torch

from torch.utils.data import Dataset, TensorDataset
from typing import Sequence, Literal


type CanaryTuple = tuple[torch.Tensor, list[int]]
type Canary = Literal["gaussian_noise"]

def gaussian_noise(num_canaries: int, dim: Sequence[int], num_classes: int, noise_scale: float, seed: int | None, device: torch.device) -> CanaryTuple:
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
    
    data = torch.randn((num_canaries, *dim), generator=generator) * noise_scale
    labels = [i % num_classes for i in range(num_canaries)]
    return data, labels

def create_canaries(name: Canary, dataset: Dataset, num_classes: int, device: torch.device, percentage: float, repetitions: int, **kwargs) -> Dataset:
    """Create a canary dataset based on the specified canary type that represents 1% of the given dataset.
    
    The total canaries generated are determined by the `percentage` parameter. Each canary is repeated `repetitions` times.
    """

    num_canaries = int(len(dataset) * (percentage/100))
    dim = dataset[0][0].shape

    match name.lower():
        case "gaussian_noise":
            data, labels = gaussian_noise(num_canaries=num_canaries, dim=dim, num_classes=num_classes, device=device, **kwargs)
        case _:
            raise ValueError(f"Unknown canary: {name}")

    canary_data = data.repeat(repetitions, 1, 1, 1)
    canary_labels = labels * repetitions
    canary_dataset = TensorDataset(canary_data, torch.tensor(canary_labels))

    return canary_dataset