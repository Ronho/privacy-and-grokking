from collections.abc import Sequence
from typing import Literal

import torch
from torch.utils.data import Dataset, TensorDataset

type CanaryTuple = tuple[torch.Tensor, list[int]]
type Canary = Literal["gaussian_noise", "watermark"]


def gaussian_noise(
    num_canaries: int, dim: Sequence[int], num_classes: int, noise_scale: float, seed: int | None
) -> CanaryTuple:
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    data = torch.randn((num_canaries, *dim), generator=generator) * noise_scale
    labels = [i % num_classes for i in range(num_canaries)]
    return data, labels


def watermark(
    num_canaries: int,
    dim: Sequence[int],
    num_classes: int,
    dataset: Dataset,
    square_size: int,
    seed: int | None,
) -> CanaryTuple:
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    indices = torch.randint(0, len(dataset), (num_canaries,), generator=generator)
    data = torch.stack([dataset[i][0] for i in indices])
    original_labels = torch.tensor([dataset[i][1] for i in indices])

    height = dim[-2]
    width = dim[-1]

    square_size = min(square_size, height, width)

    data[:, :, -square_size:, -square_size:] = 1.0

    if num_classes > 1:
        offsets = torch.randint(1, num_classes, (num_canaries,), generator=generator)
        labels_tensor = (original_labels + offsets) % num_classes
        labels = labels_tensor.tolist()
    else:
        labels = [0] * num_canaries

    return data, labels


def create_canaries(
    name: Canary, dataset: Dataset, num_classes: int, percentage: float, repetitions: int, **kwargs
) -> TensorDataset:
    """Create a canary dataset based on the specified canary type that represents 1% of the given dataset.

    The total canaries generated are determined by the `percentage` parameter. Each canary is repeated `repetitions` times.
    """

    num_canaries = int(len(dataset) * (percentage / 100))
    dim = dataset[0][0].shape

    match name.lower():
        case "gaussian_noise":
            data, labels = gaussian_noise(
                num_canaries=num_canaries, dim=dim, num_classes=num_classes, **kwargs
            )
        case "watermark":
            data, labels = watermark(
                num_canaries=num_canaries,
                dim=dim,
                num_classes=num_classes,
                dataset=dataset,
                **kwargs,
            )
        case _:
            raise ValueError(f"Unknown canary: {name}")

    canary_data = data.repeat(repetitions, 1, 1, 1)
    canary_labels = labels * repetitions
    canary_dataset = TensorDataset(canary_data, torch.tensor(canary_labels))

    return canary_dataset
