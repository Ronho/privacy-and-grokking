from collections.abc import Sequence
from typing import Literal

import torch
from torchvision import datasets, transforms

from privacy_and_grokking.datasets.canaries.base import Canary, CanaryConfig
from privacy_and_grokking.datasets.sets.base import CACHE_PATH


class OODNaturalCanary:
    def __init__(self, dim: Sequence[int]):
        self.dim = dim
        CACHE_PATH.mkdir(exist_ok=True)

        transform_list = []

        if dim[-3] == 3 and dim[-1] == 32:
            # CIFAR-10 -> CIFAR-100
            transform_list.append(transforms.ToTensor())
            transform = transforms.Compose(transform_list)
            self.dataset = datasets.CIFAR100(
                root=CACHE_PATH, train=True, download=True, transform=transform
            )
        elif dim[-3] == 1 and dim[-1] in (28, 32):
            # MNIST -> FashionMNIST
            if dim[-1] == 32:
                transform_list.append(transforms.Pad(2))
            transform_list.append(transforms.ToTensor())
            transform = transforms.Compose(transform_list)
            self.dataset = datasets.FashionMNIST(
                root=CACHE_PATH, train=True, download=True, transform=transform
            )
        else:
            raise ValueError(f"No default OOD dataset configured for dimension {dim}")

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        seed = torch.hash_tensor(image)
        index = abs(seed.item()) % len(self.dataset)
        ood_image, _ = self.dataset[index]
        return ood_image


class OODNaturalCanaryConfig(CanaryConfig):
    name: Literal["ood_natural"] = "ood_natural"

    def __call__(self, dim: Sequence[int]) -> Canary:
        return OODNaturalCanary(dim=dim)
