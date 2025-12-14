import torch

from torchvision import datasets, transforms
from typing import Literal, Sequence

from ..path_keeper import get_path_keeper

type Size = Literal["small", "full"]
type Canary = Literal["gaussian_noise"]

class Dataset:
    def __init__(self, size: Size, num_classes: int):
        self.size = size
        self.num_classes = num_classes

class GaussianNoiseCanary:
    """ ONLY FOR MNIST FOR NOW (since using int) """
    def __init__(self, dim: Sequence[int], noise_scale: float, seed: int | None = 64):
        self.dim = dim
        self.noise_scale = noise_scale
        self.seed = seed
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def generate(self, num: int) -> torch.Tensor:
        return torch.randn((num, *self.dim), generator=self.generator) * self.noise_scale

class MNIST(Dataset):
    def __init__(self, size: Size, canary: Canary | None = None):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)), # MNIST mean/std
        ])
        pk = get_path_keeper()
        train = datasets.MNIST(
            root=pk.CACHE,
            train=True, 
            transform=self.transform,
            download=True
        )
        self.test = datasets.MNIST(
            root=pk.CACHE,
            train=False, 
            transform=self.transform,
            download=True
        )
        super().__init__(size=size, num_classes=len(train.classes))

        if size == "small":
            SAMPLE_SIZE = 1000
            samples_per_class = SAMPLE_SIZE // self.num_classes
            indices = {c: [] for c in range(self.num_classes)}
            for idx, label in enumerate(train.targets):
                label = label.item()
                if len(indices[label]) < samples_per_class:
                    indices[label].append(idx)
                if all(len(idxs) >= samples_per_class for idxs in indices.values()):
                    break
            self.train = torch.utils.data.Subset(train, [idx for idxs in indices.values() for idx in idxs])
        else:
            self.train = train

        if canary == "gaussian_noise":
            # Wrap to ensure labels are tensors (MNIST returns ints by default)
            class TensorWrapper(torch.utils.data.Dataset):
                def __init__(self, dataset):
                    self.dataset = dataset
                
                def __len__(self):
                    return len(self.dataset)
                
                def __getitem__(self, idx):
                    x, y = self.dataset[idx]
                    if not isinstance(y, torch.Tensor):
                        y = torch.tensor(y, dtype=torch.long)
                    return x, y
            
            canary_generator = GaussianNoiseCanary(dim=(1, 28, 28), noise_scale=1.0)
            num_canaries = int(len(self.train) * 0.001)  # 0.1% of training data

            canary_data = canary_generator.generate(num_canaries)
            canary_labels = torch.tensor([i % self.num_classes for i in range(num_canaries)], dtype=torch.long)

            # Repeat canaries to increase their presence in the dataset: 1% are canaries now.
            canary_data = canary_data.repeat(10, 1, 1, 1)
            canary_labels = canary_labels.repeat(10)

            self.canary_dataset = torch.utils.data.TensorDataset(canary_data, canary_labels)
            self.train = torch.utils.data.ConcatDataset([TensorWrapper(self.train), self.canary_dataset])
    
    def __call__(self):
        return self.train, self.test
