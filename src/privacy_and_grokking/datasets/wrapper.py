import torch

from torch.utils.data import Dataset, ConcatDataset, Subset, TensorDataset
from .canaries import Canary, create_canaries
from .datasets import Data, create_dataset


def stratified_split(dataset: Dataset, num_classes: int, train_ratio: float) -> tuple[Dataset, Dataset]:
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be between 0 and 1.")

    indices = {c: [] for c in range(num_classes)}
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        indices[int(label)].append(idx)

    train_indices = []
    val_indices = []
    for idxs in indices.values():
        split_point = int(len(idxs) * train_ratio)
        train_indices.extend(idxs[:split_point])
        val_indices.extend(idxs[split_point:])

    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)
    return train_subset, val_subset

def create_subset(size: int | None, dataset: Dataset, num_classes: int) -> Subset:
    if size is None:
        subset_indices = list(range(len(dataset)))
    else:
        if size > len(dataset):
            raise ValueError("Requested subset size exceeds dataset size.")

        samples_per_class = size // num_classes
        indices = {c: [] for c in range(num_classes)}
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            if len(indices[int(label)]) < samples_per_class:
                indices[int(label)].append(idx)
            if all(len(idxs) >= samples_per_class for idxs in indices.values()):
                break
        subset_indices = [idx for idxs in indices.values() for idx in idxs]

    return Subset(dataset, subset_indices)

def get_dataset(name: Data, train_ratio: float, train_size: int | None, canary: Canary | None = None, **kwargs) -> tuple[Dataset, Dataset, Dataset, torch.Size, int]:
    container = create_dataset(name)
    train, val = stratified_split(container["trainval"], container["num_classes"], train_ratio=train_ratio)
    subset = create_subset(train_size, train, container["num_classes"])
    if canary is None:
        canary_dataset = TensorDataset(torch.empty(0, *container["input_shape"]), torch.empty(0))
    else:
        canary_dataset = create_canaries(name=canary, dataset=subset, num_classes=container["num_classes"], **kwargs)
        
    train = ConcatDataset([subset, canary_dataset])

    return train, val, container["test"], container["input_shape"], container["num_classes"]
