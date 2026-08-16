import torch
from torch.utils.data import Dataset
from typing import Literal
from pydantic import Field

from privacy_and_grokking.datasets.sets.base import (
    DataContainer,
    DatasetConfig,
    Normalization,
)


class ModularAdditionDataset(Dataset):
    def __init__(self, p: int, train: bool, train_fraction: float = 0.3, seed: int = 42):
        self.p = p
        
        # generate all pairs (a, b)
        a = torch.arange(p).repeat_interleave(p)
        b = torch.arange(p).repeat(p)
        labels = (a + b) % p
        
        # inputs as sequences of one-hot vectors
        # token 1: a (one-hot, dim P+1)
        a_one_hot = torch.nn.functional.one_hot(a, num_classes=p+1).float()
        # token 2: b (one-hot, dim P+1)
        b_one_hot = torch.nn.functional.one_hot(b, num_classes=p+1).float()
        # token 3: '=' (index P, one-hot, dim P+1)
        eq_token = torch.full((p * p,), fill_value=p, dtype=torch.long)
        eq_one_hot = torch.nn.functional.one_hot(eq_token, num_classes=p+1).float()
        
        # stack into sequence [a, b, =] with shape (p*p, 3, p+1)
        features = torch.stack([a_one_hot, b_one_hot, eq_one_hot], dim=1)
        
        # shuffle and split
        rng = torch.Generator().manual_seed(seed)
        indices = torch.randperm(p * p, generator=rng)
        
        num_train = int(p * p * train_fraction)
        
        if train:
            self.indices = indices[:num_train]
        else:
            self.indices = indices[num_train:]
            
        self.features = features[self.indices]
        self.labels = labels[self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


class ModularAdditionConfig(DatasetConfig):
    name: Literal["modular_addition"] = "modular_addition"
    p: int = Field(default=113)
    train_fraction: float = Field(default=0.3)
    seed: int = Field(default=42)

    def __call__(self) -> DataContainer:
        train = ModularAdditionDataset(
            p=self.p, 
            train=True, 
            train_fraction=self.train_fraction, 
            seed=self.seed
        )
        test = ModularAdditionDataset(
            p=self.p, 
            train=False, 
            train_fraction=self.train_fraction, 
            seed=self.seed
        )
        
        return DataContainer(
            train=train,
            test=test,
            num_classes=self.p,
            input_shape=torch.Size([3, self.p + 1]),
            normalization=None,
        )
