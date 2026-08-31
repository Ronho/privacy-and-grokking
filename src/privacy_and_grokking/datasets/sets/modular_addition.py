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
    def __init__(
        self,
        p: int,
        train: bool,
        num_train_per_class: int,
        num_test_per_class: int,
        seed: int = 42,
    ):
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
        
        # Stratified train/test split
        rng = torch.Generator().manual_seed(seed)
        
        train_indices_list = []
        test_indices_list = []
        
        # We know each class has exactly p samples
        
        for c in range(p):
            class_indices = (labels == c).nonzero().view(-1)
            # Shuffle indices for this class
            perm = torch.randperm(len(class_indices), generator=rng)
            shuffled_indices = class_indices[perm]
            
            train_indices_list.append(shuffled_indices[:num_train_per_class])
            test_indices_list.append(shuffled_indices[num_train_per_class : num_train_per_class + num_test_per_class])
            
        if train:
            self.indices = torch.cat(train_indices_list)
        else:
            self.indices = torch.cat(test_indices_list)
            
        self.features = features[self.indices]
        self.labels = labels[self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


class ModularAdditionConfig(DatasetConfig):
    name: Literal["modular_addition"] = "modular_addition"
    p: int = Field(default=113)
    num_train_per_class: int | None = Field(default=None)
    num_test_per_class: int | None = Field(default=None)
    seed: int = Field(default=42)

    def __call__(self) -> DataContainer:
        train = ModularAdditionDataset(
            p=self.p, 
            train=True,
            num_train_per_class=self.num_train_per_class,
            num_test_per_class=self.num_test_per_class,
            seed=self.seed
        )
        test = ModularAdditionDataset(
            p=self.p, 
            train=False,
            num_train_per_class=self.num_train_per_class,
            num_test_per_class=self.num_test_per_class,
            seed=self.seed
        )
        
        return DataContainer(
            train=train,
            test=test,
            num_classes=self.p,
            input_shape=torch.Size([3, self.p + 1]),
            normalization=None,
        )
