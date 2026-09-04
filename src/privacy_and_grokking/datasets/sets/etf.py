import math
from typing import Literal

import torch
from pydantic import Field
from torch.utils.data import Dataset

from privacy_and_grokking.datasets.sets.base import (
    DataContainer,
    DatasetConfig,
    Normalization,
)


class ETFDataset(Dataset):
    def __init__(
        self,
        num_samples: int,
        num_classes: int,
        dimension: int,
        noise_std: float,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.dimension = dimension
        self.noise_std = noise_std

        if dimension < num_classes - 1:
            raise ValueError(
                f"Dimension ({dimension}) must be at least num_classes - 1 ({num_classes - 1}) for a Simplex ETF."
            )

        rng = torch.Generator().manual_seed(seed)

        # Construct Simplex ETF in R^K
        # M = sqrt(K / (K-1)) * (I - 1/K 1 1^T)
        I = torch.eye(num_classes)
        ones = torch.ones(num_classes, num_classes)
        # The columns of etf_K are the ETF vertices (shape: K x K)
        etf_K = math.sqrt(num_classes / (num_classes - 1)) * (I - (1.0 / num_classes) * ones)

        # Adjust dimension
        if dimension > num_classes:
            padded_etf = torch.zeros(dimension, num_classes)
            padded_etf[:num_classes, :] = etf_K

            # Generate random orthogonal matrix in R^dimension to rotate the ETF randomly
            random_matrix = torch.randn(dimension, dimension, generator=rng)
            Q, _ = torch.linalg.qr(random_matrix)

            self.class_means = Q @ padded_etf
        elif dimension == num_classes:
            self.class_means = etf_K
        else:  # dimension == num_classes - 1
            # SVD to drop the zero singular value dimension
            U, _, _ = torch.linalg.svd(etf_K)
            # The rank is K-1, so we take the first K-1 left singular vectors
            # Project ETF into K-1 dimensions
            self.class_means = U[:, :dimension].T @ etf_K

        # self.class_means is of shape (dimension, num_classes)

        # Generate samples
        self.labels = torch.randint(0, num_classes, (num_samples,), generator=rng)

        # Base points: shape (num_samples, dimension)
        self.features = self.class_means[:, self.labels].T

        # Add noise
        if noise_std > 0:
            noise = torch.randn(num_samples, dimension, generator=rng) * noise_std
            self.features += noise

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


class ETFConfig(DatasetConfig):
    name: Literal["etf"] = "etf"
    num_samples_train: int = Field(default=10000)
    num_samples_test: int = Field(default=2000)
    num_classes: int = Field(default=10)
    dimension: int = Field(default=10)
    noise_std: float = Field(default=0.0)
    seed: int = Field(default=42)

    def __call__(self) -> DataContainer:
        train = ETFDataset(
            num_samples=self.num_samples_train,
            num_classes=self.num_classes,
            dimension=self.dimension,
            noise_std=self.noise_std,
            seed=self.seed,
        )
        test = ETFDataset(
            num_samples=self.num_samples_test,
            num_classes=self.num_classes,
            dimension=self.dimension,
            noise_std=self.noise_std,
            seed=self.seed + 1,  # Different seed for test set noise
        )
        return DataContainer(
            train=train,
            test=test,
            num_classes=self.num_classes,
            input_shape=torch.Size([self.dimension]),
            normalization=Normalization(mean=[0.0], std=[1.0]),
        )
