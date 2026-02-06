import torch

from privacy_and_grokking.datasets import (
    DatasetConfig,
    Datasets,
    UniformNoiseCanaryConfig,
    generate_datasets,
)


def test_get_dataset_produces_correctly_sized_datasets():
    cfg = DatasetConfig(
        name=Datasets.MNIST,
        train_size=None,
        canary_share=0,
        canary_config=None,
        seed=1,
    )
    train, test = generate_datasets(config=cfg)
    assert len(train) == 60_000
    assert len(test) == 10_000


def test_get_dataset_is_reproducible():
    cfg = DatasetConfig(
        name=Datasets.MNIST,
        train_size=1000,
        canary_share=0.1,
        canary_config=UniformNoiseCanaryConfig(),
        seed=1,
    )

    train1, test1 = generate_datasets(config=cfg)
    train2, test2 = generate_datasets(config=cfg)

    for (data1, label1), (data2, label2) in zip(train1, train2, strict=True):
        assert torch.equal(data1, data2)
        assert label1 == label2
    for (data1, label1), (data2, label2) in zip(test1, test2, strict=True):
        assert torch.equal(data1, data2)
        assert label1 == label2
