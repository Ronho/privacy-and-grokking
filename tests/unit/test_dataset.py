import pytest
import torch

from privacy_and_grokking.datasets import (
    DatasetConfig,
    Datasets,
    MaskingConfig,
    Maskings,
    UniformNoiseCanaryConfig,
    create_masking,
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


@pytest.mark.parametrize("masking_name", list(Maskings))
def test_create_masking_is_reproducible(masking_name):
    cfg = MaskingConfig(name=masking_name, num_models=256, p=0.5, seed=1)
    masking_1 = create_masking(config=cfg, num_samples=1000, num_classes=10)
    masking_2 = create_masking(config=cfg, num_samples=1000, num_classes=10)
    mask_1 = masking_1()
    mask_2 = masking_2()
    assert torch.equal(mask_1, mask_2)


@pytest.mark.parametrize("masking_name", list(Maskings))
def test_create_masking_called_twice_returns_different_masks(masking_name):
    cfg = MaskingConfig(name=masking_name, num_models=256, p=0.5, seed=1)
    masking = create_masking(config=cfg, num_samples=1000, num_classes=10)
    mask_1 = masking()
    mask_2 = masking()
    assert not torch.equal(mask_1, mask_2)
