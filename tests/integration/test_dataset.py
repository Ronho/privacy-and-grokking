import torch

from privacy_and_grokking.datasets import get_dataset


def test_get_dataset_produces_correctly_sized_datasets():
    train, val, test = get_dataset(
        name="mnist",
        train_ratio=0.5,
        train_size=1000,
        canary="gaussian_noise",
        percentage=1.0,
        repetitions=1,
        seed=1,
        noise_scale=1.0
    )
    assert len(train) == 1_010
    assert len(val) == 30_003
    assert len(test) == 10_000

def test_get_dataset_is_reproducible():
    train1, val1, test1 = get_dataset(
        name="mnist",
        train_ratio=0.5,
        train_size=1000,
        canary="gaussian_noise",
        percentage=1.0,
        repetitions=1,
        seed=1,
        noise_scale=1.0
    )
    train2, val2, test2 = get_dataset(
        name="mnist",
        train_ratio=0.5,
        train_size=1000,
        canary="gaussian_noise",
        percentage=1.0,
        repetitions=1,
        seed=1,
        noise_scale=1.0
    )
    for (data1, label1), (data2, label2) in zip(train1, train2):
        assert torch.equal(data1, data2)
        assert label1 == label2
    for (data1, label1), (data2, label2) in zip(val1, val2):
        assert torch.equal(data1, data2)
        assert label1 == label2
    for (data1, label1), (data2, label2) in zip(test1, test2):
        assert torch.equal(data1, data2)
        assert label1 == label2

def test_get_dataset_different_seeds_produce_different_canaries():
    train1, _, _ = get_dataset(
        name="mnist",
        train_ratio=0.5,
        train_size=1000,
        canary="gaussian_noise",
        percentage=1.0,
        repetitions=1,
        seed=1,
        noise_scale=1.0
    )
    train2, _, _ = get_dataset(
        name="mnist",
        train_ratio=0.5,
        train_size=1000,
        canary="gaussian_noise",
        percentage=1.0,
        repetitions=1,
        seed=2,
        noise_scale=1.0
    )
    canary_data1 = [data for idx, (data, _)in enumerate(train1) if idx >= 1000]
    canary_data2 = [data for idx, (data, _) in enumerate(train2) if idx >= 1000]
    differences_found = any(not torch.equal(d1, d2) for d1, d2 in zip(canary_data1, canary_data2))
    assert differences_found, "Canary datasets are identical despite different seeds"
