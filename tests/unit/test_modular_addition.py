import torch

from privacy_and_grokking.datasets.masking.paired_stratified import PairedStratifiedMasking
from privacy_and_grokking.datasets.sets.modular_addition import (
    ModularAdditionConfig,
    ModularAdditionDataset,
)


def test_modular_addition_defaults():
    # Test default behavior
    dataset_train = ModularAdditionDataset(p=113, train=True, train_fraction=0.3, seed=42)
    dataset_test = ModularAdditionDataset(p=113, train=False, train_fraction=0.3, seed=42)

    expected_train_per_class = int(113 * 0.3)  # 33
    expected_test_per_class = 113 - expected_train_per_class  # 80

    assert len(dataset_train) == 113 * expected_train_per_class
    assert len(dataset_test) == 113 * expected_test_per_class

    train_counts = torch.bincount(dataset_train.labels)
    test_counts = torch.bincount(dataset_test.labels)

    assert (train_counts == expected_train_per_class).all()
    assert (test_counts == expected_test_per_class).all()


def test_modular_addition_explicit_counts():
    # Test explicit train/test size per class
    num_train_per_class = 90
    num_test_per_class = 23
    p = 113

    train_dataset = ModularAdditionDataset(
        p=p,
        train=True,
        num_train_per_class=num_train_per_class,
        num_test_per_class=num_test_per_class,
        seed=42,
    )
    test_dataset = ModularAdditionDataset(
        p=p,
        train=False,
        num_train_per_class=num_train_per_class,
        num_test_per_class=num_test_per_class,
        seed=42,
    )

    assert len(train_dataset) == p * num_train_per_class
    assert len(test_dataset) == p * num_test_per_class

    train_counts = torch.bincount(train_dataset.labels)
    test_counts = torch.bincount(test_dataset.labels)

    assert (train_counts == num_train_per_class).all()
    assert (test_counts == num_test_per_class).all()

    # Apply paired_stratified mask (50% split)
    masking = PairedStratifiedMasking(
        num_samples=len(train_dataset), num_classes=p, num_models=2, p=0.5, seed=42
    )
    mask = masking._generate_mask(train_dataset.labels)

    # Twins should partition classes perfectly
    twin1_labels = train_dataset.labels[mask[:, 0]]
    twin2_labels = train_dataset.labels[mask[:, 1]]

    twin1_counts = torch.bincount(twin1_labels)
    twin2_counts = torch.bincount(twin2_labels)

    assert (twin1_counts == num_train_per_class // 2).all()
    assert (twin2_counts == num_train_per_class // 2).all()


def test_modular_addition_config_validation():
    # Test config creates expected datasets
    config = ModularAdditionConfig(
        p=113,
        num_train_per_class=90,
        num_test_per_class=23,
    )
    container = config()

    assert len(container.train) == 113 * 90
    assert len(container.test) == 113 * 23
