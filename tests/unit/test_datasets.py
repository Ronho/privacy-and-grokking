import pytest
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from privacy_and_grokking.config import TrainConfig
from privacy_and_grokking.datasets import DatasetConfig
from privacy_and_grokking.datasets.canaries import (
    CanaryConfig,
    GaussianNoiseCanaryConfig,
    LabelNoiseCanaryConfig,
    OODNaturalCanaryConfig,
    SquareWatermarkCanaryConfig,
    UniformNoiseCanaryConfig,
)
from privacy_and_grokking.datasets.masking import PairedStratifiedMaskingConfig
from privacy_and_grokking.datasets.sets.base import CACHE_PATH, DataContainer, Normalization
from privacy_and_grokking.datasets.sets.cifar10 import CIFAR10Config
from privacy_and_grokking.datasets.sets.mnist import MNISTConfig
from privacy_and_grokking.datasets.sets.modular_addition import ModularAdditionConfig


class TestDatasetConfigs:
    def test_builds_complete_dataset_from_config(self, train_config: TrainConfig):
        dataset = train_config.data()

        assert isinstance(dataset, DataContainer)
        assert isinstance(dataset.train, Dataset)
        assert isinstance(dataset.test, Dataset)
        assert dataset.train_canary is None
        assert dataset.test_canary is None
        assert isinstance(dataset.num_classes, int)
        assert isinstance(dataset.input_shape, torch.Size)
        assert isinstance(dataset.normalization, Normalization) or dataset.normalization is None


class TestMNIST:
    @pytest.fixture
    def dataset_config_no_grokking(self) -> DatasetConfig:
        return DatasetConfig(
            data=MNISTConfig(),
            mask=PairedStratifiedMaskingConfig(
                num_models=6,
                p=0.5,
                seed=0,
                model_index=0,
            ),
            seed=0,
        )

    @pytest.fixture
    def dataset_config_grokking(self) -> DatasetConfig:
        return DatasetConfig(
            data=MNISTConfig(),
            mask=PairedStratifiedMaskingConfig(
                num_models=6,
                p=0.5,
                seed=0,
                model_index=0,
            ),
            train_size=2000,
            seed=0,
        )

    def test_dataset_config(
        self,
        train_config: TrainConfig,
        dataset_config_grokking: DatasetConfig,
        dataset_config_no_grokking: DatasetConfig,
    ):
        """Allows us to check that we cover every relevant case."""
        if train_config.data.data.name != "mnist":
            pytest.skip(f"Skipping non-MNIST config: {train_config.name}")
        assert (
            train_config.data == dataset_config_grokking
            or train_config.data == dataset_config_no_grokking
        )

    def test_dataset_config_grokking(self, dataset_config_grokking: DatasetConfig):
        dataset = dataset_config_grokking()

        assert dataset.input_shape == torch.Size([1, 28, 28])
        assert dataset.num_classes == 10
        assert dataset.normalization == Normalization(mean=[0.1307], std=[0.3081])
        assert dataset.train_canary is None
        assert dataset.test_canary is None

        assert len(dataset.train) == 1000
        assert len(dataset.test) == 10000

        class_counts = torch.bincount(torch.tensor([y for x, y in dataset.train]))
        assert torch.all(class_counts == 100)

    def test_dataset_config_no_grokking(self, dataset_config_no_grokking: DatasetConfig):
        dataset = dataset_config_no_grokking()

        assert dataset.input_shape == torch.Size([1, 28, 28])
        assert dataset.num_classes == 10
        assert dataset.normalization == Normalization(mean=[0.1307], std=[0.3081])
        assert dataset.train_canary is None
        assert dataset.test_canary is None

        assert len(dataset.train) == 25000
        assert len(dataset.test) == 10000

        # All classes equally represented
        class_counts = torch.bincount(torch.tensor([y for x, y in dataset.train]))
        assert torch.all(class_counts == 2500)

    @pytest.mark.parametrize(("a", "b"), [(0, 1), (2, 3), (4, 5)])
    @pytest.mark.parametrize(
        "config_name", ["dataset_config_no_grokking", "dataset_config_grokking"]
    )
    def test_dataset_config_no_overlap(
        self, a: int, b: int, config_name: str, request: pytest.FixtureRequest
    ):
        base_config: DatasetConfig = request.getfixturevalue(config_name)

        config_a = base_config.model_copy(deep=True)
        config_a.mask.model_index = a

        config_b = base_config.model_copy(deep=True)
        config_b.mask.model_index = b

        indices_a = set(config_a().train.indices)
        indices_b = set(config_b().train.indices)

        assert indices_a.isdisjoint(indices_b)

    @pytest.mark.parametrize(
        "canary_config",
        [
            GaussianNoiseCanaryConfig(num=100),
            UniformNoiseCanaryConfig(num=100),
            SquareWatermarkCanaryConfig(num=100, square_size=5),
            LabelNoiseCanaryConfig(num=100),
            OODNaturalCanaryConfig(num=100),
        ],
        ids=["gaussian_noise", "uniform_noise", "square_watermark", "label_noise", "ood_natural"],
    )
    @pytest.mark.parametrize(
        ("config_name", "expected_train_size", "expected_per_class"),
        [
            ("dataset_config_grokking", 1000, 100),
            ("dataset_config_no_grokking", 25000, 2500),
        ],
    )
    def test_canaries_distribution(
        self,
        canary_config: CanaryConfig,
        config_name: str,
        expected_train_size: int,
        expected_per_class: int,
        request: pytest.FixtureRequest,
    ):
        base_config: DatasetConfig = request.getfixturevalue(config_name)
        config = base_config.model_copy(deep=True)
        config.canary = canary_config

        dataset = config()

        # 1. Train-Daten: Gleiche Gesamtzahl an Traindaten (Rohdaten + Canaries)
        assert dataset.train_canary is not None
        assert len(dataset.train) + len(dataset.train_canary) == expected_train_size

        # Klassen in Train sind weiterhin gleich verteilt
        train_labels = torch.tensor([y for _, y in dataset.train])
        train_canary_labels = torch.tensor([y for _, y in dataset.train_canary])
        total_train_labels = torch.cat([train_labels, train_canary_labels])

        train_class_counts = torch.bincount(total_train_labels, minlength=10)
        assert torch.all(train_class_counts == expected_per_class)

        # 2. Test-Daten: Canaries kommen on top (test_canary)
        # und sind gleichverteilt über alle Klassen
        assert dataset.test_canary is not None
        assert len(dataset.test_canary) == canary_config.num

        test_canary_labels = torch.tensor([y for _, y in dataset.test_canary])
        test_canary_counts = torch.bincount(test_canary_labels, minlength=10)
        assert torch.all(test_canary_counts == canary_config.num // 10)

    def test_canary_dataloader_collate(self, dataset_config_grokking: DatasetConfig):
        config = dataset_config_grokking.model_copy(deep=True)
        config.canary = LabelNoiseCanaryConfig(num=100)
        dataset = config()
        ds = ConcatDataset([dataset.train, dataset.train_canary])
        loader = DataLoader(ds, batch_size=200, shuffle=True)
        for _, y in loader:
            assert isinstance(y, torch.Tensor)
            assert len(y) > 0


cifar10_canaries: list[CanaryConfig] = [
    GaussianNoiseCanaryConfig(num=100),
    UniformNoiseCanaryConfig(num=100),
    SquareWatermarkCanaryConfig(num=100, square_size=5),
    LabelNoiseCanaryConfig(num=100),
]
if (CACHE_PATH / "cifar-100-python").exists():
    cifar10_canaries.append(OODNaturalCanaryConfig(num=100))


class TestCIFAR10:
    @pytest.fixture
    def dataset_config_no_grokking(self) -> DatasetConfig:
        return DatasetConfig(
            data=CIFAR10Config(),
            mask=PairedStratifiedMaskingConfig(
                num_models=6,
                p=0.5,
                seed=0,
                model_index=0,
            ),
            seed=0,
        )

    @pytest.fixture
    def dataset_config_grokking(self) -> DatasetConfig:
        return DatasetConfig(
            data=CIFAR10Config(),
            mask=PairedStratifiedMaskingConfig(
                num_models=6,
                p=0.5,
                seed=0,
                model_index=0,
            ),
            train_size=2000,
            seed=0,
        )

    def test_dataset_config(
        self,
        train_config: TrainConfig,
        dataset_config_grokking: DatasetConfig,
        dataset_config_no_grokking: DatasetConfig,
    ):
        """Allows us to check that we cover every relevant case."""
        if train_config.data.data.name != "cifar10":
            pytest.skip(f"Skipping non-CIFAR10 config: {train_config.name}")
        assert (
            train_config.data == dataset_config_grokking
            or train_config.data == dataset_config_no_grokking
        )

    def test_dataset_config_grokking(self, dataset_config_grokking: DatasetConfig):
        dataset = dataset_config_grokking()

        assert dataset.input_shape == torch.Size([3, 32, 32])
        assert dataset.num_classes == 10
        assert dataset.normalization == Normalization(
            mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]
        )
        assert dataset.train_canary is None
        assert dataset.test_canary is None

        assert len(dataset.train) == 1000
        assert len(dataset.test) == 10000

        class_counts = torch.bincount(torch.tensor([y for x, y in dataset.train]))
        assert torch.all(class_counts == 100)

    def test_dataset_config_no_grokking(self, dataset_config_no_grokking: DatasetConfig):
        dataset = dataset_config_no_grokking()

        assert dataset.input_shape == torch.Size([3, 32, 32])
        assert dataset.num_classes == 10
        assert dataset.normalization == Normalization(
            mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]
        )
        assert dataset.train_canary is None
        assert dataset.test_canary is None

        assert len(dataset.train) == 25000
        assert len(dataset.test) == 10000

        # All classes equally represented
        class_counts = torch.bincount(torch.tensor([y for x, y in dataset.train]))
        assert torch.all(class_counts == 2500)

    @pytest.mark.parametrize(("a", "b"), [(0, 1), (2, 3), (4, 5)])
    @pytest.mark.parametrize(
        "config_name", ["dataset_config_no_grokking", "dataset_config_grokking"]
    )
    def test_dataset_config_no_overlap(
        self, a: int, b: int, config_name: str, request: pytest.FixtureRequest
    ):
        base_config: DatasetConfig = request.getfixturevalue(config_name)

        config_a = base_config.model_copy(deep=True)
        config_a.mask.model_index = a

        config_b = base_config.model_copy(deep=True)
        config_b.mask.model_index = b

        indices_a = set(config_a().train.indices)
        indices_b = set(config_b().train.indices)

        assert indices_a.isdisjoint(indices_b)

    @pytest.mark.parametrize(
        "canary_config",
        cifar10_canaries,
        ids=[c.name for c in cifar10_canaries],
    )
    @pytest.mark.parametrize(
        ("config_name", "expected_train_size", "expected_per_class"),
        [
            ("dataset_config_grokking", 1000, 100),
            ("dataset_config_no_grokking", 25000, 2500),
        ],
    )
    def test_canaries_distribution(
        self,
        canary_config: CanaryConfig,
        config_name: str,
        expected_train_size: int,
        expected_per_class: int,
        request: pytest.FixtureRequest,
    ):
        base_config: DatasetConfig = request.getfixturevalue(config_name)
        config = base_config.model_copy(deep=True)
        config.canary = canary_config

        dataset = config()

        # 1. Train-Daten: Gleiche Gesamtzahl an Traindaten (Rohdaten + Canaries)
        assert dataset.train_canary is not None
        assert len(dataset.train) + len(dataset.train_canary) == expected_train_size

        # Klassen in Train sind weiterhin gleich verteilt
        train_labels = torch.tensor([y for _, y in dataset.train])
        train_canary_labels = torch.tensor([y for _, y in dataset.train_canary])
        total_train_labels = torch.cat([train_labels, train_canary_labels])

        train_class_counts = torch.bincount(total_train_labels, minlength=10)
        assert torch.all(train_class_counts == expected_per_class)

        # 2. Test-Daten: Canaries kommen on top (test_canary)
        # und sind gleichverteilt über alle Klassen
        assert dataset.test_canary is not None
        assert len(dataset.test_canary) == canary_config.num

        test_canary_labels = torch.tensor([y for _, y in dataset.test_canary])
        test_canary_counts = torch.bincount(test_canary_labels, minlength=10)
        assert torch.all(test_canary_counts == canary_config.num // 10)

    def test_canary_dataloader_collate(self, dataset_config_grokking: DatasetConfig):
        config = dataset_config_grokking.model_copy(deep=True)
        config.canary = LabelNoiseCanaryConfig(num=100)
        dataset = config()
        ds = ConcatDataset([dataset.train, dataset.train_canary])
        loader = DataLoader(ds, batch_size=200, shuffle=True)
        for _, y in loader:
            assert isinstance(y, torch.Tensor)
            assert len(y) > 0


class TestModularAddition:
    @pytest.fixture
    def dataset_config_grokking(self) -> DatasetConfig:
        return DatasetConfig(
            data=ModularAdditionConfig(
                p=113,
                num_train_per_class=90,
                num_test_per_class=23,
            ),
            mask=PairedStratifiedMaskingConfig(
                num_models=6,
                p=0.5,
                seed=0,
                model_index=0,
            ),
            seed=0,
        )

    def test_dataset_config(
        self, train_config: TrainConfig, dataset_config_grokking: DatasetConfig
    ):
        """Allows us to check that we cover every relevant case."""
        if train_config.data.data.name != "modular_addition":
            pytest.skip(f"Skipping non-ModularAddition config: {train_config.name}")
        assert train_config.data == dataset_config_grokking

    def test_dataset_config_grokking(self, dataset_config_grokking: DatasetConfig):
        dataset = dataset_config_grokking()

        assert dataset.input_shape == torch.Size([3, 114])
        assert dataset.num_classes == 113
        assert dataset.normalization is None
        assert dataset.train_canary is None
        assert dataset.test_canary is None

        assert len(dataset.train) == 5085
        assert len(dataset.test) == 2599

        class_counts = torch.bincount(torch.tensor([y for x, y in dataset.train]))
        assert torch.all(class_counts == 45)

    @pytest.mark.parametrize(("a", "b"), [(0, 1), (2, 3), (4, 5)])
    def test_dataset_config_no_overlap(
        self, a: int, b: int, dataset_config_grokking: DatasetConfig
    ):
        config_a = dataset_config_grokking.model_copy(deep=True)
        config_a.mask.model_index = a

        config_b = dataset_config_grokking.model_copy(deep=True)
        config_b.mask.model_index = b

        indices_a = set(config_a().train.indices)
        indices_b = set(config_b().train.indices)

        assert indices_a.isdisjoint(indices_b)

    @pytest.mark.parametrize(
        "canary_config",
        [
            GaussianNoiseCanaryConfig(num=226),
            UniformNoiseCanaryConfig(num=226),
            SquareWatermarkCanaryConfig(num=226, square_size=5),
            LabelNoiseCanaryConfig(num=226),
        ],
        ids=["gaussian_noise", "uniform_noise", "square_watermark", "label_noise"],
    )
    def test_canaries_distribution(
        self,
        canary_config: CanaryConfig,
        dataset_config_grokking: DatasetConfig,
    ):
        config = dataset_config_grokking.model_copy(deep=True)
        config.canary = canary_config

        dataset = config()

        # 1. Train-Daten: Gleiche Gesamtzahl an Traindaten (Rohdaten + Canaries)
        assert dataset.train_canary is not None
        assert len(dataset.train) + len(dataset.train_canary) == 5085

        # Klassen in Train sind weiterhin gleich verteilt
        train_labels = torch.tensor([y for _, y in dataset.train])
        train_canary_labels = torch.tensor([y for _, y in dataset.train_canary])
        total_train_labels = torch.cat([train_labels, train_canary_labels])

        train_class_counts = torch.bincount(total_train_labels, minlength=113)
        assert torch.all(train_class_counts == 45)

        # 2. Test-Daten: Canaries kommen on top (test_canary)
        # und sind gleichverteilt über alle Klassen
        assert dataset.test_canary is not None
        assert len(dataset.test_canary) == canary_config.num

        test_canary_labels = torch.tensor([y for _, y in dataset.test_canary])
        test_canary_counts = torch.bincount(test_canary_labels, minlength=113)
        assert torch.all(test_canary_counts == canary_config.num // 113)

    def test_canary_dataloader_collate(self, dataset_config_grokking: DatasetConfig):
        config = dataset_config_grokking.model_copy(deep=True)
        config.canary = LabelNoiseCanaryConfig(num=226)
        dataset = config()
        ds = ConcatDataset([dataset.train, dataset.train_canary])
        loader = DataLoader(ds, batch_size=200, shuffle=True)
        for _, y in loader:
            assert isinstance(y, torch.Tensor)
            assert len(y) > 0
