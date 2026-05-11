import pytest
import torch
from torch.utils.data import Dataset

from privacy_and_grokking.datasets.base import (
    CanaryDataset,
    DatasetConfig,
    distribute_a_across_b,
)
from privacy_and_grokking.datasets.canaries import (
    SquareWatermarkCanary,
    SquareWatermarkCanaryConfig,
    UniformNoiseCanary,
    UniformNoiseCanaryConfig,
    create_canary_generator,
)
from privacy_and_grokking.datasets.canary_class_assignment import (
    alternative_derange_balanced_indices,
    derange_balanced_indices,
    random_derange_indices,
)
from privacy_and_grokking.datasets.gpu import GpuDataset
from privacy_and_grokking.datasets.masking import Mask
from privacy_and_grokking.datasets.masking.balanced_stratified import (
    BalancedStratifiedMasking,
    BalancedStratifiedMaskingConfig,
)
from privacy_and_grokking.datasets.masking.independent_stratified import (
    IndependentStratifiedMasking,
    IndependentStratifiedMaskingConfig,
)
from privacy_and_grokking.datasets.masking.partitioned_stratified import (
    PartitionedStratifiedMasking,
    PartitionedStratifiedMaskingConfig,
)
from privacy_and_grokking.datasets.masking.uniform import (
    UniformMasking,
    UniformMaskingConfig,
)

# --- Helpers ---


class FakeDataset(Dataset):
    """A simple in-memory dataset for testing."""

    def __init__(
        self, num_samples: int, num_classes: int, input_shape: tuple[int, ...] = (1, 8, 8)
    ):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.input_shape = input_shape
        torch.manual_seed(0)
        self.images = torch.randn(num_samples, *input_shape)
        self.labels = torch.arange(num_samples) % num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx].item()


NUM_SAMPLES = 100
NUM_CLASSES = 5
INPUT_SHAPE = (1, 8, 8)


# --- Tests for distribute_a_across_b ---


class TestDistributeAcrossB:
    def test_sum_equals_a(self):
        result = distribute_a_across_b(10, 3)
        assert result.sum().item() == 10

    def test_even_distribution(self):
        result = distribute_a_across_b(12, 4)
        assert torch.all(result == 3)

    def test_remainder_distributed(self):
        result = distribute_a_across_b(10, 3)
        # 10 // 3 = 3, remainder 1 -> first bucket gets 4
        assert result[0].item() == 4
        assert result[1].item() == 3
        assert result[2].item() == 3

    def test_a_less_than_b(self):
        result = distribute_a_across_b(2, 5)
        assert result.sum().item() == 2
        assert (result <= 1).all()

    def test_zero_a(self):
        result = distribute_a_across_b(0, 3)
        assert result.sum().item() == 0


# --- Tests for Masking Base ---


class TestMaskingBase:
    def test_invalid_p_raises(self):
        with pytest.raises(ValueError, match="p must be between 0 and 1"):
            UniformMasking(num_samples=100, num_classes=5, num_models=2, p=1.5, seed=42)

    def test_classes_length_mismatch_raises(self):
        masking = UniformMasking(num_samples=100, num_classes=5, num_models=2, p=0.5, seed=42)
        wrong_classes = torch.zeros(50, dtype=torch.long)
        with pytest.raises(ValueError, match="Length of classes must match num_samples"):
            masking(classes=wrong_classes)

    def test_no_classes_generates_even_distribution(self):
        masking = UniformMasking(num_samples=100, num_classes=5, num_models=2, p=0.5, seed=42)
        mask = masking(classes=None)
        assert mask.shape == (100, 2)


# --- Tests for UniformMasking ---


class TestUniformMasking:
    def test_output_shape(self):
        masking = UniformMasking(
            num_samples=NUM_SAMPLES, num_classes=NUM_CLASSES, num_models=3, p=0.5, seed=42
        )
        classes = torch.arange(NUM_SAMPLES) % NUM_CLASSES
        mask = masking(classes=classes)
        assert mask.shape == (NUM_SAMPLES, 3)
        assert mask.dtype == torch.bool

    def test_approximate_proportion(self):
        masking = UniformMasking(
            num_samples=1000, num_classes=NUM_CLASSES, num_models=2, p=0.5, seed=42
        )
        classes = torch.arange(1000) % NUM_CLASSES
        mask = masking(classes=classes)
        # Each model should have approximately 50% of samples
        for model_idx in range(2):
            proportion = mask[:, model_idx].float().mean().item()
            assert 0.4 < proportion < 0.6

    def test_deterministic_with_seed(self):
        kwargs = {
            "num_samples": NUM_SAMPLES,
            "num_classes": NUM_CLASSES,
            "num_models": 2,
            "p": 0.5,
            "seed": 123,
        }
        classes = torch.arange(NUM_SAMPLES) % NUM_CLASSES
        mask1 = UniformMasking(**kwargs)(classes=classes)
        mask2 = UniformMasking(**kwargs)(classes=classes)
        assert torch.equal(mask1, mask2)

    def test_config_creates_masking(self):
        cfg = UniformMaskingConfig(name="uniform", num_models=2, p=0.5, seed=42)
        masking = cfg(num_samples=NUM_SAMPLES, num_classes=NUM_CLASSES)
        assert isinstance(masking, UniformMasking)


# --- Tests for IndependentStratifiedMasking ---


class TestIndependentStratifiedMasking:
    def test_output_shape(self):
        masking = IndependentStratifiedMasking(
            num_samples=NUM_SAMPLES, num_classes=NUM_CLASSES, num_models=3, p=0.5, seed=42
        )
        classes = torch.arange(NUM_SAMPLES) % NUM_CLASSES
        mask = masking(classes=classes)
        assert mask.shape == (NUM_SAMPLES, 3)

    def test_stratified_selection(self):
        masking = IndependentStratifiedMasking(
            num_samples=NUM_SAMPLES, num_classes=NUM_CLASSES, num_models=2, p=0.5, seed=42
        )
        classes = torch.arange(NUM_SAMPLES) % NUM_CLASSES
        mask = masking(classes=classes)
        # Each model should select ~50% from each class
        for model_idx in range(2):
            for c in range(NUM_CLASSES):
                class_mask = classes == c
                selected = mask[class_mask, model_idx].sum().item()
                total_in_class = class_mask.sum().item()
                expected = int(total_in_class * 0.5)
                assert selected == expected

    def test_deterministic_with_seed(self):
        kwargs = {
            "num_samples": NUM_SAMPLES,
            "num_classes": NUM_CLASSES,
            "num_models": 2,
            "p": 0.5,
            "seed": 99,
        }
        classes = torch.arange(NUM_SAMPLES) % NUM_CLASSES
        mask1 = IndependentStratifiedMasking(**kwargs)(classes=classes)
        mask2 = IndependentStratifiedMasking(**kwargs)(classes=classes)
        assert torch.equal(mask1, mask2)

    def test_config_creates_masking(self):
        cfg = IndependentStratifiedMaskingConfig(
            name="independent_stratified", num_models=2, p=0.5, seed=42
        )
        masking = cfg(num_samples=NUM_SAMPLES, num_classes=NUM_CLASSES)
        assert isinstance(masking, IndependentStratifiedMasking)


# --- Tests for PartitionedStratifiedMasking ---


class TestPartitionedStratifiedMasking:
    def test_output_shape(self):
        masking = PartitionedStratifiedMasking(
            num_samples=NUM_SAMPLES, num_classes=NUM_CLASSES, num_models=2, p=0.5, seed=42
        )
        classes = torch.arange(NUM_SAMPLES) % NUM_CLASSES
        mask = masking(classes=classes)
        assert mask.shape == (NUM_SAMPLES, 2)

    def test_partitions_are_disjoint(self):
        masking = PartitionedStratifiedMasking(
            num_samples=NUM_SAMPLES, num_classes=NUM_CLASSES, num_models=2, p=0.5, seed=42
        )
        classes = torch.arange(NUM_SAMPLES) % NUM_CLASSES
        mask = masking(classes=classes)
        # No sample should be in both models
        overlap = (mask[:, 0] & mask[:, 1]).sum().item()
        assert overlap == 0

    def test_all_samples_assigned(self):
        masking = PartitionedStratifiedMasking(
            num_samples=NUM_SAMPLES, num_classes=NUM_CLASSES, num_models=2, p=0.5, seed=42
        )
        classes = torch.arange(NUM_SAMPLES) % NUM_CLASSES
        mask = masking(classes=classes)
        # Every sample should be in exactly one model
        assigned = mask.any(dim=1)
        assert assigned.all()

    def test_deterministic_with_seed(self):
        kwargs = {
            "num_samples": NUM_SAMPLES,
            "num_classes": NUM_CLASSES,
            "num_models": 2,
            "p": 0.5,
            "seed": 77,
        }
        classes = torch.arange(NUM_SAMPLES) % NUM_CLASSES
        mask1 = PartitionedStratifiedMasking(**kwargs)(classes=classes)
        mask2 = PartitionedStratifiedMasking(**kwargs)(classes=classes)
        assert torch.equal(mask1, mask2)

    def test_config_creates_masking(self):
        cfg = PartitionedStratifiedMaskingConfig(
            name="partitioned_stratified", num_models=2, p=0.5, seed=42
        )
        masking = cfg(num_samples=NUM_SAMPLES, num_classes=NUM_CLASSES)
        assert isinstance(masking, PartitionedStratifiedMasking)


# --- Tests for BalancedStratifiedMasking ---


class TestBalancedStratifiedMasking:
    def test_output_shape(self):
        masking = BalancedStratifiedMasking(
            num_samples=NUM_SAMPLES, num_classes=NUM_CLASSES, num_models=3, p=0.5, seed=42
        )
        classes = torch.arange(NUM_SAMPLES) % NUM_CLASSES
        mask = masking(classes=classes)
        assert mask.shape == (NUM_SAMPLES, 3)

    def test_each_model_gets_correct_count(self):
        masking = BalancedStratifiedMasking(
            num_samples=NUM_SAMPLES, num_classes=NUM_CLASSES, num_models=3, p=0.5, seed=42
        )
        classes = torch.arange(NUM_SAMPLES) % NUM_CLASSES
        mask = masking(classes=classes)
        expected_per_model = int(NUM_SAMPLES * 0.5)
        for model_idx in range(3):
            count = mask[:, model_idx].sum().item()
            assert count == expected_per_model

    def test_deterministic_with_seed(self):
        kwargs = {
            "num_samples": NUM_SAMPLES,
            "num_classes": NUM_CLASSES,
            "num_models": 2,
            "p": 0.5,
            "seed": 55,
        }
        classes = torch.arange(NUM_SAMPLES) % NUM_CLASSES
        mask1 = BalancedStratifiedMasking(**kwargs)(classes=classes)
        mask2 = BalancedStratifiedMasking(**kwargs)(classes=classes)
        assert torch.equal(mask1, mask2)

    def test_config_creates_masking(self):
        cfg = BalancedStratifiedMaskingConfig(
            name="balanced_stratified", num_models=2, p=0.5, seed=42
        )
        masking = cfg(num_samples=NUM_SAMPLES, num_classes=NUM_CLASSES)
        assert isinstance(masking, BalancedStratifiedMasking)


# --- Tests for Mask discriminated union ---


class TestMaskDiscriminator:
    def test_uniform_from_dict(self):
        from pydantic import TypeAdapter

        adapter = TypeAdapter(Mask)
        cfg = adapter.validate_python({"name": "uniform", "num_models": 2, "p": 0.5})
        assert isinstance(cfg, UniformMaskingConfig)

    def test_independent_stratified_from_dict(self):
        from pydantic import TypeAdapter

        adapter = TypeAdapter(Mask)
        cfg = adapter.validate_python({"name": "independent_stratified", "num_models": 2, "p": 0.5})
        assert isinstance(cfg, IndependentStratifiedMaskingConfig)

    def test_partitioned_stratified_from_dict(self):
        from pydantic import TypeAdapter

        adapter = TypeAdapter(Mask)
        cfg = adapter.validate_python({"name": "partitioned_stratified", "num_models": 2, "p": 0.5})
        assert isinstance(cfg, PartitionedStratifiedMaskingConfig)

    def test_balanced_stratified_from_dict(self):
        from pydantic import TypeAdapter

        adapter = TypeAdapter(Mask)
        cfg = adapter.validate_python({"name": "balanced_stratified", "num_models": 2, "p": 0.5})
        assert isinstance(cfg, BalancedStratifiedMaskingConfig)


# --- Tests for Canaries ---


class TestSquareWatermarkCanary:
    def test_applies_watermark(self):
        canary = SquareWatermarkCanary(dim=(1, 8, 8), square_size=3)
        image = torch.zeros(1, 8, 8)
        result = canary(image)
        # Bottom-right 3x3 should be 1.0
        assert torch.all(result[:, -3:, -3:] == 1.0)
        # Rest should remain 0
        assert torch.all(result[:, :-3, :] == 0.0)

    def test_square_size_clamped_to_image(self):
        canary = SquareWatermarkCanary(dim=(1, 4, 4), square_size=10)
        assert canary.square_size == 4

    def test_config_creates_canary(self):
        cfg = SquareWatermarkCanaryConfig(name="square_watermark", square_size=2)
        canary = cfg(dim=(1, 8, 8))
        assert isinstance(canary, SquareWatermarkCanary)

    def test_config_name(self):
        cfg = SquareWatermarkCanaryConfig(name="square_watermark", square_size=2)
        assert cfg.name == "square_watermark"


class TestUniformNoiseCanary:
    def test_replaces_image_with_noise(self):
        canary = UniformNoiseCanary(dim=(1, 8, 8))
        image = torch.zeros(1, 8, 8)
        result = canary(image)
        # Result should not be all zeros (noise was applied)
        assert not torch.all(result == 0.0)
        # Values should be in [0, 1) range (from torch.rand)
        assert result.min() >= 0.0
        assert result.max() < 1.0

    def test_output_shape(self):
        canary = UniformNoiseCanary(dim=(3, 16, 16))
        image = torch.zeros(3, 16, 16)
        result = canary(image)
        assert result.shape == (3, 16, 16)

    def test_config_creates_canary(self):
        cfg = UniformNoiseCanaryConfig(name="uniform_noise")
        canary = cfg(dim=(1, 8, 8))
        assert isinstance(canary, UniformNoiseCanary)

    def test_config_name(self):
        cfg = UniformNoiseCanaryConfig(name="uniform_noise")
        assert cfg.name == "uniform_noise"


class TestCreateCanaryGenerator:
    def test_square_watermark(self):
        cfg = SquareWatermarkCanaryConfig(name="square_watermark", square_size=2)
        canary = create_canary_generator(config=cfg, dim=(1, 8, 8))
        image = torch.zeros(1, 8, 8)
        result = canary(image)
        assert torch.all(result[:, -2:, -2:] == 1.0)

    def test_uniform_noise(self):
        cfg = UniformNoiseCanaryConfig(name="uniform_noise")
        canary = create_canary_generator(config=cfg, dim=(1, 8, 8))
        image = torch.zeros(1, 8, 8)
        result = canary(image)
        assert result.shape == (1, 8, 8)


# --- Tests for Canary Class Assignment ---


class TestDerangeBalancedIndices:
    def test_no_fixed_points(self):
        canary_lookup = {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8]}
        result = derange_balanced_indices(canary_lookup, seed=42)
        original = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
        # No label should match its original class
        assert torch.all(result != original)

    def test_preserves_class_counts(self):
        canary_lookup = {0: [0, 1], 1: [2, 3], 2: [4, 5]}
        result = derange_balanced_indices(canary_lookup, seed=42)
        # Each class should appear exactly twice in the result
        for cls in range(3):
            assert (result == cls).sum().item() == 2

    def test_deterministic(self):
        canary_lookup = {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8]}
        r1 = derange_balanced_indices(canary_lookup, seed=42)
        r2 = derange_balanced_indices(canary_lookup, seed=42)
        assert torch.equal(r1, r2)


class TestAlternativeDerangeBalancedIndices:
    def test_no_fixed_points(self):
        canary_lookup = {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8]}
        result = alternative_derange_balanced_indices(canary_lookup, seed=42)
        original = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
        assert torch.all(result != original)

    def test_preserves_class_counts(self):
        canary_lookup = {0: [0, 1], 1: [2, 3], 2: [4, 5]}
        result = alternative_derange_balanced_indices(canary_lookup, seed=42)
        for cls in range(3):
            assert (result == cls).sum().item() == 2

    def test_deterministic(self):
        canary_lookup = {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8]}
        r1 = alternative_derange_balanced_indices(canary_lookup, seed=42)
        r2 = alternative_derange_balanced_indices(canary_lookup, seed=42)
        assert torch.equal(r1, r2)


class TestRandomDerangeIndices:
    def test_output_length(self):
        canary_lookup = {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8]}
        result = random_derange_indices(canary_lookup, seed=42)
        assert len(result) == 9

    def test_deterministic(self):
        canary_lookup = {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8]}
        r1 = random_derange_indices(canary_lookup, seed=42)
        r2 = random_derange_indices(canary_lookup, seed=42)
        assert torch.equal(r1, r2)

    def test_labels_shifted(self):
        canary_lookup = {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8]}
        result = random_derange_indices(canary_lookup, seed=42)
        original = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
        # At least some labels should differ (shift-based)
        assert not torch.equal(result, original)


# --- Tests for CanaryDataset ---


class TestCanaryDataset:
    def test_len(self):
        dataset = FakeDataset(NUM_SAMPLES, NUM_CLASSES)
        subset_indices = torch.arange(NUM_SAMPLES)
        cd = CanaryDataset(dataset=dataset, subset_indices=subset_indices, num_classes=NUM_CLASSES)
        assert len(cd) == NUM_SAMPLES

    def test_getitem_no_canary(self):
        dataset = FakeDataset(NUM_SAMPLES, NUM_CLASSES)
        subset_indices = torch.arange(NUM_SAMPLES)
        cd = CanaryDataset(dataset=dataset, subset_indices=subset_indices, num_classes=NUM_CLASSES)
        img, lbl = cd[0]
        assert isinstance(img, torch.Tensor)
        assert isinstance(lbl, torch.Tensor)
        assert img.shape == INPUT_SHAPE

    def test_getitem_with_canary(self):
        dataset = FakeDataset(NUM_SAMPLES, NUM_CLASSES)
        subset_indices = torch.arange(NUM_SAMPLES)
        canary_indices = torch.tensor([0, 1, 2])
        canary_labels = torch.tensor([1, 2, 3])
        canary_transform = SquareWatermarkCanary(dim=INPUT_SHAPE, square_size=2)

        cd = CanaryDataset(
            dataset=dataset,
            subset_indices=subset_indices,
            num_classes=NUM_CLASSES,
            canary_indices=canary_indices,
            canary_labels=canary_labels,
            canary_transform=canary_transform,
        )
        img, lbl = cd[0]
        # Canary transform should have been applied (watermark in bottom-right)
        assert torch.all(img[:, -2:, -2:] == 1.0)
        # Label should be the canary label
        assert lbl.item() == 1

    def test_index_out_of_range(self):
        dataset = FakeDataset(NUM_SAMPLES, NUM_CLASSES)
        subset_indices = torch.arange(10)
        cd = CanaryDataset(dataset=dataset, subset_indices=subset_indices, num_classes=NUM_CLASSES)
        with pytest.raises(IndexError):
            cd[10]

    def test_canary_index_without_transform_raises(self):
        dataset = FakeDataset(NUM_SAMPLES, NUM_CLASSES)
        subset_indices = torch.arange(NUM_SAMPLES)
        canary_indices = torch.tensor([0])

        cd = CanaryDataset(
            dataset=dataset,
            subset_indices=subset_indices,
            num_classes=NUM_CLASSES,
            canary_indices=canary_indices,
            canary_labels=None,
            canary_transform=None,
        )
        with pytest.raises(RuntimeError):
            cd[0]


# --- Tests for GpuDataset ---


class TestGpuDataset:
    def test_len(self):
        dataset = FakeDataset(20, NUM_CLASSES)
        gpu_ds = GpuDataset(dataset, device=torch.device("cpu"))
        assert len(gpu_ds) == 20

    def test_getitem(self):
        dataset = FakeDataset(20, NUM_CLASSES)
        gpu_ds = GpuDataset(dataset, device=torch.device("cpu"))
        img, lbl = gpu_ds[0]
        assert isinstance(img, torch.Tensor)
        assert isinstance(lbl, torch.Tensor)
        assert img.shape == INPUT_SHAPE

    def test_all_data_on_device(self):
        dataset = FakeDataset(20, NUM_CLASSES)
        device = torch.device("cpu")
        gpu_ds = GpuDataset(dataset, device=device)
        assert gpu_ds.images.device == device
        assert gpu_ds.labels.device == device

    def test_labels_are_long(self):
        dataset = FakeDataset(20, NUM_CLASSES)
        gpu_ds = GpuDataset(dataset, device=torch.device("cpu"))
        assert gpu_ds.labels.dtype == torch.long


# --- Tests for DatasetConfig ---


class TestDatasetConfig:
    def test_apply_mask(self):
        dataset = FakeDataset(NUM_SAMPLES, NUM_CLASSES)
        cfg = DatasetConfig(
            data={"name": "mnist"},
            mask={"name": "uniform", "num_models": 2, "p": 0.5, "model_index": 0, "seed": 42},
        )
        result = cfg.apply_mask(dataset, num_classes=NUM_CLASSES)
        # Should return a Subset with fewer samples
        assert len(result) < NUM_SAMPLES

    def test_apply_mask_none(self):
        dataset = FakeDataset(NUM_SAMPLES, NUM_CLASSES)
        cfg = DatasetConfig(data={"name": "mnist"}, mask=None)
        result = cfg.apply_mask(dataset, num_classes=NUM_CLASSES)
        # Should return the original dataset unchanged
        assert result is dataset

    def test_apply_canary_none(self):
        dataset = FakeDataset(NUM_SAMPLES, NUM_CLASSES)
        cfg = DatasetConfig(data={"name": "mnist"}, canary=None)
        result = cfg.apply_canary(dataset, num_classes=NUM_CLASSES)
        assert result is dataset

    def test_apply_canary_zero_share(self):
        dataset = FakeDataset(NUM_SAMPLES, NUM_CLASSES)
        cfg = DatasetConfig(
            data={"name": "mnist"},
            canary={"name": "square_watermark", "share": 0, "square_size": 2},
            seed=42,
        )
        result = cfg.apply_canary(dataset, num_classes=NUM_CLASSES)
        assert result is dataset

    def test_apply_canary_requires_seed(self):
        dataset = FakeDataset(NUM_SAMPLES, NUM_CLASSES)
        cfg = DatasetConfig(
            data={"name": "mnist"},
            canary={"name": "square_watermark", "share": 0.1, "square_size": 2},
            seed=None,
        )
        with pytest.raises(ValueError, match="seed is required"):
            cfg.apply_canary(dataset, num_classes=NUM_CLASSES)

    def test_apply_canary_returns_canary_dataset(self):
        dataset = FakeDataset(NUM_SAMPLES, NUM_CLASSES)
        cfg = DatasetConfig(
            data={"name": "mnist"},
            canary={"name": "square_watermark", "share": 0.1, "square_size": 2},
            seed=42,
        )
        result = cfg.apply_canary(dataset, num_classes=NUM_CLASSES)
        assert isinstance(result, CanaryDataset)

    def test_apply_canary_with_train_size(self):
        dataset = FakeDataset(NUM_SAMPLES, NUM_CLASSES)
        cfg = DatasetConfig(
            data={"name": "mnist"},
            canary={"name": "square_watermark", "share": 0.1, "square_size": 2},
            seed=42,
            train_size=50,
        )
        result = cfg.apply_canary(dataset, num_classes=NUM_CLASSES)
        assert isinstance(result, CanaryDataset)
        assert len(result) == 50

    def test_apply_canary_train_size_exceeds_raises(self):
        dataset = FakeDataset(NUM_SAMPLES, NUM_CLASSES)
        cfg = DatasetConfig(
            data={"name": "mnist"},
            canary={"name": "square_watermark", "share": 0.1, "square_size": 2},
            seed=42,
            train_size=200,
        )
        with pytest.raises(ValueError, match="train_size exceeds dataset size"):
            cfg.apply_canary(dataset, num_classes=NUM_CLASSES)


# --- Tests for Canary discriminated union ---


class TestCanaryDiscriminator:
    def test_square_watermark_from_dict(self):
        from pydantic import TypeAdapter

        from privacy_and_grokking.datasets.canaries import CanaryType

        adapter = TypeAdapter(CanaryType)
        cfg = adapter.validate_python({"name": "square_watermark", "share": 0.1, "square_size": 3})
        assert isinstance(cfg, SquareWatermarkCanaryConfig)

    def test_uniform_noise_from_dict(self):
        from pydantic import TypeAdapter

        from privacy_and_grokking.datasets.canaries import CanaryType

        adapter = TypeAdapter(CanaryType)
        cfg = adapter.validate_python({"name": "uniform_noise", "share": 0.2})
        assert isinstance(cfg, UniformNoiseCanaryConfig)
