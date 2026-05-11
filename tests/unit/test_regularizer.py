import pytest
import torch

from privacy_and_grokking.loss.regularizer.mmd import MMDRegularizerConfig
from privacy_and_grokking.loss.regularizer.overlap import OverlapRegularizerConfig
from privacy_and_grokking.loss.regularizer.overlap_adaptive import OverlapAdaptiveRegularizerConfig
from privacy_and_grokking.loss.regularizer.overlap_kde import OverlapKDERegularizerConfig
from privacy_and_grokking.loss.regularizer.per_sample_distance import (
    PerSampleDistanceRegularizerConfig,
)
from privacy_and_grokking.loss.regularizer_source.gaussian import GaussianNoiseConfig


def _make_gaussian_source(std: float = 0.1, num_noisy_samples: int = 1):
    """Helper to create a Gaussian noise source config for regularizer tests."""
    return GaussianNoiseConfig(num_noisy_samples=num_noisy_samples, mean=0.0, std=std)


class TestMMDRegularizer:
    def test_returns_callable(self):
        cfg = MMDRegularizerConfig(name="mmd", source=_make_gaussian_source(), bandwidth=0.1)
        reg_fn = cfg()
        assert callable(reg_fn)

    def test_output_is_scalar(self):
        cfg = MMDRegularizerConfig(name="mmd", source=_make_gaussian_source(), bandwidth=0.1)
        reg_fn = cfg()
        train_losses = torch.randn(32).abs()
        result = reg_fn(train_losses)
        assert result.dim() == 0

    def test_output_is_non_negative(self):
        cfg = MMDRegularizerConfig(name="mmd", source=_make_gaussian_source(), bandwidth=0.1)
        reg_fn = cfg()
        train_losses = torch.randn(32).abs()
        result = reg_fn(train_losses)
        assert result.item() >= 0.0

    def test_identical_distributions_give_low_mmd(self):
        # With zero noise, train and val losses are identical -> MMD ≈ 0
        cfg = MMDRegularizerConfig(name="mmd", source=_make_gaussian_source(std=0.0), bandwidth=0.1)
        reg_fn = cfg()
        train_losses = torch.randn(32).abs()
        result = reg_fn(train_losses)
        assert result.item() < 1e-5

    def test_different_distributions_give_higher_mmd(self):
        # Large noise -> distributions differ -> higher MMD
        cfg_low = MMDRegularizerConfig(
            name="mmd", source=_make_gaussian_source(std=0.0), bandwidth=0.1
        )
        cfg_high = MMDRegularizerConfig(
            name="mmd", source=_make_gaussian_source(std=5.0), bandwidth=0.1
        )
        train_losses = torch.ones(32)
        result_low = cfg_low()(train_losses)
        result_high = cfg_high()(train_losses)
        assert result_high.item() > result_low.item()


class TestOverlapRegularizer:
    def test_returns_callable(self):
        cfg = OverlapRegularizerConfig(
            name="overlap", source=_make_gaussian_source(), n_bins=50, sigma=0.05
        )
        reg_fn = cfg()
        assert callable(reg_fn)

    def test_output_is_scalar(self):
        cfg = OverlapRegularizerConfig(
            name="overlap", source=_make_gaussian_source(), n_bins=50, sigma=0.05
        )
        reg_fn = cfg()
        train_losses = torch.randn(32).abs()
        result = reg_fn(train_losses)
        assert result.dim() == 0

    def test_identical_distributions_give_low_regularization(self):
        # Zero noise -> identical distributions -> overlap ≈ 1 -> reg ≈ 0
        cfg = OverlapRegularizerConfig(
            name="overlap", source=_make_gaussian_source(std=0.0), n_bins=50, sigma=0.05
        )
        reg_fn = cfg()
        train_losses = torch.randn(64).abs()
        result = reg_fn(train_losses)
        assert result.item() < 0.1

    def test_output_bounded_between_zero_and_one(self):
        cfg = OverlapRegularizerConfig(
            name="overlap", source=_make_gaussian_source(std=0.5), n_bins=50, sigma=0.05
        )
        reg_fn = cfg()
        train_losses = torch.randn(32).abs()
        result = reg_fn(train_losses)
        assert 0.0 <= result.item() <= 1.0


class TestOverlapAdaptiveRegularizer:
    def test_returns_callable(self):
        cfg = OverlapAdaptiveRegularizerConfig(
            name="overlap_adaptive", source=_make_gaussian_source()
        )
        reg_fn = cfg()
        assert callable(reg_fn)

    def test_output_is_scalar(self):
        cfg = OverlapAdaptiveRegularizerConfig(
            name="overlap_adaptive", source=_make_gaussian_source()
        )
        reg_fn = cfg()
        train_losses = torch.randn(32).abs()
        result = reg_fn(train_losses)
        assert result.dim() == 0

    def test_identical_distributions_give_low_regularization(self):
        cfg = OverlapAdaptiveRegularizerConfig(
            name="overlap_adaptive", source=_make_gaussian_source(std=0.0)
        )
        reg_fn = cfg()
        train_losses = torch.randn(64).abs()
        result = reg_fn(train_losses)
        assert result.item() < 0.1


class TestOverlapKDERegularizer:
    def test_returns_callable(self):
        cfg = OverlapKDERegularizerConfig(
            name="overlap_kde", source=_make_gaussian_source(), n_points=100
        )
        reg_fn = cfg()
        assert callable(reg_fn)

    def test_output_is_scalar(self):
        cfg = OverlapKDERegularizerConfig(
            name="overlap_kde", source=_make_gaussian_source(), n_points=100
        )
        reg_fn = cfg()
        train_losses = torch.randn(32).abs()
        result = reg_fn(train_losses)
        assert result.dim() == 0

    def test_identical_distributions_give_low_regularization(self):
        cfg = OverlapKDERegularizerConfig(
            name="overlap_kde", source=_make_gaussian_source(std=0.0), n_points=100
        )
        reg_fn = cfg()
        train_losses = torch.randn(64).abs()
        result = reg_fn(train_losses)
        # KDE bandwidth estimation introduces some residual even for identical samples
        assert result.item() < 0.5


class TestPerSampleDistanceRegularizer:
    def test_returns_callable(self):
        cfg = PerSampleDistanceRegularizerConfig(
            name="per_sample_distance", source=_make_gaussian_source(), metric="l1"
        )
        reg_fn = cfg()
        assert callable(reg_fn)

    def test_output_is_scalar_1d(self):
        cfg = PerSampleDistanceRegularizerConfig(
            name="per_sample_distance", source=_make_gaussian_source(), metric="l1"
        )
        reg_fn = cfg()
        train_losses = torch.randn(16).abs()
        result = reg_fn(train_losses)
        assert result.dim() == 0

    def test_output_is_non_negative(self):
        cfg = PerSampleDistanceRegularizerConfig(
            name="per_sample_distance", source=_make_gaussian_source(), metric="l1"
        )
        reg_fn = cfg()
        train_losses = torch.randn(16).abs()
        result = reg_fn(train_losses)
        assert result.item() >= 0.0

    def test_zero_noise_gives_zero_distance(self):
        cfg = PerSampleDistanceRegularizerConfig(
            name="per_sample_distance", source=_make_gaussian_source(std=0.0), metric="l1"
        )
        reg_fn = cfg()
        train_losses = torch.randn(16).abs()
        result = reg_fn(train_losses)
        assert result.item() == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize("metric", ["l1", "l2", "huber"])
    def test_all_metrics_produce_scalar(self, metric):
        cfg = PerSampleDistanceRegularizerConfig(
            name="per_sample_distance",
            source=_make_gaussian_source(std=0.5),
            metric=metric,
        )
        reg_fn = cfg()
        train_losses = torch.randn(16).abs()
        result = reg_fn(train_losses)
        assert result.dim() == 0
        assert result.item() >= 0.0

    def test_multiple_noisy_copies(self):
        cfg = PerSampleDistanceRegularizerConfig(
            name="per_sample_distance",
            source=_make_gaussian_source(std=0.5, num_noisy_samples=3),
            metric="l1",
        )
        reg_fn = cfg()
        train_losses = torch.randn(16).abs()
        result = reg_fn(train_losses)
        assert result.dim() == 0
        assert result.item() >= 0.0

    def test_2d_input(self):
        """Test with per-class loss vectors (B, C)."""
        cfg = PerSampleDistanceRegularizerConfig(
            name="per_sample_distance",
            source=_make_gaussian_source(std=0.5),
            metric="l1",
        )
        reg_fn = cfg()
        train_losses = torch.randn(16, 5).abs()  # (B, C)
        result = reg_fn(train_losses)
        assert result.dim() == 0
        assert result.item() >= 0.0
