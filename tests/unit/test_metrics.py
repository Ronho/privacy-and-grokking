"""Tests for the unified privacy_and_grokking.metrics package."""

import torch
import torch.nn as nn

from privacy_and_grokking.metrics.config import MetricsConfig
from privacy_and_grokking.metrics.curvature import curvature
from privacy_and_grokking.metrics.distribution_overlap import (
    compute_distribution_overlap,
    compute_distribution_overlap_adaptive,
    compute_distribution_overlap_kde,
    compute_kl_divergence,
    compute_kl_divergence_adaptive,
    compute_kl_divergence_kde,
    compute_mmd,
    soft_distribution_overlap,
)
from privacy_and_grokking.metrics.norms import compute_gradient_norms, compute_weight_norms
from privacy_and_grokking.metrics.optimizer_params import get_optimizer_internals
from privacy_and_grokking.metrics.roc import compute_roc_metrics_single_step

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_model() -> nn.Module:
    """A tiny MLP for testing."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )


def _dummy_loader(n: int = 32, input_dim: int = 4, num_classes: int = 2):
    """Create a DataLoader with random data."""
    x = torch.randn(n, input_dim)
    y = torch.randint(0, num_classes, (n,))
    dataset = torch.utils.data.TensorDataset(x, y)
    return torch.utils.data.DataLoader(dataset, batch_size=16)


# ---------------------------------------------------------------------------
# MetricsConfig
# ---------------------------------------------------------------------------


class TestMetricsConfig:
    def test_defaults(self):
        cfg = MetricsConfig()
        assert cfg.log_frequency == 1000
        assert cfg.heavy_metrics_log_frequency == 10000
        assert cfg.accuracy is True
        assert cfg.curvature is True

    def test_custom_values(self):
        cfg = MetricsConfig(log_frequency=500, curvature=False, mmd=False)
        assert cfg.log_frequency == 500
        assert cfg.curvature is False
        assert cfg.mmd is False

    def test_any_distribution_metric_all_disabled(self):
        cfg = MetricsConfig(
            distribution_overlap=False,
            distribution_overlap_adaptive=False,
            distribution_overlap_kde=False,
            soft_overlap=False,
            kl_divergence=False,
            kl_divergence_adaptive=False,
            kl_divergence_kde=False,
            js_distance=False,
            js_distance_adaptive=False,
            js_distance_kde=False,
            mmd=False,
        )
        assert cfg.any_distribution_metric is False

    def test_any_distribution_metric_one_enabled(self):
        cfg = MetricsConfig(
            distribution_overlap=False,
            distribution_overlap_adaptive=False,
            distribution_overlap_kde=False,
            soft_overlap=False,
            kl_divergence=False,
            kl_divergence_adaptive=False,
            kl_divergence_kde=False,
            js_distance=False,
            js_distance_adaptive=False,
            js_distance_kde=False,
            mmd=True,
        )
        assert cfg.any_distribution_metric is True

    def test_any_attack_metric_all_disabled(self):
        cfg = MetricsConfig(
            attack_true_class_prob=False,
            attack_true_class_logit=False,
            attack_ce_loss=False,
            attack_mse_loss=False,
            attack_correctness=False,
            merlin_morgan=False,
        )
        assert cfg.any_attack_metric is False

    def test_any_attack_metric_one_enabled(self):
        cfg = MetricsConfig(
            attack_true_class_prob=False,
            attack_true_class_logit=False,
            attack_ce_loss=True,
            attack_mse_loss=False,
            attack_correctness=False,
            merlin_morgan=False,
        )
        assert cfg.any_attack_metric is True

    def test_serialization_roundtrip(self):
        cfg = MetricsConfig(log_frequency=200, accuracy=False)
        data = cfg.model_dump()
        restored = MetricsConfig.model_validate(data)
        assert restored == cfg


# ---------------------------------------------------------------------------
# Norms
# ---------------------------------------------------------------------------


class TestNorms:
    def test_weight_norms_keys(self):
        model = _simple_model()
        norms = compute_weight_norms(model)
        assert "weight_norm/total" in norms
        assert all(v >= 0 for v in norms.values())

    def test_weight_norms_per_param(self):
        model = _simple_model()
        norms = compute_weight_norms(model)
        named_params = [n for n, _ in model.named_parameters()]
        for name in named_params:
            assert f"weight_norm/{name}" in norms

    def test_gradient_norms_no_grad(self):
        model = _simple_model()
        # No backward pass yet — no gradients
        norms = compute_gradient_norms(model)
        assert norms["grad_norm/total"] == 0.0

    def test_gradient_norms_after_backward(self):
        model = _simple_model()
        x = torch.randn(4, 4)
        loss = model(x).sum()
        loss.backward()
        norms = compute_gradient_norms(model)
        assert norms["grad_norm/total"] > 0.0
        assert all(v >= 0 for v in norms.values())


# ---------------------------------------------------------------------------
# Distribution Overlap
# ---------------------------------------------------------------------------


class TestDistributionOverlap:
    def test_identical_distributions(self):
        a = torch.randn(500)
        overlap = compute_distribution_overlap(a, a.clone())
        assert overlap > 0.95

    def test_disjoint_distributions(self):
        a = torch.zeros(500)
        b = torch.ones(500) * 100
        overlap = compute_distribution_overlap(a, b)
        assert overlap < 0.05

    def test_adaptive_identical(self):
        a = torch.randn(200)
        overlap = compute_distribution_overlap_adaptive(a, a.clone())
        assert overlap > 0.90

    def test_kde_identical(self):
        a = torch.randn(200)
        overlap = compute_distribution_overlap_kde(a, a.clone())
        assert overlap > 0.90

    def test_soft_overlap_identical(self):
        a = torch.randn(200)
        result = soft_distribution_overlap(a, a.clone())
        assert result.item() > 0.80

    def test_soft_overlap_is_differentiable(self):
        a = torch.randn(100, requires_grad=True)
        b = torch.randn(100)
        result = soft_distribution_overlap(a, b)
        result.backward()
        assert a.grad is not None
        assert a.grad.shape == a.shape

    def test_empty_input(self):
        a = torch.tensor([])
        b = torch.randn(10)
        assert compute_distribution_overlap(a, b) == 0.0
        assert compute_distribution_overlap_adaptive(a, b) == 0.0
        assert compute_distribution_overlap_kde(a, b) == 0.0


class TestKLDivergence:
    def test_identical_distributions(self):
        a = torch.randn(500)
        kl = compute_kl_divergence(a, a.clone())
        assert kl < 0.1

    def test_different_distributions(self):
        a = torch.randn(500)
        b = torch.randn(500) + 5.0
        kl = compute_kl_divergence(a, b)
        assert kl > 0.5

    def test_adaptive_variant(self):
        a = torch.randn(500)
        kl = compute_kl_divergence_adaptive(a, a.clone())
        assert kl < 0.1

    def test_kde_variant(self):
        a = torch.randn(500)
        kl = compute_kl_divergence_kde(a, a.clone())
        assert kl < 0.1

    def test_empty_input(self):
        a = torch.tensor([])
        b = torch.randn(10)
        assert compute_kl_divergence(a, b) == 0.0


class TestMMD:
    def test_identical_distributions(self):
        a = torch.randn(200)
        mmd = compute_mmd(a, a.clone())
        assert mmd < 0.05

    def test_different_distributions(self):
        a = torch.randn(200)
        b = torch.randn(200) + 5.0
        mmd = compute_mmd(a, b)
        assert mmd > 0.0

    def test_non_negative(self):
        a = torch.randn(100)
        b = torch.randn(100)
        mmd = compute_mmd(a, b)
        assert mmd >= 0.0

    def test_empty_input(self):
        a = torch.tensor([])
        b = torch.randn(10)
        assert compute_mmd(a, b) == 0.0


# ---------------------------------------------------------------------------
# ROC Metrics
# ---------------------------------------------------------------------------


class TestROCMetrics:
    def test_perfect_separation(self):
        train_signals = torch.ones(100)
        test_signals = torch.zeros(100)
        metrics = compute_roc_metrics_single_step(train_signals, test_signals)
        assert metrics["auc"] == 1.0
        assert metrics["tpr-at-fpr/1"] == 1.0
        assert metrics["tpr-at-fpr/5"] == 1.0
        assert metrics["tpr-at-fpr/10"] == 1.0

    def test_random_signals(self):
        train_signals = torch.randn(200)
        test_signals = torch.randn(200)
        metrics = compute_roc_metrics_single_step(train_signals, test_signals)
        # Random signals should give AUC near 0.5
        assert 0.3 < metrics["auc"] < 0.7

    def test_custom_fpr_rates(self):
        train_signals = torch.ones(50)
        test_signals = torch.zeros(50)
        metrics = compute_roc_metrics_single_step(
            train_signals, test_signals, fpr_rates=[0.02, 0.20]
        )
        assert "tpr-at-fpr/2" in metrics
        assert "tpr-at-fpr/20" in metrics

    def test_output_keys(self):
        train_signals = torch.randn(50)
        test_signals = torch.randn(50)
        metrics = compute_roc_metrics_single_step(train_signals, test_signals)
        assert "auc" in metrics
        assert "tpr-at-fpr/1" in metrics
        assert "tpr-at-fpr/5" in metrics
        assert "tpr-at-fpr/10" in metrics


# ---------------------------------------------------------------------------
# Optimizer Internals
# ---------------------------------------------------------------------------


class TestOptimizerInternals:
    def test_adam_internals(self):
        model = _simple_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        # Do a step to populate optimizer state
        x = torch.randn(4, 4)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        stats = get_optimizer_internals(optimizer)
        # Adam should have exp_avg and exp_avg_sq
        assert len(stats) > 0
        assert any("exp_avg" in k for k in stats)
        assert any("exp_avg_sq" in k for k in stats)

    def test_sgd_no_momentum(self):
        model = _simple_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        x = torch.randn(4, 4)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        stats = get_optimizer_internals(optimizer)
        # SGD without momentum has no state tensors
        assert len(stats) == 0

    def test_sgd_with_momentum(self):
        model = _simple_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        x = torch.randn(4, 4)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        stats = get_optimizer_internals(optimizer)
        assert len(stats) > 0
        assert any("momentum_buffer" in k for k in stats)


# ---------------------------------------------------------------------------
# Curvature
# ---------------------------------------------------------------------------


class TestCurvature:
    def test_returns_expected_keys(self):
        model = _simple_model()
        loss_fn = nn.CrossEntropyLoss()
        loader = _dummy_loader()
        metrics = curvature(model, loss_fn, loader)
        assert "curvature/hessian_trace" in metrics
        assert "curvature/top_eigenvalue" in metrics

    def test_values_are_finite(self):
        model = _simple_model()
        loss_fn = nn.CrossEntropyLoss()
        loader = _dummy_loader()
        metrics = curvature(model, loss_fn, loader)
        for v in metrics.values():
            assert torch.isfinite(torch.tensor(v))


# ---------------------------------------------------------------------------
# Evaluate (integration)
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_minimal_config(self):
        """Evaluate with most metrics disabled runs without error."""
        from unittest.mock import patch

        from privacy_and_grokking.metrics.evaluate import evaluate

        model = _simple_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()
        loader = _dummy_loader()

        # Do a forward+backward so optimizer has state
        x = torch.randn(4, 4)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        cfg = MetricsConfig(
            loss_stats=True,
            accuracy=True,
            weight_norms=False,
            gradient_norms=False,
            optimizer_internals=False,
            distribution_overlap=False,
            distribution_overlap_adaptive=False,
            distribution_overlap_kde=False,
            soft_overlap=False,
            kl_divergence=False,
            kl_divergence_adaptive=False,
            kl_divergence_kde=False,
            mmd=False,
            attack_true_class_prob=False,
            attack_true_class_logit=False,
            attack_ce_loss=False,
            attack_mse_loss=False,
            attack_correctness=False,
            curvature=False,
            merlin_morgan=False,
        )

        with patch("privacy_and_grokking.metrics.evaluate.mlflow"):
            metrics = evaluate(
                model=model,
                step=0,
                optimizer=optimizer,
                loss_fn=loss_fn,
                key_prefix="test",
                train_loader=loader,
                test_loader=loader,
                compute_heavy_metrics=False,
                last_step=False,
                metrics_config=cfg,
            )

        assert "test/train/loss/mse/mean" in metrics
        assert "test/train/accuracy" in metrics
        assert "test/test/accuracy" in metrics
        # Disabled metrics should not appear
        assert not any("weight_norm" in k for k in metrics)
        assert not any("grad_norm" in k for k in metrics)
        assert not any("attack" in k for k in metrics)

    def test_full_config(self):
        """Evaluate with all metrics enabled runs without error."""
        from unittest.mock import patch

        from privacy_and_grokking.metrics.evaluate import evaluate

        model = _simple_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()
        loader = _dummy_loader()

        x = torch.randn(4, 4)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        cfg = MetricsConfig(merlin_morgan=False)  # Skip MM for speed

        with patch("privacy_and_grokking.metrics.evaluate.mlflow"):
            metrics = evaluate(
                model=model,
                step=0,
                optimizer=optimizer,
                loss_fn=loss_fn,
                key_prefix="eval",
                train_loader=loader,
                test_loader=loader,
                compute_heavy_metrics=True,
                last_step=False,
                metrics_config=cfg,
            )

        assert "eval/weight_norm/total" in metrics
        assert "eval/grad_norm/total" in metrics
        assert "eval/train/accuracy" in metrics
        assert "eval/curvature/hessian_trace" in metrics
        assert any("attack" in k for k in metrics)
        assert any("overlap" in k for k in metrics)


# ---------------------------------------------------------------------------
# Package-level imports
# ---------------------------------------------------------------------------


class TestPackageImports:
    def test_metrics_config_importable(self):
        from privacy_and_grokking.metrics import MetricsConfig

        assert MetricsConfig is not None

    def test_evaluate_importable(self):
        from privacy_and_grokking.metrics import evaluate

        assert callable(evaluate)

    def test_extraction_handler_importable(self):
        from privacy_and_grokking.metrics import extraction_handler

        assert callable(extraction_handler)
