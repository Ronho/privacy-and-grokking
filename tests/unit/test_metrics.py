import unittest.mock

import torch
import torch.nn as nn

from privacy_and_grokking.metrics.config import MetricsConfig
from privacy_and_grokking.metrics.curvature import curvature
from privacy_and_grokking.metrics.distribution_overlap import (
    compute_distribution_overlap,
    compute_mmd,
)
from privacy_and_grokking.metrics.evaluate import evaluate
from privacy_and_grokking.metrics.norms import compute_gradient_norms, compute_weight_norms
from privacy_and_grokking.metrics.roc import compute_roc_metrics_single_step
from privacy_and_grokking.models.base import ModelBase

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
        assert cfg.heavy_metrics_log_frequency == 1000
        assert cfg.accuracy is True
        assert cfg.curvature is False

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
# Evaluate & Canary Cross-Comparisons
# ---------------------------------------------------------------------------


class _DummyModel(ModelBase):
    def __init__(self, in_dim: int = 4, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(
        self, x: torch.Tensor, verbose: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        logits = self.fc(x)
        if verbose:
            return logits, x
        return logits

    def classifier(self) -> nn.Module:
        return self.fc


class TestEvaluate:
    @unittest.mock.patch("mlflow.log_metrics")
    def test_evaluate_canary_cross_comparisons(self, mock_log_metrics):
        model = _DummyModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()

        train_loader = _dummy_loader(n=32)
        test_loader = _dummy_loader(n=32)
        train_canary_loader = _dummy_loader(n=16)
        test_canary_loader = _dummy_loader(n=16)

        cfg = MetricsConfig(
            curvature=False,
            neural_collapse=False,
            distribution_overlap=True,
            attack_ce_loss=True,
            attack_mse_loss=True,
            attack_correctness=True,
            attack_true_class_prob=True,
            attack_true_class_logit=True,
        )

        metrics = evaluate(
            model=model,
            step=1,
            optimizer=opt,
            loss_fn=loss_fn,
            key_prefix="eval",
            train_loader=train_loader,
            test_loader=test_loader,
            compute_heavy_metrics=False,
            num_classes=2,
            metrics_config=cfg,
            train_canary_loader=train_canary_loader,
            test_canary_loader=test_canary_loader,
        )

        # Standard attacks (train vs test)
        assert "eval/attack/ce_loss/auc" in metrics
        assert "eval/attack/true_class_prob/auc" in metrics

        # Canary vs canary
        assert "eval/attack/canary_ce_loss/auc" in metrics
        assert "eval/loss/canary_ce/overlap" in metrics

        # Train canary vs test attacks & overlap
        assert "eval/attack/train_canary_vs_test/ce_loss/auc" in metrics
        assert "eval/attack/train_canary_vs_test/mse_loss/auc" in metrics
        assert "eval/attack/train_canary_vs_test/correctness/auc" in metrics
        assert "eval/attack/train_canary_vs_test/true_class_prob/auc" in metrics
        assert "eval/attack/train_canary_vs_test/true_class_logit/auc" in metrics
        assert "eval/loss/train_canary_vs_test/ce/overlap" in metrics
        assert "eval/loss/train_canary_vs_test/mse/overlap" in metrics

        # Train + train canary vs test attacks & overlap
        assert "eval/attack/train_plus_canary_vs_test/ce_loss/auc" in metrics
        assert "eval/attack/train_plus_canary_vs_test/mse_loss/auc" in metrics
        assert "eval/attack/train_plus_canary_vs_test/correctness/auc" in metrics
        assert "eval/attack/train_plus_canary_vs_test/true_class_prob/auc" in metrics
        assert "eval/attack/train_plus_canary_vs_test/true_class_logit/auc" in metrics
        assert "eval/loss/train_plus_canary_vs_test/ce/overlap" in metrics
        assert "eval/loss/train_plus_canary_vs_test/mse/overlap" in metrics

        mock_log_metrics.assert_called_once()