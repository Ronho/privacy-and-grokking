import pytest
import torch

from privacy_and_grokking.loss.loss.ce import CrossEntropyLossConfig
from privacy_and_grokking.loss.loss.mse import MSELossConfig


class TestMSELoss:
    def test_returns_callable(self):
        cfg = MSELossConfig(name="mse")
        loss_fn = cfg(num_classes=10)
        assert callable(loss_fn)

    def test_output_is_scalar_with_mean_reduction(self):
        cfg = MSELossConfig(name="mse", reduction="mean")
        loss_fn = cfg(num_classes=10)
        logits = torch.randn(8, 10)
        labels = torch.randint(0, 10, (8,))
        result = loss_fn(logits, labels)
        assert result.dim() == 0

    def test_output_is_per_sample_with_none_reduction(self):
        cfg = MSELossConfig(name="mse", reduction="none")
        loss_fn = cfg(num_classes=10)
        logits = torch.randn(8, 10)
        labels = torch.randint(0, 10, (8,))
        result = loss_fn(logits, labels)
        assert result.shape == (8, 10)

    def test_perfect_prediction_gives_zero_loss(self):
        cfg = MSELossConfig(name="mse", reduction="mean")
        loss_fn = cfg(num_classes=3)
        # One-hot targets for labels [0, 1, 2]
        logits = torch.eye(3)
        labels = torch.tensor([0, 1, 2])
        result = loss_fn(logits, labels)
        assert result.item() == pytest.approx(0.0, abs=1e-6)

    def test_loss_is_non_negative(self):
        cfg = MSELossConfig(name="mse", reduction="mean")
        loss_fn = cfg(num_classes=5)
        logits = torch.randn(16, 5)
        labels = torch.randint(0, 5, (16,))
        result = loss_fn(logits, labels)
        assert result.item() >= 0.0

    def test_missing_num_classes_raises(self):
        cfg = MSELossConfig(name="mse")
        with pytest.raises(ValueError, match="num_classes"):
            cfg()


class TestCrossEntropyLoss:
    def test_returns_callable(self):
        cfg = CrossEntropyLossConfig(name="cross_entropy")
        loss_fn = cfg()
        assert callable(loss_fn)

    def test_output_is_scalar_with_mean_reduction(self):
        cfg = CrossEntropyLossConfig(name="cross_entropy", reduction="mean")
        loss_fn = cfg()
        logits = torch.randn(8, 10)
        labels = torch.randint(0, 10, (8,))
        result = loss_fn(logits, labels)
        assert result.dim() == 0

    def test_output_is_per_sample_with_none_reduction(self):
        cfg = CrossEntropyLossConfig(name="cross_entropy", reduction="none")
        loss_fn = cfg()
        logits = torch.randn(8, 10)
        labels = torch.randint(0, 10, (8,))
        result = loss_fn(logits, labels)
        assert result.shape == (8,)

    def test_perfect_prediction_gives_low_loss(self):
        cfg = CrossEntropyLossConfig(name="cross_entropy", reduction="mean")
        loss_fn = cfg()
        # Very confident correct predictions
        logits = torch.zeros(3, 3)
        logits[0, 0] = 100.0
        logits[1, 1] = 100.0
        logits[2, 2] = 100.0
        labels = torch.tensor([0, 1, 2])
        result = loss_fn(logits, labels)
        assert result.item() < 1e-3

    def test_label_smoothing(self):
        cfg_no_smooth = CrossEntropyLossConfig(name="cross_entropy", label_smoothing=0.0)
        cfg_smooth = CrossEntropyLossConfig(name="cross_entropy", label_smoothing=0.1)
        loss_no_smooth = cfg_no_smooth()
        loss_smooth = cfg_smooth()

        logits = torch.randn(8, 10)
        labels = torch.randint(0, 10, (8,))
        # Label smoothing generally changes the loss value
        r1 = loss_no_smooth(logits, labels)
        r2 = loss_smooth(logits, labels)
        assert r1.item() != pytest.approx(r2.item(), abs=1e-4)

    def test_class_weights(self):
        weights = [1] * 5
        weights[0] = 10.0  # Heavily weight class 0
        cfg = CrossEntropyLossConfig(name="cross_entropy", weight=weights)
        loss_fn = cfg()
        logits = torch.randn(8, 5)
        labels = torch.zeros(8, dtype=torch.long)  # All class 0
        result = loss_fn(logits, labels)
        assert result.dim() == 0
        assert result.item() > 0.0
