import torch

from privacy_and_grokking.loss.loss import CrossEntropyLossConfig, MSELossConfig


class TestMSELoss:
    def test_returns_scalar(self):
        config = MSELossConfig(num_classes=10)
        loss_fn = config()

        logits = torch.randn(4, 10)
        labels = torch.tensor([0, 3, 5, 9])

        result = loss_fn(logits, labels)

        assert result.shape == ()
        assert result.dtype == torch.float32

    def test_perfect_prediction_is_zero(self):
        config = MSELossConfig(num_classes=3)
        loss_fn = config()

        one_hot = torch.eye(3)
        labels = torch.tensor([0, 1, 2])

        result = loss_fn(one_hot, labels)

        assert torch.isclose(result, torch.tensor(0.0), atol=1e-7)

    def test_reduction_none(self):
        config = MSELossConfig(num_classes=5, reduction="none")
        loss_fn = config()

        logits = torch.randn(3, 5)
        labels = torch.tensor([1, 2, 4])

        result = loss_fn(logits, labels)

        assert result.shape == (3, 5)

    def test_reduction_sum(self):
        config = MSELossConfig(num_classes=4, reduction="sum")
        loss_fn = config()

        logits = torch.randn(2, 4)
        labels = torch.tensor([0, 3])

        result = loss_fn(logits, labels)

        assert result.shape == ()


class TestCrossEntropyLoss:
    def test_returns_scalar(self):
        config = CrossEntropyLossConfig()
        loss_fn = config()

        logits = torch.randn(4, 10)
        labels = torch.tensor([0, 3, 5, 9])

        result = loss_fn(logits, labels)

        assert result.shape == ()
        assert result.dtype == torch.float32

    def test_perfect_prediction_is_near_zero(self):
        config = CrossEntropyLossConfig()
        loss_fn = config()

        logits = torch.zeros(3, 3)
        logits[0, 0] = 100.0
        logits[1, 1] = 100.0
        logits[2, 2] = 100.0
        labels = torch.tensor([0, 1, 2])

        result = loss_fn(logits, labels)

        assert result < 1e-4

    def test_reduction_none(self):
        config = CrossEntropyLossConfig(reduction="none")
        loss_fn = config()

        logits = torch.randn(3, 5)
        labels = torch.tensor([1, 2, 4])

        result = loss_fn(logits, labels)

        assert result.shape == (3,)

    def test_reduction_sum(self):
        config = CrossEntropyLossConfig(reduction="sum")
        loss_fn = config()

        logits = torch.randn(2, 4)
        labels = torch.tensor([0, 3])

        result = loss_fn(logits, labels)

        assert result.shape == ()


class TestCrossEntropyWeightSerialization:
    def test_validator_converts_list_to_tensor(self):
        config = CrossEntropyLossConfig(weight=[1.0, 2.0, 3.0])

        assert isinstance(config.weight, torch.Tensor)
        assert torch.equal(config.weight, torch.tensor([1.0, 2.0, 3.0]))

    def test_validator_passes_tensor_through(self):
        t = torch.tensor([0.5, 1.5])
        config = CrossEntropyLossConfig(weight=t)

        assert isinstance(config.weight, torch.Tensor)
        assert torch.equal(config.weight, t)

    def test_validator_accepts_none(self):
        config = CrossEntropyLossConfig(weight=None)

        assert config.weight is None

    def test_serializer_converts_tensor_to_list(self):
        config = CrossEntropyLossConfig(weight=[1.0, 2.0, 3.0])
        dumped = config.model_dump()

        assert dumped["weight"] == [1.0, 2.0, 3.0]

    def test_serializer_handles_none(self):
        config = CrossEntropyLossConfig()
        dumped = config.model_dump()

        assert dumped["weight"] is None

    def test_round_trip_via_json(self):
        config = CrossEntropyLossConfig(weight=[1.0, 2.0, 3.0])
        json_str = config.model_dump_json()
        restored = CrossEntropyLossConfig.model_validate_json(json_str)

        assert isinstance(restored.weight, torch.Tensor)
        assert torch.equal(restored.weight, torch.tensor([1.0, 2.0, 3.0]))
