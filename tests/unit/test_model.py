import torch
import torch.nn as nn

from privacy_and_grokking.models.cnn import CNN, CNNConfig
from privacy_and_grokking.models.mlp import MLP, MLPConfig
from privacy_and_grokking.models.mlp_batchnorm import MLPBatchNorm, MLPBatchNormConfig

MNIST_INPUT_DIM = torch.Size([1, 28, 28])
CIFAR_INPUT_DIM = torch.Size([3, 32, 32])
NUM_CLASSES = 10


class TestMLPConfig:
    def test_returns_module(self):
        cfg = MLPConfig()
        model = cfg(input_dim=MNIST_INPUT_DIM, num_classes=NUM_CLASSES)
        assert isinstance(model, MLP)

    def test_name(self):
        cfg = MLPConfig()
        assert cfg.name == "mlp"

    def test_output_shape(self):
        cfg = MLPConfig()
        model = cfg(input_dim=MNIST_INPUT_DIM, num_classes=NUM_CLASSES)
        x = torch.randn(4, *MNIST_INPUT_DIM)
        out = model(x)
        assert out.shape == (4, NUM_CLASSES)

    def test_initialization_scale_none(self):
        cfg = MLPConfig()
        assert cfg.initialization_scale is None
        model = cfg(input_dim=MNIST_INPUT_DIM, num_classes=NUM_CLASSES)
        assert isinstance(model, nn.Module)

    def test_initialization_scale_applied(self):
        cfg = MLPConfig(initialization_scale=2.0)
        model_scaled = cfg(input_dim=MNIST_INPUT_DIM, num_classes=NUM_CLASSES)

        cfg_base = MLPConfig()
        torch.manual_seed(0)
        model_base = cfg_base(input_dim=MNIST_INPUT_DIM, num_classes=NUM_CLASSES)
        torch.manual_seed(0)
        model_scaled = cfg(input_dim=MNIST_INPUT_DIM, num_classes=NUM_CLASSES)

        for p_base, p_scaled in zip(
            model_base.parameters(), model_scaled.parameters(), strict=True
        ):
            torch.testing.assert_close(p_scaled, p_base * 2.0)

    def test_last_layer(self):
        cfg = MLPConfig()
        model = cfg(input_dim=MNIST_INPUT_DIM, num_classes=NUM_CLASSES)
        assert model.last_layer.out_features == NUM_CLASSES


class TestMLPBatchNormConfig:
    def test_returns_module(self):
        cfg = MLPBatchNormConfig()
        model = cfg(input_dim=MNIST_INPUT_DIM, num_classes=NUM_CLASSES)
        assert isinstance(model, MLPBatchNorm)

    def test_name(self):
        cfg = MLPBatchNormConfig()
        assert cfg.name == "mlp_batchnorm"

    def test_output_shape(self):
        cfg = MLPBatchNormConfig()
        model = cfg(input_dim=MNIST_INPUT_DIM, num_classes=NUM_CLASSES)
        model.eval()
        x = torch.randn(4, *MNIST_INPUT_DIM)
        out = model(x)
        assert out.shape == (4, NUM_CLASSES)

    def test_initialization_scale_applied(self):
        cfg = MLPBatchNormConfig(initialization_scale=0.5)
        cfg_base = MLPBatchNormConfig()

        torch.manual_seed(0)
        model_base = cfg_base(input_dim=MNIST_INPUT_DIM, num_classes=NUM_CLASSES)
        torch.manual_seed(0)
        model_scaled = cfg(input_dim=MNIST_INPUT_DIM, num_classes=NUM_CLASSES)

        for p_base, p_scaled in zip(
            model_base.parameters(), model_scaled.parameters(), strict=True
        ):
            torch.testing.assert_close(p_scaled, p_base * 0.5)

    def test_last_layer(self):
        cfg = MLPBatchNormConfig()
        model = cfg(input_dim=MNIST_INPUT_DIM, num_classes=NUM_CLASSES)
        assert model.last_layer.out_features == NUM_CLASSES


class TestCNNConfig:
    def test_returns_module(self):
        cfg = CNNConfig()
        model = cfg(input_dim=MNIST_INPUT_DIM, num_classes=NUM_CLASSES)
        assert isinstance(model, CNN)

    def test_name(self):
        cfg = CNNConfig()
        assert cfg.name == "cnn"

    def test_output_shape_mnist(self):
        cfg = CNNConfig()
        model = cfg(input_dim=MNIST_INPUT_DIM, num_classes=NUM_CLASSES)
        x = torch.randn(4, *MNIST_INPUT_DIM)
        out = model(x)
        assert out.shape == (4, NUM_CLASSES)

    def test_output_shape_cifar(self):
        cfg = CNNConfig()
        model = cfg(input_dim=CIFAR_INPUT_DIM, num_classes=NUM_CLASSES)
        x = torch.randn(4, *CIFAR_INPUT_DIM)
        out = model(x)
        assert out.shape == (4, NUM_CLASSES)

    def test_initialization_scale_applied(self):
        cfg = CNNConfig(initialization_scale=3.0)
        cfg_base = CNNConfig()

        torch.manual_seed(0)
        model_base = cfg_base(input_dim=MNIST_INPUT_DIM, num_classes=NUM_CLASSES)
        torch.manual_seed(0)
        model_scaled = cfg(input_dim=MNIST_INPUT_DIM, num_classes=NUM_CLASSES)

        for p_base, p_scaled in zip(
            model_base.parameters(), model_scaled.parameters(), strict=True
        ):
            torch.testing.assert_close(p_scaled, p_base * 3.0)

    def test_last_layer(self):
        cfg = CNNConfig()
        model = cfg(input_dim=MNIST_INPUT_DIM, num_classes=NUM_CLASSES)
        assert model.last_layer.out_features == NUM_CLASSES


class TestModelDiscriminator:
    """Test that the discriminated union type works for JSON parsing."""

    def test_mlp_from_dict(self):
        from pydantic import TypeAdapter

        from privacy_and_grokking.models import Model

        adapter = TypeAdapter(Model)
        cfg = adapter.validate_python({"name": "mlp"})
        assert isinstance(cfg, MLPConfig)

    def test_mlp_batchnorm_from_dict(self):
        from pydantic import TypeAdapter

        from privacy_and_grokking.models import Model

        adapter = TypeAdapter(Model)
        cfg = adapter.validate_python({"name": "mlp_batchnorm"})
        assert isinstance(cfg, MLPBatchNormConfig)

    def test_cnn_from_dict(self):
        from pydantic import TypeAdapter

        from privacy_and_grokking.models import Model

        adapter = TypeAdapter(Model)
        cfg = adapter.validate_python({"name": "cnn"})
        assert isinstance(cfg, CNNConfig)

    def test_with_initialization_scale(self):
        from pydantic import TypeAdapter

        from privacy_and_grokking.models import Model

        adapter = TypeAdapter(Model)
        cfg = adapter.validate_python({"name": "mlp", "initialization_scale": 8.0})
        assert isinstance(cfg, MLPConfig)
        assert cfg.initialization_scale == 8.0
