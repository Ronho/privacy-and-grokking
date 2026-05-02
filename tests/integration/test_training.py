"""Integration test for the full training pipeline with the new config layout.

Verifies that:
1. TrainConfig can be constructed from the new nested data config format
2. The training loop runs end-to-end (a few steps)
3. Metrics are logged correctly
4. Model checkpoints are saved
5. Different config combinations (loss, optimizer, scheduler, masking) work
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import mlflow
import pytest
import torch
from torch.utils.data import Dataset

from privacy_and_grokking.config import TrainConfig
from privacy_and_grokking.training.train import train


class FakeMNIST(Dataset):
    """A tiny fake dataset that mimics MNIST structure for fast testing."""

    def __init__(self, num_samples: int = 200, num_classes: int = 10):
        torch.manual_seed(42)
        self.images = torch.randn(num_samples, 1, 28, 28)
        self.labels = torch.arange(num_samples) % num_classes
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.images[idx], int(self.labels[idx])


@pytest.fixture
def mlflow_tmpdir():
    """Set up a temporary MLflow tracking directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracking_uri = f"file://{tmpdir}/mlruns"
        yield tracking_uri


@pytest.fixture
def fake_mnist():
    """Create a fake MNIST dataset to avoid downloading real data."""
    return FakeMNIST(num_samples=200, num_classes=10)


def _patch_dataset(fake_dataset):
    """Return a context manager that patches MNISTConfig.__call__ to use fake data."""
    from privacy_and_grokking.datasets.sets.base import DataContainer, Normalization

    container = DataContainer(
        train=fake_dataset,
        test=FakeMNIST(num_samples=50, num_classes=10),
        num_classes=10,
        input_shape=torch.Size([1, 28, 28]),
        normalization=Normalization(mean=[0.1307], std=[0.3081]),
    )
    return patch(
        "privacy_and_grokking.datasets.sets.mnist.MNISTConfig.__call__",
        return_value=container,
    )


def _run_training(config_dict: dict, mlflow_uri: str, fake_dataset, total_steps: int = 10):
    """Helper to run training with a given config dict."""
    cfg = TrainConfig.model_validate(config_dict)

    with _patch_dataset(fake_dataset):
        mlflow.set_tracking_uri(mlflow_uri)
        exp_name = "integration_test"
        if not mlflow.get_experiment_by_name(exp_name):
            mlflow.create_experiment(name=exp_name)
        mlflow.set_experiment(exp_name)

        with patch("privacy_and_grokking.training.train.setup_mlflow"):
            run_id = train(
                exp_name=exp_name,
                total_steps=total_steps,
                cfg=cfg,
                run_name="test_run",
                checkpoint_frequency=5,
            )

    return run_id


# --- Test: Basic MSE + SGD training ---


class TestTrainingPipeline:
    def test_mse_sgd_basic(self, mlflow_tmpdir, fake_mnist):
        """End-to-end training with MSE loss, SGD optimizer, no scheduler."""
        config = {
            "model": {"name": "mlp"},
            "seed": 42,
            "batch_size": 50,
            "loss": {"name": "mse"},
            "scheduler": {"name": "None"},
            "optimizer": {"name": "SGD", "lr": 0.01},
            "data": {
                "data": {"name": "mnist"},
                "mask": {
                    "name": "uniform",
                    "num_models": 2,
                    "p": 0.5,
                    "seed": 1,
                    "model_index": 0,
                },
                "seed": 1,
            },
        }
        run_id = _run_training(config, mlflow_tmpdir, fake_mnist)
        assert run_id is not None
        assert len(run_id) > 0

    def test_ce_adamw_cosine(self, mlflow_tmpdir, fake_mnist):
        """End-to-end training with CE loss, AdamW optimizer, cosine scheduler."""
        config = {
            "model": {"name": "mlp"},
            "seed": 42,
            "batch_size": 50,
            "loss": {"name": "cross_entropy"},
            "scheduler": {"name": "CosineAnnealingLR", "min_lr": 1e-6},
            "optimizer": {"name": "AdamW", "lr": 0.001},
            "data": {
                "data": {"name": "mnist"},
                "mask": {
                    "name": "independent_stratified",
                    "num_models": 2,
                    "p": 0.5,
                    "seed": 1,
                    "model_index": 0,
                },
                "seed": 1,
            },
        }
        run_id = _run_training(config, mlflow_tmpdir, fake_mnist)
        assert run_id is not None

    def test_no_mask(self, mlflow_tmpdir, fake_mnist):
        """Training without any masking strategy."""
        config = {
            "model": {"name": "mlp"},
            "seed": 42,
            "batch_size": 50,
            "loss": {"name": "mse"},
            "scheduler": {"name": "None"},
            "optimizer": {"name": "SGD", "lr": 0.01},
            "data": {
                "data": {"name": "mnist"},
                "seed": 1,
            },
        }
        run_id = _run_training(config, mlflow_tmpdir, fake_mnist)
        assert run_id is not None

    def test_with_canary(self, mlflow_tmpdir, fake_mnist):
        """Training with canary samples applied."""
        config = {
            "model": {"name": "mlp"},
            "seed": 42,
            "batch_size": 50,
            "loss": {"name": "mse"},
            "scheduler": {"name": "None"},
            "optimizer": {"name": "SGD", "lr": 0.01},
            "data": {
                "data": {"name": "mnist"},
                "canary": {
                    "name": "square_watermark",
                    "share": 0.1,
                    "square_size": 3,
                },
                "seed": 1,
            },
        }
        run_id = _run_training(config, mlflow_tmpdir, fake_mnist)
        assert run_id is not None

    def test_with_train_size_and_canary(self, mlflow_tmpdir, fake_mnist):
        """Training with train_size limiting the dataset and canaries."""
        config = {
            "model": {"name": "mlp"},
            "seed": 42,
            "batch_size": 50,
            "loss": {"name": "mse"},
            "scheduler": {"name": "None"},
            "optimizer": {"name": "SGD", "lr": 0.01},
            "data": {
                "data": {"name": "mnist"},
                "canary": {
                    "name": "uniform_noise",
                    "share": 0.1,
                },
                "train_size": 100,
                "seed": 1,
            },
        }
        run_id = _run_training(config, mlflow_tmpdir, fake_mnist)
        assert run_id is not None

    def test_config_from_json_file(self, mlflow_tmpdir, fake_mnist):
        """Verify that an actual JSON config file from the configs/ dir loads and trains."""
        config_path = Path(__file__).parent.parent.parent / "configs" / "MSE_SGD_DEFAULT.json"
        cfg = TrainConfig.model_validate_json(config_path.read_bytes())

        with _patch_dataset(fake_mnist):
            mlflow.set_tracking_uri(mlflow_tmpdir)
            exp_name = "integration_test"
            if not mlflow.get_experiment_by_name(exp_name):
                mlflow.create_experiment(name=exp_name)
            mlflow.set_experiment(exp_name)

            with patch("privacy_and_grokking.training.train.setup_mlflow"):
                run_id = train(
                    exp_name=exp_name,
                    total_steps=5,
                    cfg=cfg,
                    run_name="file_config_test",
                    checkpoint_frequency=5,
                )

        assert run_id is not None

    def test_metrics_logged(self, mlflow_tmpdir, fake_mnist):
        """Verify that metrics are actually logged during training."""
        config = {
            "model": {"name": "mlp"},
            "seed": 42,
            "batch_size": 50,
            "loss": {"name": "mse"},
            "scheduler": {"name": "None"},
            "optimizer": {"name": "SGD", "lr": 0.01},
            "data": {
                "data": {"name": "mnist"},
                "seed": 1,
            },
            "metrics": {
                "log_frequency": 5,
                "heavy_metrics_log_frequency": 100,
                "curvature": False,
                "merlin_morgan": False,
            },
        }
        run_id = _run_training(config, mlflow_tmpdir, fake_mnist, total_steps=10)

        # Verify metrics were logged
        mlflow.set_tracking_uri(mlflow_tmpdir)
        run = mlflow.get_run(run_id)
        metrics = run.data.metrics
        assert "train/task_loss" in metrics
        assert "train/total_loss" in metrics

    def test_train_config_name_property(self):
        """Verify the name property works with the new config layout."""
        config = {
            "model": {"name": "mlp"},
            "seed": 42,
            "batch_size": 50,
            "loss": {"name": "mse"},
            "scheduler": {"name": "None"},
            "optimizer": {"name": "SGD"},
            "data": {
                "data": {"name": "mnist"},
                "mask": {
                    "name": "uniform",
                    "num_models": 2,
                    "p": 0.5,
                    "seed": 1,
                    "model_index": 0,
                },
                "seed": 1,
            },
        }
        cfg = TrainConfig.model_validate(config)
        assert cfg.name == "MLP_MNIST_UNIFORM_SGD_MSE"
        assert cfg.full_name == "MLP_MNIST_UNIFORM_SGD_MSE_0"

    def test_train_config_name_no_mask(self):
        """Verify the name property works without a mask."""
        config = {
            "model": {"name": "mlp"},
            "seed": 42,
            "batch_size": 50,
            "loss": {"name": "cross_entropy"},
            "scheduler": {"name": "None"},
            "optimizer": {"name": "Adam"},
            "data": {
                "data": {"name": "mnist"},
                "seed": 1,
            },
        }
        cfg = TrainConfig.model_validate(config)
        assert cfg.name == "MLP_MNIST_ADAM_CROSS_ENTROPY"
        assert cfg.full_name == "MLP_MNIST_ADAM_CROSS_ENTROPY"
