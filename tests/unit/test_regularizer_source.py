import torch

from privacy_and_grokking.loss.regularizer_source.gaussian import GaussianNoiseConfig
from privacy_and_grokking.loss.regularizer_source.salt_and_pepper import SaltAndPepperNoiseConfig


class TestGaussianNoiseSource:
    def test_returns_callable(self):
        cfg = GaussianNoiseConfig(num_noisy_samples=1, mean=0.0, std=1.0)
        func = cfg()
        assert callable(func)

    def test_output_shape_single_copy(self):
        cfg = GaussianNoiseConfig(num_noisy_samples=1, mean=0.0, std=0.5)
        func = cfg()
        x = torch.randn(4, 3, 8, 8)
        out = func(x)
        assert out.shape == (4, 3, 8, 8)

    def test_output_shape_multiple_copies(self):
        cfg = GaussianNoiseConfig(num_noisy_samples=3, mean=0.0, std=0.5)
        func = cfg()
        x = torch.randn(4, 3, 8, 8)
        out = func(x)
        # Each sample repeated 3 times
        assert out.shape == (12, 3, 8, 8)

    def test_zero_noise_returns_input(self):
        cfg = GaussianNoiseConfig(num_noisy_samples=1, mean=0.0, std=0.0)
        func = cfg()
        x = torch.randn(4, 3, 8, 8)
        out = func(x)
        assert torch.equal(out, x)

    def test_nonzero_noise_differs_from_input(self):
        cfg = GaussianNoiseConfig(num_noisy_samples=1, mean=0.0, std=1.0)
        func = cfg()
        x = torch.randn(4, 3, 8, 8)
        out = func(x)
        assert not torch.equal(out, x)

    def test_output_is_detached(self):
        cfg = GaussianNoiseConfig(num_noisy_samples=1, mean=0.0, std=1.0)
        func = cfg()
        x = torch.randn(4, 3, 8, 8, requires_grad=True)
        out = func(x)
        assert not out.requires_grad

    def test_different_calls_produce_different_noise(self):
        cfg = GaussianNoiseConfig(num_noisy_samples=1, mean=0.0, std=1.0)
        func = cfg()
        x = torch.randn(4, 3, 8, 8)
        out1 = func(x)
        out2 = func(x)
        assert not torch.equal(out1, out2)


class TestSaltAndPepperNoiseSource:
    def test_returns_callable(self):
        cfg = SaltAndPepperNoiseConfig(num_noisy_samples=1, fraction=0.1)
        func = cfg()
        assert callable(func)

    def test_output_shape_single_copy(self):
        cfg = SaltAndPepperNoiseConfig(num_noisy_samples=1, fraction=0.1)
        func = cfg()
        x = torch.randn(4, 3, 8, 8)
        out = func(x)
        assert out.shape == (4, 3, 8, 8)

    def test_output_shape_multiple_copies(self):
        cfg = SaltAndPepperNoiseConfig(num_noisy_samples=2, fraction=0.1)
        func = cfg()
        x = torch.randn(4, 3, 8, 8)
        out = func(x)
        assert out.shape == (8, 3, 8, 8)

    def test_zero_fraction_returns_input(self):
        cfg = SaltAndPepperNoiseConfig(num_noisy_samples=1, fraction=0.0)
        func = cfg()
        x = torch.randn(4, 3, 8, 8)
        out = func(x)
        assert torch.equal(out, x)

    def test_none_fraction_returns_input(self):
        cfg = SaltAndPepperNoiseConfig(num_noisy_samples=1, fraction=None)
        func = cfg()
        x = torch.randn(4, 3, 8, 8)
        out = func(x)
        assert torch.equal(out, x)

    def test_nonzero_fraction_modifies_input(self):
        cfg = SaltAndPepperNoiseConfig(num_noisy_samples=1, fraction=0.5)
        func = cfg()
        x = torch.randn(4, 3, 8, 8)
        out = func(x)
        assert not torch.equal(out, x)

    def test_output_values_are_salt_or_pepper_or_original(self):
        cfg = SaltAndPepperNoiseConfig(num_noisy_samples=1, fraction=1.0)
        func = cfg()
        x = torch.tensor([[[[0.2, 0.5, 0.8]]]])  # (1, 1, 1, 3)
        out = func(x)
        lo = x.min().item()
        hi = x.max().item()
        # With fraction=1.0, all values should be either lo or hi
        for val in out.flatten().tolist():
            assert val == lo or val == hi

    def test_output_is_detached(self):
        cfg = SaltAndPepperNoiseConfig(num_noisy_samples=1, fraction=0.5)
        func = cfg()
        x = torch.randn(4, 3, 8, 8, requires_grad=True)
        out = func(x)
        assert not out.requires_grad
