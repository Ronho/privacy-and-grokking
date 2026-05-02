import torch

from privacy_and_grokking.optimizer.adam import AdamConfig
from privacy_and_grokking.optimizer.adamw import AdamWConfig
from privacy_and_grokking.optimizer.rmsprop import RMSpropConfig
from privacy_and_grokking.optimizer.sgd import SGDConfig


def _make_params():
    """Helper to create a simple parameter list for optimizer tests."""
    return torch.nn.Linear(10, 2).parameters()


class TestAdamConfig:
    def test_returns_optimizer(self):
        cfg = AdamConfig()
        opt = cfg(_make_params())
        assert isinstance(opt, torch.optim.Adam)

    def test_default_lr(self):
        cfg = AdamConfig()
        opt = cfg(_make_params())
        assert opt.defaults["lr"] == 0.001

    def test_custom_lr(self):
        cfg = AdamConfig(lr=0.01)
        opt = cfg(_make_params())
        assert opt.defaults["lr"] == 0.01

    def test_custom_betas(self):
        cfg = AdamConfig(betas=(0.8, 0.99))
        opt = cfg(_make_params())
        assert opt.defaults["betas"] == (0.8, 0.99)

    def test_custom_weight_decay(self):
        cfg = AdamConfig(weight_decay=0.1)
        opt = cfg(_make_params())
        assert opt.defaults["weight_decay"] == 0.1

    def test_amsgrad(self):
        cfg = AdamConfig(amsgrad=True)
        opt = cfg(_make_params())
        assert opt.defaults["amsgrad"] is True


class TestAdamWConfig:
    def test_returns_optimizer(self):
        cfg = AdamWConfig()
        opt = cfg(_make_params())
        assert isinstance(opt, torch.optim.AdamW)

    def test_default_lr(self):
        cfg = AdamWConfig()
        opt = cfg(_make_params())
        assert opt.defaults["lr"] == 0.001

    def test_default_weight_decay(self):
        cfg = AdamWConfig()
        opt = cfg(_make_params())
        assert opt.defaults["weight_decay"] == 0.01

    def test_custom_lr(self):
        cfg = AdamWConfig(lr=0.05)
        opt = cfg(_make_params())
        assert opt.defaults["lr"] == 0.05

    def test_custom_eps(self):
        cfg = AdamWConfig(eps=1e-6)
        opt = cfg(_make_params())
        assert opt.defaults["eps"] == 1e-6


class TestRMSpropConfig:
    def test_returns_optimizer(self):
        cfg = RMSpropConfig()
        opt = cfg(_make_params())
        assert isinstance(opt, torch.optim.RMSprop)

    def test_default_lr(self):
        cfg = RMSpropConfig()
        opt = cfg(_make_params())
        assert opt.defaults["lr"] == 0.01

    def test_custom_momentum(self):
        cfg = RMSpropConfig(momentum=0.9)
        opt = cfg(_make_params())
        assert opt.defaults["momentum"] == 0.9

    def test_centered(self):
        cfg = RMSpropConfig(centered=True)
        opt = cfg(_make_params())
        assert opt.defaults["centered"] is True


class TestSGDConfig:
    def test_returns_optimizer(self):
        cfg = SGDConfig()
        opt = cfg(_make_params())
        assert isinstance(opt, torch.optim.SGD)

    def test_default_lr(self):
        cfg = SGDConfig()
        opt = cfg(_make_params())
        assert opt.defaults["lr"] == 0.001

    def test_custom_momentum(self):
        cfg = SGDConfig(momentum=0.9)
        opt = cfg(_make_params())
        assert opt.defaults["momentum"] == 0.9

    def test_nesterov_requires_momentum(self):
        cfg = SGDConfig(momentum=0.9, nesterov=True)
        opt = cfg(_make_params())
        assert opt.defaults["nesterov"] is True

    def test_custom_weight_decay(self):
        cfg = SGDConfig(weight_decay=1e-4)
        opt = cfg(_make_params())
        assert opt.defaults["weight_decay"] == 1e-4
