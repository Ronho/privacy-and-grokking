import pytest
import torch

from privacy_and_grokking.scheduler.cosine_annealing import CosineAnnealingLRConfig
from privacy_and_grokking.scheduler.none import NoScheduler, NoSchedulerConfig


def _make_optimizer():
    """Helper to create a simple optimizer for scheduler tests."""
    model = torch.nn.Linear(10, 2)
    return torch.optim.SGD(model.parameters(), lr=0.1)


class TestNoSchedulerConfig:
    def test_returns_scheduler(self):
        cfg = NoSchedulerConfig()
        sched = cfg(_make_optimizer())
        assert isinstance(sched, NoScheduler)

    def test_step_does_nothing(self):
        cfg = NoSchedulerConfig()
        opt = _make_optimizer()
        initial_lr = opt.defaults["lr"]
        sched = cfg(opt)
        for _ in range(10):
            sched.step()
        assert opt.defaults["lr"] == initial_lr

    def test_accepts_extra_kwargs(self):
        cfg = NoSchedulerConfig()
        sched = cfg(_make_optimizer(), optimization_steps=100, last_epoch=-1)
        assert isinstance(sched, NoScheduler)


class TestCosineAnnealingLRConfig:
    def test_returns_scheduler(self):
        cfg = CosineAnnealingLRConfig(min_lr=1e-5)
        sched = cfg(_make_optimizer(), optimization_steps=100)
        assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_missing_optimization_steps_raises(self):
        cfg = CosineAnnealingLRConfig(min_lr=1e-5)
        with pytest.raises(ValueError, match="optimization_steps"):
            cfg(_make_optimizer())

    def test_lr_decreases_over_steps(self):
        cfg = CosineAnnealingLRConfig(min_lr=0.0)
        opt = _make_optimizer()
        sched = cfg(opt, optimization_steps=100)
        initial_lr = opt.param_groups[0]["lr"]
        for _ in range(50):
            sched.step()
        mid_lr = opt.param_groups[0]["lr"]
        assert mid_lr < initial_lr

    def test_lr_reaches_min_at_t_max(self):
        min_lr = 1e-5
        cfg = CosineAnnealingLRConfig(min_lr=min_lr)
        opt = _make_optimizer()
        sched = cfg(opt, optimization_steps=100)
        for _ in range(100):
            sched.step()
        final_lr = opt.param_groups[0]["lr"]
        assert final_lr == pytest.approx(min_lr, abs=1e-7)

    def test_custom_last_epoch(self):
        cfg = CosineAnnealingLRConfig(min_lr=0.0)
        opt = _make_optimizer()
        # PyTorch requires initial_lr in param_groups when last_epoch != -1
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
        # Start from step 50 of 100
        sched = cfg(opt, optimization_steps=100, last_epoch=49)
        sched.step()
        # After one more step (step 50 of 100), LR should be reduced
        lr_at_50 = opt.param_groups[0]["lr"]
        assert lr_at_50 < opt.defaults["lr"]

    def test_default_last_epoch_is_minus_one(self):
        cfg = CosineAnnealingLRConfig(min_lr=0.0)
        opt = _make_optimizer()
        sched = cfg(opt, optimization_steps=100)
        # At last_epoch=-1, LR should still be the initial LR
        assert opt.param_groups[0]["lr"] == opt.defaults["lr"]
