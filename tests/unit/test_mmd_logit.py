"""Unit tests for the differentiable MMD logit regularizer.

Tests validate the implementation against ignite.metrics.MaximumMeanDiscrepancy
where applicable and check differentiability and correctness.

Ignite's metric uses an unbiased U-statistic (see source):
    XX = (K_XX.sum() - n) / (n*(n-1))
    YY = (K_YY.sum() - m) / (m*(m-1))
    XY = K_XY.mean()
    MMD² = XX + YY - 2*XY

Our module ports this formula with:
- No .detach() (gradients retained)
- Support for n ≠ m (generalised squared-norm expansion)
- Optional median-heuristic bandwidth
"""

import pytest
import torch

from privacy_and_grokking.losses import MMDLogitRegularizer
from privacy_and_grokking.config.loss import MMDLogitRegularizerConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ignite_mmd2(x: torch.Tensor, y: torch.Tensor, var: float) -> float:
    """Reference implementation matching ignite's MaximumMeanDiscrepancy.

    Requires x.shape == y.shape (as Ignite does).
    Returns MMD² as a float (no gradient).
    """
    with torch.no_grad():
        xx = torch.mm(x, x.t())
        yy = torch.mm(y, y.t())
        zz = torch.mm(x, y.t())
        rx = xx.diag().unsqueeze(0).expand_as(xx)
        ry = yy.diag().unsqueeze(0).expand_as(yy)
        dxx = rx.t() + rx - 2.0 * xx
        dyy = ry.t() + ry - 2.0 * yy
        dxy = rx.t() + ry - 2.0 * zz
        n = x.shape[0]
        XX = (torch.exp(-0.5 * dxx / var).sum() - n) / (n * (n - 1))
        YY = (torch.exp(-0.5 * dyy / var).sum() - n) / (n * (n - 1))
        XY = torch.exp(-0.5 * dxy / var).mean()
        return (XX + YY - 2.0 * XY).item()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMMDLogitRegularizer:

    # ------------------------------------------------------------------
    # Numerical parity with Ignite (fixed var, equal batch sizes)
    # ------------------------------------------------------------------

    def test_matches_ignite_formula_equal_sizes(self):
        """Result must match Ignite's formula exactly when n == m."""
        var = 2.0
        reg = MMDLogitRegularizer(var=var)
        x = torch.randn(16, 10)
        y = torch.randn(16, 10)

        our_val     = reg(x, y).item()
        ignite_val  = _ignite_mmd2(x, y, var)
        assert abs(our_val - ignite_val) < 1e-5, (
            f"Mismatch vs Ignite: ours={our_val:.7f}, ignite={ignite_val:.7f}"
        )

    def test_matches_ignite_multiple_var_values(self):
        """Parity holds across several bandwidth values."""
        x = torch.randn(32, 10)
        y = torch.randn(32, 10)
        for var in (0.1, 0.5, 1.0, 5.0, 10.0):
            reg = MMDLogitRegularizer(var=var)
            ours   = reg(x, y).item()
            ignite = _ignite_mmd2(x, y, var)
            assert abs(ours - ignite) < 1e-5, (
                f"Mismatch at var={var}: ours={ours:.7f}, ignite={ignite:.7f}"
            )

    # ------------------------------------------------------------------
    # Symmetry and degenerate cases
    # ------------------------------------------------------------------

    def test_symmetric(self):
        """MMD² is symmetric: MMD²(x, y) == MMD²(y, x)."""
        reg = MMDLogitRegularizer(var=1.0)
        x = torch.randn(32, 10)
        y = torch.randn(32, 10)
        assert abs(reg(x, y).item() - reg(y, x).item()) < 1e-5

    def test_near_zero_for_same_distribution(self):
        """MMD² should be small (close to zero) for two draws from the same distribution.

        Note: passing the *exact same tensor* to both arguments gives a slightly
        negative value with the unbiased U-statistic.  This is expected because
        the diagonal (k(x_i, x_i) = 1) is included in the cross-term K_XY but
        excluded from the within-group terms K_XX / K_YY.  The estimator has
        E[MMD²] = 0 in expectation over random draws — it does *not* evaluate
        to exactly 0 on any single pair of samples.  We just confirm the
        magnitude is small and consistent with Ignite's formula.
        """
        reg = MMDLogitRegularizer(var=1.0)
        torch.manual_seed(0)
        x = torch.randn(128, 10)
        y = torch.randn(128, 10)       # independent draw from same distribution
        val = reg(x, y).item()
        # Unbiased U-stat; should be close to 0 (within ~0.05 for n=128)
        assert abs(val) < 0.1, f"MMD² unexpectedly large for same distribution: {val}"

    def test_non_negative_for_separated_distributions(self):
        """MMD² > 0 when distributions are clearly separated."""
        reg = MMDLogitRegularizer(var=1.0)
        x = torch.randn(32, 10) - 10.0
        y = torch.randn(32, 10) + 10.0
        assert reg(x, y).item() > 0.0

    # ------------------------------------------------------------------
    # Different batch sizes (n ≠ m)
    # ------------------------------------------------------------------

    def test_different_batch_sizes(self):
        """Module must handle n ≠ m (e.g. mini-batch vs. 3/class proxy set)."""
        reg = MMDLogitRegularizer(var=1.0)
        x = torch.randn(64, 10)     # training mini-batch
        y = torch.randn(30, 10)     # 3 samples × 10 classes
        val = reg(x, y)
        assert val.isfinite(), f"Non-finite MMD² for n != m: {val}"

    def test_small_proxy_size(self):
        """Edge case: very small proxy set (1 sample per class = 10 total)."""
        reg = MMDLogitRegularizer(var=1.0)
        x = torch.randn(200, 10)
        y = torch.randn(10, 10)
        val = reg(x, y)
        assert val.isfinite()

    # ------------------------------------------------------------------
    # Gradient flow
    # ------------------------------------------------------------------

    def test_gradient_flows_through_member_logits(self):
        reg = MMDLogitRegularizer(var=1.0)
        x = torch.randn(32, 10, requires_grad=True)
        y = torch.randn(32, 10)
        reg(x, y).backward()
        assert x.grad is not None and x.grad.abs().sum() > 0

    def test_gradient_flows_through_nonmember_logits(self):
        reg = MMDLogitRegularizer(var=1.0)
        x = torch.randn(32, 10)
        y = torch.randn(32, 10, requires_grad=True)
        reg(x, y).backward()
        assert y.grad is not None and y.grad.abs().sum() > 0

    def test_gradient_flows_unequal_sizes(self):
        """Gradients must propagate correctly when n ≠ m."""
        reg = MMDLogitRegularizer(var=1.0)
        x = torch.randn(64, 10, requires_grad=True)
        y = torch.randn(30, 10, requires_grad=True)
        reg(x, y).backward()
        assert x.grad is not None and x.grad.abs().sum() > 0
        assert y.grad is not None and y.grad.abs().sum() > 0

    # ------------------------------------------------------------------
    # Median bandwidth heuristic
    # ------------------------------------------------------------------

    def test_median_bandwidth_finite(self):
        """var=None (median heuristic) must produce a finite scalar."""
        reg = MMDLogitRegularizer(var=None)
        x = torch.randn(32, 10)
        y = torch.randn(32, 10)
        val = reg(x, y)
        assert val.isfinite()

    def test_median_bandwidth_gradient_still_flows(self):
        """Gradient through logits still works when using median bandwidth."""
        reg = MMDLogitRegularizer(var=None)
        x = torch.randn(32, 10, requires_grad=True)
        y = torch.randn(32, 10)
        reg(x, y).backward()
        assert x.grad is not None and x.grad.abs().sum() > 0

    # ------------------------------------------------------------------
    # Config integration
    # ------------------------------------------------------------------

    def test_config_build_returns_regularizer(self):
        cfg = MMDLogitRegularizerConfig(weight=0.1, samples_per_class=3)
        reg = cfg.build()
        assert isinstance(reg, MMDLogitRegularizer)

    def test_config_build_with_fixed_var(self):
        cfg = MMDLogitRegularizerConfig(weight=0.1, var=2.0, samples_per_class=3)
        reg = cfg.build()
        assert reg._fixed_var is not None
        assert abs(reg._fixed_var.item() - 2.0) < 1e-6

    def test_config_default_var_is_none(self):
        cfg = MMDLogitRegularizerConfig()
        reg = cfg.build()
        assert reg._fixed_var is None
