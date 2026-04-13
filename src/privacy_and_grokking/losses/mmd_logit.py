"""Differentiable two-distribution MMD regularizer on model logit vectors.

Ports the kernel computation from ``ignite.metrics.MaximumMeanDiscrepancy``
(pytorch-ignite) into a differentiable ``nn.Module`` suitable for use as a
training regularization term.

Ignite's ``MaximumMeanDiscrepancy`` calls ``.detach()`` on its inputs inside
``update()``, making it an evaluation-only metric that cannot propagate
gradients through the model.  This module exposes the exact same Gaussian RBF
kernel and unbiased U-statistic formula from Ignite's source, but retains
the autograd graph so that minimising the penalty updates model weights.

Reference implementation
------------------------
Source: ignite/metrics/maximum_mean_discrepancy.py (pytorch-ignite)
URL:    https://docs.pytorch.org/ignite/_modules/ignite/metrics/\
        maximum_mean_discrepancy.html

Ignite's kernel (reproduced for comparison)::

    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)
    dxx = rx.t() + rx  - 2.0 * xx
    dyy = ry.t() + ry  - 2.0 * yy
    dxy = rx.t() + ry  - 2.0 * zz     # requires n == m

    XX  = (exp(-0.5 * dxx / var).sum() - n) / (n*(n-1))  # unbiased U-stat
    YY  = (exp(-0.5 * dyy / var).sum() - n) / (n*(n-1))
    XY  = exp(-0.5 * dxy / var).sum()  / (n*n)           # biased cross-term
    MMD² = XX + YY - 2*XY

Departures from Ignite
----------------------
* No ``.detach()`` — gradients flow through ``f_member`` and ``f_nonmember``.
* Squared-norm expansion is rewritten to support different batch sizes
  ``n ≠ m`` (e.g. full training batch vs. small per-class proxy set).
* ``var=None`` (default): bandwidth σ² is determined via the median pairwise
  squared-distance heuristic, computed inside ``torch.no_grad()`` so it is
  treated as a fixed hyperparameter rather than a trainable parameter.
* ``var`` is σ² (same naming convention as Ignite).

Mathematical summary
--------------------
Feature:  f(x; w) = model logits ∈ ℝ^C.
Kernel:   k(a, b) = exp(−‖a − b‖² / (2σ²))   (Gaussian RBF).

    MMD²(P, Q) ≈ U_XX + U_YY − 2·C_XY

where U_XX, U_YY are unbiased U-statistics (diagonal excluded) and C_XY is the
unbiased cross-term mean.  Minimising MMD² pushes the model to produce the same
logit distribution on member and proxy non-member samples, reducing the signal
available to a logit-based membership inference attack.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _mmd2_unbiased(
    x: torch.Tensor,
    y: torch.Tensor,
    var: torch.Tensor,
) -> torch.Tensor:
    """Differentiable unbiased MMD² between rows of x (n, C) and y (m, C).

    Kernel computation ported from ``ignite.metrics.MaximumMeanDiscrepancy``
    with the following changes:
      - No ``.detach()`` on inputs.
      - Squared-norm expansion generalised to n ≠ m via broadcasting.

    Args:
        x:   Member logit matrix,     shape (n, C).
        y:   Non-member logit matrix, shape (m, C).
        var: Scalar bandwidth σ² > 0.

    Returns:
        Scalar MMD² estimate (may be slightly negative due to U-statistic
        variance at small sample sizes).
    """
    n = x.shape[0]
    m = y.shape[0]

    # ── squared-norm rows ────────────────────────────────────────────────────
    # rx[i] = ‖x[i]‖²,  shape (n, 1)
    # ry[j] = ‖y[j]‖²,  shape (m, 1)
    rx = (x * x).sum(dim=1, keepdim=True)
    ry = (y * y).sum(dim=1, keepdim=True)

    # ── pairwise squared L2 distances ────────────────────────────────────────
    # d[i,j] = ‖x[i] - x[j]‖² = ‖x[i]‖² + ‖x[j]‖² - 2 x[i]·x[j]
    dxx = rx + rx.t() - 2.0 * torch.mm(x, x.t())  # (n, n)
    dyy = ry + ry.t() - 2.0 * torch.mm(y, y.t())  # (m, m)
    dxy = rx + ry.t() - 2.0 * torch.mm(x, y.t())  # (n, m)  — broadcast OK

    # ── RBF kernel ───────────────────────────────────────────────────────────
    K_XX = torch.exp(-0.5 * dxx / var)  # (n, n)
    K_YY = torch.exp(-0.5 * dyy / var)  # (m, m)
    K_XY = torch.exp(-0.5 * dxy / var)  # (n, m)

    # ── unbiased U-statistics (Ignite formula, generalised to n ≠ m) ─────────
    # Within-group: subtract diagonal (k(x_i, x_i) = 1 always)
    u_XX = (K_XX.sum() - n) / (n * (n - 1))
    u_YY = (K_YY.sum() - m) / (m * (m - 1))
    # Cross-term: no diagonal to exclude (x and y are independent sets)
    u_XY = K_XY.mean()

    return u_XX + u_YY - 2.0 * u_XY


def _median_var(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Median pairwise squared distance heuristic for σ².

    Concatenates all rows, computes upper-triangle pairwise squared distances,
    returns the median.  Run inside ``torch.no_grad()`` by the caller.

    Args:
        x: (n, C) tensor.
        y: (m, C) tensor.

    Returns:
        Scalar σ² estimate, clamped to at least 1e-6.
    """
    all_pts = torch.cat([x, y], dim=0)              # (n+m, C)
    r = (all_pts * all_pts).sum(dim=1, keepdim=True)  # (n+m, 1)
    sq_dists = r + r.t() - 2.0 * torch.mm(all_pts, all_pts.t())  # (n+m, n+m)
    idx = torch.triu_indices(sq_dists.shape[0], sq_dists.shape[0], offset=1)
    return sq_dists[idx[0], idx[1]].median().clamp(min=1e-6)


class MMDLogitRegularizer(nn.Module):
    """Differentiable MMD regularizer on C-dimensional model logit vectors.

    Computes MMD²(f_member, f_nonmember) between two sets of logit vectors
    using an unbiased U-statistic with a Gaussian RBF kernel, following the
    formula from ``ignite.metrics.MaximumMeanDiscrepancy`` (pytorch-ignite).

    Unlike Ignite's metric class, this module retains the autograd graph so
    that the penalty can be used as a training loss term.

    Args:
        var: Kernel bandwidth σ².  Matches Ignite's ``var`` parameter.
             When ``None`` (default) the median pairwise squared-distance
             heuristic is computed with ``torch.no_grad`` each forward call.
        eps: Guard against zero bandwidth; applied only in the ``None`` case.

    Shape:
        forward(f_member, f_nonmember):
            f_member        – (n, C) logit vectors for member samples.
            f_nonmember     – (m, C) logit vectors for proxy non-member samples.
            output          – scalar MMD² ∈ ℝ  (gradient flows through both).

    Notes:
        - n and m may differ; Ignite requires n == m, this module does not.
        - The penalty is already in squared units (MMD²), consistent with
          Ignite's ``compute()`` output before the final ``.sqrt()``.

    Example::

        reg = MMDLogitRegularizer()
        logits_batch = model(x_train)          # (n, C) — member logits
        logits_proxy = model(proxy_nonmember)  # (m, C) — proxy non-member logits
        task_loss    = criterion(logits_batch, y_train)
        total_loss   = task_loss + lam * reg(logits_batch, logits_proxy)
    """

    def __init__(self, var: float | None = None, eps: float = 1e-6) -> None:
        super().__init__()
        if var is not None:
            self.register_buffer("_fixed_var", torch.tensor(float(var)))
        else:
            self._fixed_var = None
        self.eps = eps

    def _get_var(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self._fixed_var is not None:
            return self._fixed_var
        with torch.no_grad():
            return _median_var(x, y)

    def forward(
        self,
        f_member: torch.Tensor,
        f_nonmember: torch.Tensor,
    ) -> torch.Tensor:
        """Return MMD²(f_member, f_nonmember).

        Args:
            f_member:    (n, C) logit vectors for member (training) samples.
            f_nonmember: (m, C) logit vectors for proxy non-member samples.

        Returns:
            Scalar tensor; gradient flows through both inputs.
        """
        var = self._get_var(f_member, f_nonmember)
        return _mmd2_unbiased(f_member, f_nonmember, var)
