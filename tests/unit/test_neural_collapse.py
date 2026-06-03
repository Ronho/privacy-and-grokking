import torch

from privacy_and_grokking.metrics.neural_collapse import (
    compute_all_nc_metrics,
    compute_nc0,
)


def test_compute_nc0_is_zero_when_all_row_sums_are_zero():
    w = torch.tensor(
        [
            [1.0, -1.0, 0.0],
            [2.0, -3.0, 1.0],
            [-0.5, 0.5, 0.0],
        ]
    )
    assert compute_nc0(w) == 0.0


def test_compute_nc0_is_positive_when_row_sums_not_zero():
    w = torch.tensor(
        [
            [1.0, 2.0, -1.0],
            [0.5, 0.5, 0.5],
        ]
    )
    # Row sums are [2.0, 1.5], so mean absolute row sum is 1.75
    assert torch.isclose(torch.tensor(compute_nc0(w)), torch.tensor(1.75))


def test_compute_all_nc_metrics_populates_nc0():
    features = torch.tensor(
        [
            [1.0, 0.0],
            [1.2, 0.1],
            [-1.0, 0.0],
            [-1.1, -0.1],
        ]
    )
    labels = torch.tensor([0, 0, 1, 1])
    w = torch.tensor(
        [
            [1.0, -1.0],
            [2.0, -2.0],
        ]
    )

    nc = compute_all_nc_metrics(features, labels, classifier_weight=w, classifier_bias=None)

    assert hasattr(nc, "nc0")
    assert nc.nc0 == 0.0
