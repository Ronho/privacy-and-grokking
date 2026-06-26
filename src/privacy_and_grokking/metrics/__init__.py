from privacy_and_grokking.metrics.config import MetricsConfig
from privacy_and_grokking.metrics.evaluate import evaluate
from privacy_and_grokking.metrics.neural_collapse import (
    NeuralCollapseMetrics,
    compute_all_nc_metrics,
    compute_nc0,
    compute_nc1,
    compute_nc2,
    compute_nc3,
    compute_nc4,
    compute_rnc1,
)
from privacy_and_grokking.metrics.nhsic import (
    compute_hsic,
    compute_nhsic,
    nhsic_features_vs_inputs,
    nhsic_features_vs_labels,
)


def extraction_handler(exp_name: str, run_id: str) -> None:
    """Re-evaluate all saved checkpoints for a given run and log metrics.

    Lazily imports the full extraction machinery to avoid pulling in the
    heavy config/dataset dependency chain at module load time.
    """
    from privacy_and_grokking.metrics.extraction import (
        extraction_handler as _handler,
    )

    _handler(exp_name, run_id)


__all__ = [
    "MetricsConfig",
    "NeuralCollapseMetrics",
    "compute_all_nc_metrics",
    "compute_hsic",
    "compute_nc0",
    "compute_nc1",
    "compute_nc2",
    "compute_nc3",
    "compute_nc4",
    "compute_nhsic",
    "compute_rnc1",
    "evaluate",
    "extraction_handler",
    "nhsic_features_vs_inputs",
    "nhsic_features_vs_labels",
]
