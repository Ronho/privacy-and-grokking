from privacy_and_grokking.metrics.config import MetricsConfig
from privacy_and_grokking.metrics.evaluate import evaluate


def extraction_handler(exp_name: str, run_id: str) -> None:
    """Re-evaluate all saved checkpoints for a given run and log metrics.

    Lazily imports the full extraction machinery to avoid pulling in the
    heavy config/dataset dependency chain at module load time.
    """
    from privacy_and_grokking.metrics.extraction import (
        extraction_handler as _handler,
    )

    _handler(exp_name, run_id)


__all__ = ["MetricsConfig", "evaluate", "extraction_handler"]
