"""Data-fetching layer that wraps MLflow metric/artifact retrieval for visualization."""

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import mlflow
import numpy as np
import torch
from mlflow.tracking import MlflowClient

from privacy_and_grokking.config import TrainConfig


@dataclass
class MetricHistory:
    key: str
    steps: list[int] = field(default_factory=list)
    values: list[float] = field(default_factory=list)

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        idx = np.argsort(self.steps)
        return np.array(self.steps)[idx], np.array(self.values)[idx]


@dataclass
class RunData:
    run_id: str
    run_name: str
    config: TrainConfig
    metrics: dict[str, MetricHistory]
    train_activations: torch.Tensor | None = None
    test_activations: torch.Tensor | None = None
    train_labels: torch.Tensor | None = None
    test_labels: torch.Tensor | None = None
    # All activation snapshots keyed by checkpoint step; populated only when
    # load_run_data is called with load_all_activations=True.
    all_step_activations: dict[int, dict[str, torch.Tensor]] | None = None


TRAINING_METRICS = [
    "validation.train.accuracy",
    "validation.test.accuracy",
    "validation.train.loss",
    "validation.test.loss",
]

EXTRACTION_LOSS_METRICS = [
    "extraction.train.loss.mean",
    "extraction.train.loss.std",
    "extraction.test.loss.mean",
    "extraction.test.loss.std",
    "extraction.loss.overlap",
]

WEIGHT_NORM_KEYS = [
    "weight_norm/total",
]

MIA_AUC_KEYS = [
    "mia_prob/auc",
    "mia_logit/auc",
    "mia_ce_loss/auc",
    "mia_mse_loss/auc",
    "mia_correctness/auc",
    "mia_merlin_morgan_ce/auc",
    "mia_merlin_morgan_mse/auc",
]

MIA_TPR_KEYS = [
    "mia_prob/tpr-at-1-fpr",
    "mia_prob/tpr-at-5-fpr",
    "mia_prob/tpr-at-10-fpr",
    "mia_logit/tpr-at-1-fpr",
    "mia_logit/tpr-at-5-fpr",
    "mia_logit/tpr-at-10-fpr",
    "mia_ce_loss/tpr-at-1-fpr",
    "mia_ce_loss/tpr-at-5-fpr",
    "mia_ce_loss/tpr-at-10-fpr",
]


def _discover_weight_norm_keys(run_id: str) -> list[str]:
    client = MlflowClient()
    run = client.get_run(run_id)
    return [k for k in run.data.metrics if k.startswith("weight_norm/")]


def _discover_gradient_norm_keys(run_id: str) -> list[str]:
    client = MlflowClient()
    run = client.get_run(run_id)
    return [k for k in run.data.metrics if k.startswith("grad_norm/")]


def fetch_metric_history(
    run_id: str,
    keys: list[str],
) -> dict[str, MetricHistory]:
    client = MlflowClient()
    result: dict[str, MetricHistory] = {}
    for key in keys:
        try:
            raw = client.get_metric_history(run_id, key)
        except Exception:
            continue
        if not raw:
            continue
        hist = MetricHistory(key=key)
        for m in raw:
            hist.steps.append(m.step)
            hist.values.append(m.value)
        result[key] = hist
    return result


def fetch_activations(run_id: str, step: int) -> dict[str, torch.Tensor] | None:
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            mlflow.artifacts.download_artifacts(
                artifact_uri=f"runs:/{run_id}/activations/{step}.pt",
                dst_path=tmpdir,
            )
            data = torch.load(
                Path(tmpdir) / f"{step}.pt",
                map_location="cpu",
                weights_only=True,
            )
            return data
    except Exception:
        return None


def list_activation_steps(run_id: str) -> list[int]:
    """Return sorted list of checkpoint steps that have saved activations."""
    client = MlflowClient()
    try:
        artifacts = client.list_artifacts(run_id, path="activations")
        steps = []
        for a in artifacts:
            stem = Path(a.path).stem
            if stem.isdigit():
                steps.append(int(stem))
        return sorted(steps)
    except Exception:
        return []


def fetch_all_step_activations(
    run_id: str,
) -> dict[int, dict[str, torch.Tensor]]:
    """Load every available activation snapshot for *run_id*.

    Returns a mapping of {step: activation_dict}.  Steps with corrupt or
    missing files are silently skipped.
    """
    steps = list_activation_steps(run_id)
    result: dict[int, dict[str, torch.Tensor]] = {}
    for step in steps:
        data = fetch_activations(run_id, step)
        if data is not None:
            result[step] = data
    return result


def load_run_config(run_id: str) -> TrainConfig:
    raw = mlflow.artifacts.load_dict(f"runs:/{run_id}/training_config.json")
    return TrainConfig.model_validate(raw)


def load_run_data(run_id: str, *, load_all_activations: bool = False) -> RunData:
    config = load_run_config(run_id)

    client = MlflowClient()
    run_info = client.get_run(run_id)
    run_name: str = run_info.info.run_name or run_id

    wn_keys = _discover_weight_norm_keys(run_id)
    gn_keys = _discover_gradient_norm_keys(run_id)
    all_keys = (
        TRAINING_METRICS + EXTRACTION_LOSS_METRICS + wn_keys + gn_keys + MIA_AUC_KEYS + MIA_TPR_KEYS
    )
    metrics = fetch_metric_history(run_id, all_keys)
    last_step = max((hist.steps[-1] for hist in metrics.values() if hist.steps), default=0)

    # Single-step activations (last checkpoint)
    act_data = fetch_activations(run_id, last_step)
    train_acts = act_data["train_activations"] if act_data else None
    test_acts = act_data["test_activations"] if act_data else None
    train_labels = act_data["train_labels"] if act_data else None
    test_labels = act_data["test_labels"] if act_data else None

    # All-steps activations (only when requested)
    all_step_activations: dict[int, dict[str, torch.Tensor]] | None = None
    if load_all_activations:
        all_step_activations = fetch_all_step_activations(run_id)

    return RunData(
        run_id=run_id,
        run_name=run_name,
        config=config,
        metrics=metrics,
        train_activations=train_acts,
        test_activations=test_acts,
        train_labels=train_labels,
        test_labels=test_labels,
        all_step_activations=all_step_activations,
    )
