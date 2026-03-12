import tempfile
from pathlib import Path

import mlflow
import torch
from mlflow.tracking import MlflowClient
from torch.utils.data import Subset
from tqdm import tqdm

from privacy_and_grokking.config import TrainConfig
from privacy_and_grokking.datasets import (
    create_masking,
    generate_datasets,
    mask_dataset,
)
from privacy_and_grokking.extraction.activations import (
    extract_all_layer_activations,
    extract_penultimate_activations,
)
from privacy_and_grokking.extraction.mia_merlin_morgan import (
    compute_merlin_morgan_signals,
)
from privacy_and_grokking.extraction.mia_simple import (
    compute_mia_signals,
)
from privacy_and_grokking.extraction.roc import (
    compute_roc_metrics_single_step,
)
from privacy_and_grokking.models import create_model
from privacy_and_grokking.utils import Logger, get_device, setup_mlflow


def _list_checkpoint_steps(run_id: str) -> list[int]:
    """Discover all checkpoint steps available for a run."""
    client = MlflowClient()
    artifacts = client.list_artifacts(run_id, path="checkpoints")
    steps = []
    for artifact in artifacts:
        name = artifact.path.split("/")[-1]
        if name.isdigit():
            steps.append(int(name))
    return sorted(steps)


def _compute_loss_distribution_overlap(
    train_losses: torch.Tensor,
    test_losses: torch.Tensor,
    n_bins: int = 100,
) -> float:
    """Compute the histogram-intersection overlap of two loss distributions.

    Returns a value in ``[0, 1]`` where ``1.0`` means identical distributions
    and ``0.0`` means completely disjoint.  Both tensors are expected to be
    1-D and values must be finite; any NaN/Inf are silently dropped.
    """
    train_losses = train_losses.flatten().float()
    test_losses = test_losses.flatten().float()
    train_losses = train_losses[torch.isfinite(train_losses)]
    test_losses = test_losses[torch.isfinite(test_losses)]
    if train_losses.numel() == 0 or test_losses.numel() == 0:
        return 0.0

    all_losses = torch.cat([train_losses, test_losses])
    lo = float(all_losses.min().item())
    hi = float(all_losses.max().item())
    if hi <= lo:
        return 1.0

    train_hist = torch.histc(train_losses, bins=n_bins, min=lo, max=hi)
    test_hist = torch.histc(test_losses, bins=n_bins, min=lo, max=hi)

    # Normalise to probability mass functions
    train_hist = train_hist / train_hist.sum().clamp(min=1e-12)
    test_hist = test_hist / test_hist.sum().clamp(min=1e-12)

    return float(torch.minimum(train_hist, test_hist).sum().item())


def _compute_weight_norms(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Compute L2 (Frobenius) norm per named parameter and total."""
    norms: dict[str, float] = {}
    all_params = []

    for name, param in state_dict.items():
        norms[f"weight_norm/{name}"] = torch.linalg.norm(param.float()).item()
        all_params.append(param.float().flatten())

    if all_params:
        norms["weight_norm/total"] = torch.linalg.norm(torch.cat(all_params)).item()
    else:
        norms["weight_norm/total"] = 0.0

    return norms


def _step_wise(run_id: str, *, save_all_activations: bool = False) -> None:
    logger = Logger.get()
    device = get_device()

    steps = _list_checkpoint_steps(run_id)
    if not steps:
        logger.warning("No checkpoints found for run.", run_id=run_id)
        return

    # Dataset
    cfg = TrainConfig.model_validate(
        mlflow.artifacts.load_dict(f"runs:/{run_id}/training_config.json")
    )
    train_ds, test_ds = generate_datasets(cfg.dataset)
    num_classes = train_ds.num_classes
    masking = create_masking(
        config=cfg.dataset_mask,
        num_samples=len(train_ds),
        num_classes=num_classes,
    )
    train_subset = mask_dataset(
        masking,
        train_ds,
        cfg.dataset_mask_idx,
    )

    # Subsamples for Merlin Morgan
    subsample_size = min(len(train_subset), len(test_ds))
    mm_train = Subset(
        train_subset,
        list(range(subsample_size)),
    )
    mm_test = Subset(
        test_ds,
        list(range(subsample_size)),
    )

    for step in tqdm(steps, desc="Extracting Data", unit="ckpt"):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_uri = f"runs:/{run_id}/checkpoints/{step}/model.pth"
            mlflow.artifacts.download_artifacts(
                artifact_uri=artifact_uri,
                dst_path=tmpdir,
            )
            model_path = Path(tmpdir) / "model.pth"
            state_dict = torch.load(
                model_path,
                map_location=device,
                weights_only=True,
            )

        # Weight Norms
        norms = _compute_weight_norms(state_dict)
        mlflow.log_metrics(norms, step=step)

        # Build model
        model = create_model(
            name=cfg.model,
            input_dim=train_ds.input_shape,
            num_classes=num_classes,
            initialization_scale=cfg.initialization_scale,
        )
        model.to(device)
        model.load_state_dict(state_dict)
        model.eval()

        # Simple MIA signals
        t_pr, t_lo, t_ce, t_mse, t_cor = compute_mia_signals(
            model,
            train_subset,
        )
        e_pr, e_lo, e_ce, e_mse, e_cor = compute_mia_signals(
            model,
            test_ds,
        )

        # Merlin/Morgan signals
        t_ce_v, t_mse_v = compute_merlin_morgan_signals(
            model,
            mm_train,
            num_classes,
        )
        e_ce_v, e_mse_v = compute_merlin_morgan_signals(
            model,
            mm_test,
            num_classes,
        )

        # Compute & log ROC metrics
        # Each entry: (metric_prefix, train_signal, test_signal)
        # Loss-based signals are negated so that higher = more likely member.
        attacks: list[tuple[str, torch.Tensor, torch.Tensor]] = [
            ("mia_prob", t_pr.squeeze(), e_pr.squeeze()),
            ("mia_logit", t_lo.squeeze(), e_lo.squeeze()),
            ("mia_ce_loss", -t_ce.squeeze(), -e_ce.squeeze()),
            ("mia_mse_loss", -t_mse.squeeze(), -e_mse.squeeze()),
            ("mia_correctness", t_cor.squeeze(), e_cor.squeeze()),
            ("mia_merlin_morgan_ce", t_ce_v, e_ce_v),
            ("mia_merlin_morgan_mse", t_mse_v, e_mse_v),
        ]

        roc_metrics: dict[str, float] = {}
        for prefix, tr_sig, te_sig in attacks:
            m = compute_roc_metrics_single_step(tr_sig, te_sig)
            for key, value in m.items():
                roc_metrics[f"{prefix}/{key}"] = value

        mlflow.log_metrics(roc_metrics, step=step)

        # Loss distribution statistics and overlap (CE loss)
        t_ce_flat = t_ce.squeeze().float()
        e_ce_flat = e_ce.squeeze().float()
        loss_dist_metrics: dict[str, float] = {
            "extraction.train.loss.mean": float(t_ce_flat.mean().item()),
            "extraction.train.loss.std": float(t_ce_flat.std().item()),
            "extraction.test.loss.mean": float(e_ce_flat.mean().item()),
            "extraction.test.loss.std": float(e_ce_flat.std().item()),
            "extraction.loss.overlap": _compute_loss_distribution_overlap(t_ce_flat, e_ce_flat),
        }
        mlflow.log_metrics(loss_dist_metrics, step=step)

        # Activations for penultimate layer and all layers
        if save_all_activations or step == steps[-1]:
            train_acts, train_labels = extract_penultimate_activations(model, train_subset)
            test_acts, test_labels = extract_penultimate_activations(model, test_ds)
            train_layer_acts, _ = extract_all_layer_activations(model, train_subset)
            test_layer_acts, _ = extract_all_layer_activations(model, test_ds)
            payload = {
                "train_activations": train_acts,
                "test_activations": test_acts,
                "train_labels": train_labels,
                "test_labels": test_labels,
                "train_layer_activations": train_layer_acts,
                "test_layer_activations": test_layer_acts,
                "step": step,
            }
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / f"{step}.pt"
                torch.save(payload, path)
                mlflow.log_artifact(str(path), artifact_path="activations")


def extraction_handler(exp_name: str, run_id: str, *, save_all_activations: bool = False) -> None:
    setup_mlflow(exp_name)
    with (
        Logger() as logger,
        mlflow.start_run(run_id=run_id),
    ):
        logger.info(
            "Starting data extraction for run.",
            run_id=run_id,
            save_all_activations=save_all_activations,
        )
        _step_wise(run_id, save_all_activations=save_all_activations)
        logger.info(
            "Completed data extraction for run.",
            run_id=run_id,
        )
