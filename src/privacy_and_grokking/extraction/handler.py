import os
import random
import tempfile
from collections import defaultdict
from pathlib import Path

import mlflow
import numpy as np
import torch
from mlflow.tracking import MlflowClient
from torch import nn
from torch.utils.data import Subset
from tqdm import tqdm

from privacy_and_grokking.config import TrainConfig
from privacy_and_grokking.datasets import (
    create_masking,
    generate_datasets,
    mask_dataset,
)
from privacy_and_grokking.extraction.distribution_overlap import compute_distribution_overlap
from privacy_and_grokking.models import create_model
from privacy_and_grokking.utils import Logger, get_device, setup_mlflow

MERLIN_MORGAN_NOISY_SAMPLES = 100
MERLIN_MORGAN_NOISE_SCALE = 0.01


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _list_checkpoint_steps(run_id: str) -> list[int]:
    client = MlflowClient()
    artifacts = client.list_artifacts(run_id, path="checkpoints")
    steps = []
    for artifact in artifacts:
        parts = artifact.path.split("/")
        # Expected patterns: "checkpoints/123" or "checkpoints/123/model.pth"
        if len(parts) >= 2 and parts[0] == "checkpoints":
            candidate = parts[1]
            if candidate.isdigit():
                steps.append(int(candidate))
    return sorted(set(steps))


def _stratified_indices(dataset, n: int) -> list[int]:
    class_indices: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        class_indices[int(label)].append(idx)

    num_classes = len(class_indices)
    per_class = n // num_classes

    generator = torch.Generator().manual_seed(4711)
    selected: list[int] = []
    # Note: this may select fewer than n samples if some classes have fewer than per_class samples.
    for indices in class_indices.values():
        t = torch.tensor(indices)
        perm = torch.randperm(len(t), generator=generator)
        chosen = t[perm[: min(per_class, len(t))]]
        selected.extend(chosen.tolist())

    return selected


def _get_datasets(cfg: TrainConfig):
    train, test = generate_datasets(cfg.dataset)
    masking = create_masking(
        config=cfg.dataset_mask,
        num_samples=len(train),
        num_classes=train.num_classes,
    )
    train_subset = mask_dataset(
        masking,
        train,
        cfg.dataset_mask_idx,
    )
    subsample_size = min(len(train_subset), len(test))  # type: ignore
    train_sub = Subset(
        train_subset,
        _stratified_indices(train_subset, subsample_size),
    )
    test_sub = Subset(
        test,
        _stratified_indices(test, subsample_size),
    )
    num_workers = min(4, os.cpu_count() or 1) if torch.cuda.is_available() else 0
    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0

    train_loader = torch.utils.data.DataLoader(
        train_sub,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        worker_init_fn=seed_worker if num_workers > 0 else None,
    )
    test_loader = torch.utils.data.DataLoader(
        test_sub,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        worker_init_fn=seed_worker if num_workers > 0 else None,
    )

    return train_loader, test_loader, train.input_shape, train.num_classes


def _iterate_dataloader(dataloader, device, model, last_step: bool):
    from privacy_and_grokking.metrics import MetricComputer

    buffers_accum: dict[str, list[torch.Tensor]] = {}
    label_list_accum: list[torch.Tensor] = []
    handles: list = []
    if last_step:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                key = name
                buffers_accum[key] = []

                def _make_hook(k: str):
                    def _hook(_module: nn.Module, _inp: tuple, output: torch.Tensor) -> None:
                        buffers_accum[k].append(output.detach().cpu())

                    return _hook

                handles.append(module.register_forward_hook(_make_hook(key)))

    try:
        prob_list = []
        logit_list = []
        ce_list = []
        mse_list = []
        correctness_list = []
        mm_ce_list = []
        mm_mse_list = []

        with torch.no_grad():
            for x, y in dataloader:
                if last_step:
                    label_list_accum.append(y)
                batch_result = MetricComputer._process_batch(model, x, y, device, compute_mm=True)
                prob_list.append(batch_result["prob"])
                logit_list.append(batch_result["logit"])
                ce_list.append(batch_result["ce_loss"])
                mse_list.append(batch_result["mse_loss"])
                correctness_list.append(batch_result["correctness"])
                mm_ce_list.append(batch_result["mm_ce"])
                mm_mse_list.append(batch_result["mm_mse"])
    finally:
        if last_step:
            for h in handles:
                h.remove()
            # Convert list of tensors to single tensor per layer
            buffers = {k: torch.cat(v, dim=0) for k, v in buffers_accum.items()}
            label_list = torch.cat(label_list_accum, dim=0)
        else:
            buffers = {}
            label_list = torch.tensor([])

    cat_correct_probs = torch.cat(prob_list, dim=0).squeeze()
    cat_correct_logits = torch.cat(logit_list, dim=0).squeeze()
    cat_ce_losses = torch.cat(ce_list, dim=0).squeeze()
    cat_mse_losses = torch.cat(mse_list, dim=0).squeeze()
    cat_correctness_list = torch.cat(correctness_list, dim=0).squeeze()
    cat_mm_ce_votes = torch.cat(mm_ce_list, dim=0).squeeze()
    cat_mm_mse_votes = torch.cat(mm_mse_list, dim=0).squeeze()

    return (
        cat_correct_probs,
        cat_correct_logits,
        cat_ce_losses,
        cat_mse_losses,
        cat_correctness_list,
        cat_mm_ce_votes,
        cat_mm_mse_votes,
        buffers,
        label_list,
    )


def _extract_weight_norm(model, step: int):
    from privacy_and_grokking.metrics import MetricComputer

    norms = MetricComputer.compute_weight_norms(model)
    mlflow.log_metrics(norms, step=step)


def _step_wise(run_id: str) -> None:
    from privacy_and_grokking.metrics import MetricComputer

    logger = Logger.get()
    device = get_device()

    steps = _list_checkpoint_steps(run_id)
    if not steps:
        logger.warning("No checkpoints found for run.", run_id=run_id)
        return

    cfg = TrainConfig.model_validate(
        mlflow.artifacts.load_dict(f"runs:/{run_id}/training_config.json")
    )

    train, test, input_shape, num_classes = _get_datasets(cfg)

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
        model = create_model(
            name=cfg.model,
            input_dim=input_shape,
            num_classes=num_classes,
            initialization_scale=None,
        )
        model.to(device)
        model.load_state_dict(state_dict)
        model.eval()

        _extract_weight_norm(model, step)

        last_step = step == steps[-1]
        tr_cp, tr_cl, tr_ce, tr_mse, tr_corr, tr_mm_ce, tr_mm_mse, tr_acts, tr_labels = (
            _iterate_dataloader(train, device, model, last_step)
        )
        te_cp, te_cl, te_ce, te_mse, te_corr, te_mm_ce, te_mm_mse, te_acts, te_labels = (
            _iterate_dataloader(test, device, model, last_step)
        )

        train_signals = {
            "prob": tr_cp,
            "logit": tr_cl,
            "ce_loss": tr_ce,
            "mse_loss": tr_mse,
            "correctness": tr_corr,
            "mm_ce": tr_mm_ce,
            "mm_mse": tr_mm_mse,
        }
        test_signals = {
            "prob": te_cp,
            "logit": te_cl,
            "ce_loss": te_ce,
            "mse_loss": te_mse,
            "correctness": te_corr,
            "mm_ce": te_mm_ce,
            "mm_mse": te_mm_mse,
        }

        roc_metrics = MetricComputer.compute_attack_auc_metrics(
            train_signals, test_signals, include_mm=True
        )

        signal_to_prefix = {
            "prob": "mia_prob",
            "logit": "mia_logit",
            "ce_loss": "mia_ce_loss",
            "mse_loss": "mia_mse_loss",
            "correctness": "mia_correctness",
            "mm_ce": "mia_merlin_morgan_ce",
            "mm_mse": "mia_merlin_morgan_mse",
        }
        renamed_metrics = {}
        for key, value in roc_metrics.items():
            # key format: attack/{signal}/{metric}
            parts = key.split("/")
            if len(parts) == 3 and parts[0] == "attack":
                signal = parts[1]
                metric = parts[2]
                if signal in signal_to_prefix:
                    new_prefix = signal_to_prefix[signal]
                    renamed_metrics[f"{new_prefix}/{metric}"] = value
                else:
                    renamed_metrics[key] = value
            else:
                renamed_metrics[key] = value
        mlflow.log_metrics(renamed_metrics, step=step)

        tr_ce_flat = tr_ce.squeeze().float()
        te_ce_flat = te_ce.squeeze().float()
        loss_dist_metrics: dict[str, float] = {
            "extraction.train.loss.mean": float(tr_ce_flat.mean().item()),
            "extraction.train.loss.std": float(tr_ce_flat.std().item()),
            "extraction.test.loss.mean": float(te_ce_flat.mean().item()),
            "extraction.test.loss.std": float(te_ce_flat.std().item()),
            "extraction.loss.overlap": compute_distribution_overlap(tr_ce_flat, te_ce_flat),
        }
        mlflow.log_metrics(loss_dist_metrics, step=step)

        if last_step:
            payload = {
                "train_activations": tr_acts,
                "test_activations": te_acts,
                "train_labels": tr_labels,
                "test_labels": te_labels,
                "step": step,
            }
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / f"{step}.pt"
                torch.save(payload, path)
                mlflow.log_artifact(str(path), artifact_path="activations")


def extraction_handler(exp_name: str, run_id: str) -> None:
    setup_mlflow(exp_name)
    with (
        Logger() as logger,
        mlflow.start_run(run_id=run_id),
    ):
        logger.info("Starting data extraction for run.", run_id=run_id)
        _step_wise(run_id)
        logger.info(
            "Completed data extraction for run.",
            run_id=run_id,
        )
