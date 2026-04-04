import os
import random
import tempfile
from collections import defaultdict
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.profiler
from pydantic import BaseModel
from tqdm import tqdm

from privacy_and_grokking.config import (
    TrainConfig,
)
from privacy_and_grokking.datasets import create_masking, generate_datasets, mask_dataset
from privacy_and_grokking.extraction.roc import compute_roc_metrics_single_step
from privacy_and_grokking.metrics import MetricComputer
from privacy_and_grokking.models import create_model
from privacy_and_grokking.utils import (
    Logger,
    eval_mode,
    get_device,
    get_git_changes,
    set_all_seeds,
    setup_mlflow,
)

LOG_FREQUENCY = 500


# State keys that are step counters, not moment/buffer tensors.
_OPTIMIZER_STEP_KEYS = {"step"}


def _log_optimizer_internals(optimizer: torch.optim.Optimizer, step: int) -> None:
    """Log aggregated optimizer state tensors (moments, buffers) as mlflow metrics.

    For each tracked state key (e.g. exp_avg, exp_avg_sq, square_avg,
    momentum_buffer, grad_avg) the L2 norm, mean, and mean-absolute-value
    across *all* parameters are logged under ``optimizer/<key>/{norm,mean,abs_mean}``.
    """
    param_states = optimizer.state_dict()["state"]
    if not param_states:
        return

    key_tensors: dict[str, list[torch.Tensor]] = defaultdict(list)
    for param_state in param_states.values():
        for key, val in param_state.items():
            if key in _OPTIMIZER_STEP_KEYS:
                continue
            if isinstance(val, torch.Tensor) and val.is_floating_point():
                key_tensors[key].append(val.detach().float().flatten())

    metrics: dict[str, float] = {}
    for key, tensors in key_tensors.items():
        cat = torch.cat(tensors)
        metrics[f"optimizer/{key}/norm"] = torch.linalg.norm(cat).item()
        metrics[f"optimizer/{key}/mean"] = cat.mean().item()
        metrics[f"optimizer/{key}/abs_mean"] = cat.abs().mean().item()

    mlflow.log_metrics(metrics, step=step)


# Number of Rademacher probes for the Hutchinson trace estimator.
_CURVATURE_HUTCHINSON_SAMPLES = 5
# Number of power-iteration steps for the top eigenvalue estimate.
_CURVATURE_POWER_ITER = 20


def _hvp(
    loss: torch.Tensor,
    params: list[nn.Parameter],
    v: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Hessian-vector product Hv via double back-propagation."""
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    gv = sum((g * vi).sum() for g, vi in zip(grads, v))
    return list(torch.autograd.grad(gv, params, retain_graph=True))


def _vec_norm(tensors: list[torch.Tensor]) -> torch.Tensor:
    """L2 norm of a list of tensors treated as a single flat vector."""
    return torch.sqrt(sum(t.pow(2).sum() for t in tensors))


def _log_curvature(
    model: nn.Module,
    loss_fn,
    loader: torch.utils.data.DataLoader,
    step: int,
    device: torch.device,
) -> None:
    """Estimate loss-landscape curvature and log to mlflow.

    Metrics logged:
      - ``hessian/trace``  – Hutchinson estimator of tr(H), averaged
        over ``_CURVATURE_HUTCHINSON_SAMPLES`` Rademacher probes.
      - ``hessian/top_eigenvalue`` – power-iteration estimate of λ_max(H)
        using ``_CURVATURE_POWER_ITER`` steps.
      - ``curvature/hessian_trace``, ``curvature/top_eigenvalue`` – legacy names (deprecated).

    Both are computed on a single mini-batch sampled from *loader* with the
    model temporarily set to train mode.
    """
    was_training = model.training
    model.train()
    try:
        x, y = next(iter(loader))
        x, y = x.to(device), y.to(device)
        params = [p for p in model.parameters() if p.requires_grad]

        # Shared forward pass — graph kept for double back-prop.
        logits = model(x)
        loss = loss_fn(logits, y)

        # --- Hutchinson trace: E_v[v^T H v], v ~ Rademacher{±1} ---
        trace_acc = torch.zeros(1, device=device)
        for _ in range(_CURVATURE_HUTCHINSON_SAMPLES):
            v = [torch.randint_like(p.data, 0, 2).float().mul_(2).sub_(1) for p in params]
            hv = _hvp(loss, params, v)
            trace_acc += sum((hvi * vi).sum() for hvi, vi in zip(hv, v))
        hessian_trace = (trace_acc / _CURVATURE_HUTCHINSON_SAMPLES).item()

        # --- Power iteration for top eigenvalue λ_max(H) ---
        v = [torch.randn_like(p.data) for p in params]
        v = [vi / _vec_norm(v) for vi in v]

        top_eigenvalue = 0.0
        for _ in range(_CURVATURE_POWER_ITER):
            hv = _hvp(loss, params, v)
            top_eigenvalue = sum((hvi * vi).sum() for hvi, vi in zip(hv, v)).item()
            hv_norm = _vec_norm(hv)
            if hv_norm < 1e-12:
                break
            v = [hvi / hv_norm for hvi in hv]

        mlflow.log_metrics(
            {
                "hessian/trace": hessian_trace,
                "hessian/top_eigenvalue": top_eigenvalue,
                "curvature/hessian_trace": hessian_trace,
                "curvature/top_eigenvalue": top_eigenvalue,
            },
            step=step,
        )
    except Exception:
        Logger.get().warning("Curvature estimation failed; skipping.", exc_info=True)
    finally:
        model.train(was_training)


def evaluate(
    step: int,
    model: nn.Module,
    loss_fn,
    eval_train_loader,
    eval_test_loader,
    compute_mm: bool = False,
) -> tuple[float, float, float, float]:
    """Compute and log all metrics for current model state.

    Returns: (train_loss_mean, test_loss_mean, train_accuracy, test_accuracy)
    """
    with eval_mode(model):
        device = get_device()

        # Compute basic loss and accuracy metrics
        basic_metrics, train_losses, test_losses = MetricComputer.compute_basic_metrics(
            model, loss_fn, eval_train_loader, eval_test_loader, device
        )

        # Compute weight and gradient norms
        weight_norms = MetricComputer.compute_weight_norms(model)
        gradient_norms = MetricComputer.compute_gradient_norms(model)

        # Compute attack signals (with Merlin-Morgan optionally)
        train_signals = MetricComputer.compute_attack_signals(
            model, loss_fn, eval_train_loader, device, compute_mm=compute_mm
        )
        test_signals = MetricComputer.compute_attack_signals(
            model, loss_fn, eval_test_loader, device, compute_mm=compute_mm
        )
        attack_metrics = MetricComputer.compute_attack_auc_metrics(
            train_signals, test_signals, include_mm=compute_mm
        )

        # Combine all metrics with consistent naming
        all_metrics = {}
        # Basic metrics already have keys like "train/loss_mean" - add validation prefix
        for key, value in basic_metrics.items():
            all_metrics[f"validation.{key}"] = value
        # Norms keep their original names (weight_norm/..., grad_norm/...)
        all_metrics.update(weight_norms)
        all_metrics.update(gradient_norms)
        # Attack metrics already have "attack/" prefix
        all_metrics.update(attack_metrics)

        mlflow.log_metrics(all_metrics, step=step)

        return (
            basic_metrics["train/loss_mean"],
            basic_metrics["test/loss_mean"],
            basic_metrics["train/accuracy"],
            basic_metrics["test/accuracy"],
        )


def save_model(model: nn.Module, optimizer: torch.optim.Optimizer, step: int) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Save optimizer state
        checkpoint_dir = f"checkpoints/{step}"
        model_path = tmpdir_path / "model.pth"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(str(model_path), checkpoint_dir)

        optimizer_path = tmpdir_path / "optimizer.pth"
        torch.save(optimizer.state_dict(), optimizer_path)
        mlflow.log_artifact(str(optimizer_path), checkpoint_dir)

        rng_state_path = tmpdir_path / "rng_state.pth"
        states = {
            "random": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch-cuda": torch.cuda.get_rng_state_all(),
        }
        torch.save(states, rng_state_path)
        mlflow.log_artifact(str(rng_state_path), checkpoint_dir)


def load_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    run_id: str,
    step: int,
    device: torch.device,
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        checkpoint_dir = f"runs:/{run_id}/checkpoints/{step}/"

        model_path = str(tmpdir_path / "model.pth")
        mlflow.artifacts.download_artifacts(
            artifact_uri=f"{checkpoint_dir}/model.pth",
            dst_path=str(tmpdir_path),
        )
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

        optimizer_path = str(tmpdir_path / "optimizer.pth")
        mlflow.artifacts.download_artifacts(
            artifact_uri=f"{checkpoint_dir}/optimizer.pth",
            dst_path=str(tmpdir_path),
        )
        optimizer.load_state_dict(
            torch.load(optimizer_path, map_location=device, weights_only=True)
        )

        mlflow.artifacts.download_artifacts(
            artifact_uri=f"{checkpoint_dir}/rng_state.pth",
            dst_path=str(tmpdir_path),
        )
        rng_dst = str(tmpdir_path / "rng_state.pth")
        states = torch.load(rng_dst, weights_only=False)
        random.setstate(states["random"])
        np.random.set_state(states["numpy"])
        torch.set_rng_state(states["torch"])
        if torch.cuda.is_available() and states["torch-cuda"]:
            torch.cuda.set_rng_state_all(states["torch-cuda"])


class RestartConfig(BaseModel):
    run_id: str
    checkpoint: int


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train_handle(
    cfg: TrainConfig | RestartConfig,
    optimization_steps: int,
    checkpoint_frequency: int = LOG_FREQUENCY,
) -> None:
    logger = Logger.get()
    if isinstance(cfg, RestartConfig):
        logger.info("Restarting training", checkpoint=cfg.checkpoint)
        config = TrainConfig.model_validate(
            mlflow.artifacts.load_dict(f"runs:/{cfg.run_id}/training_config.json")
        )
        restart_index = int(mlflow.get_run(cfg.run_id).data.tags.get("restart_count", "0"))
        restart_index += 1
        restart = True
    else:
        logger.info("Starting new training run")
        mlflow.log_dict(cfg.model_dump(), "training_config.json")
        config = cfg
        restart_index = 0
        restart = False
    mlflow.set_tag("restart_count", str(restart_index))
    mlflow.log_dict(get_git_changes(), f"git/restart_{restart_index}.json")

    device_name = get_device()
    device = torch.device(device_name)
    logger.info(f"Using device {device_name}", device=device_name)

    num_workers = min(4, os.cpu_count() or 1) if torch.cuda.is_available() else 0
    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0

    logger.info("Preparing dataset.")
    train, test = generate_datasets(config=config.dataset)
    masking = create_masking(
        config=config.dataset_mask,
        num_samples=len(train),
        num_classes=train.num_classes,
    )
    train_subset = mask_dataset(masking, train, config.dataset_mask_idx)

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        worker_init_fn=seed_worker if num_workers > 0 else None,
    )
    eval_train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        worker_init_fn=seed_worker if num_workers > 0 else None,
    )
    eval_test_loader = torch.utils.data.DataLoader(
        test,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        worker_init_fn=seed_worker if num_workers > 0 else None,
    )
    batch_offset = cfg.checkpoint % len(train_loader) if restart else 0

    logger.info("Preparing model.")
    model = create_model(
        name=config.model,
        input_dim=train.input_shape,
        num_classes=train.num_classes,
        initialization_scale=config.initialization_scale,
    )
    model.to(device)

    logger.info("Preparing optimizer and loss function.")
    loss_fn = config.loss(num_classes=train.num_classes)
    loss_fn_eval = config.loss(num_classes=train.num_classes, reduction="none")
    optimizer = config.optimizer(params=model.parameters())

    logger.info("Preparing seeds and defaults.")
    torch.set_default_dtype(torch.float32)
    set_all_seeds(config.seed)

    if restart:
        load_model(model, optimizer, cfg.run_id, cfg.checkpoint, device)

    scheduler = config.scheduler(
        optimizer=optimizer,
        optimization_steps=optimization_steps,
        checkpoint=cfg.checkpoint if restart else -1,
    )

    logger.info("Starting training loop.")
    enable_profiler = os.environ.get("PAG_PROFILE", "").lower() in ("1", "true", "yes")
    step = cfg.checkpoint if restart else 0
    with tqdm(total=optimization_steps) as pbar:
        pbar.update(step)
        prof = None
        if enable_profiler:
            prof = torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=100, warmup=50, active=50, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler"),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            )
            prof.start()
        eval_count = 0
        while step < optimization_steps:
            for x, y in train_loader:
                # Skip batches we've already processed in this epoch
                if restart and batch_offset > 0:
                    batch_offset -= 1
                    continue

                if step >= optimization_steps:
                    break

                # Validation frequency conditions
                if (
                    (step < 50)
                    or (step < LOG_FREQUENCY and step % 50 == 0)
                    or (step % LOG_FREQUENCY == 0)
                ):
                    eval_count += 1
                    heavy_metrics = eval_count % 2 == 0
                    train_loss_mean, test_loss_mean, train_accuracy, test_accuracy = evaluate(
                        step,
                        model,
                        loss_fn_eval,
                        eval_train_loader,
                        eval_test_loader,
                        compute_mm=heavy_metrics,
                    )
                    _log_optimizer_internals(optimizer, step)
                    if heavy_metrics:
                        _log_curvature(model, loss_fn, eval_train_loader, step, device)
                    pbar.set_description(
                        f"L: {train_loss_mean:1.1e}|{test_loss_mean:1.1e}. A: {train_accuracy * 100:2.1f}%|{test_accuracy * 100:2.1f}%"
                    )

                # Model checkpoint frequency (separate from validation)
                if step % checkpoint_frequency == 0:
                    save_model(model, optimizer, step)

                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = loss_fn(logits, y)
                loss.backward()

                optimizer.step()
                scheduler.step()

                step += 1
                if prof is not None:
                    prof.step()
                pbar.update(1)
        if prof is not None:
            prof.stop()
            if os.path.exists("./profiler"):
                mlflow.log_artifacts("./profiler", artifact_path="profiler")
    logger.info("Saving results.")
    x, _ = next(iter(train_loader))
    evaluate(step, model, loss_fn_eval, eval_train_loader, eval_test_loader, compute_mm=True)
    _log_optimizer_internals(optimizer, step)
    _log_curvature(model, loss_fn, eval_train_loader, step, device)
    save_model(model, optimizer, step)

    logger.info(f"Ending training: '{config.name}'")


def train(
    exp_name: str,
    total_steps: int,
    cfg: TrainConfig | RestartConfig,
    run_name: str | None = None,
    checkpoint_frequency: int | None = None,
) -> str:
    run_name = run_name or (cfg.full_name if isinstance(cfg, TrainConfig) else cfg.run_id)
    if checkpoint_frequency is None:
        checkpoint_frequency = LOG_FREQUENCY

    setup_mlflow(exp_name)

    log_handler = Logger()
    run_id = cfg.run_id if isinstance(cfg, RestartConfig) else None
    with (
        log_handler as logger,
        mlflow.start_run(run_id=run_id, run_name=run_name, log_system_metrics=True) as mlflow_run,
    ):
        returned_run_id = mlflow_run.info.run_id
        try:
            logger.bind(run_id=returned_run_id)
            train_handle(cfg, total_steps, checkpoint_frequency)
        except Exception as e:
            logger.error(f"Training failed with error: {e}", exc_info=True)
        finally:
            if log_handler.log_file_path and mlflow.active_run() is not None:
                mlflow.log_artifact(str(log_handler.log_file_path), "training_logs")
            logger.info("Training done")
    return returned_run_id
