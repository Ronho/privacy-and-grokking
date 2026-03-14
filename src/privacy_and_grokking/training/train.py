import random
import tempfile
from collections import defaultdict
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
from pydantic import BaseModel
from tqdm import tqdm

from privacy_and_grokking.config import (
    TrainConfig,
)
from privacy_and_grokking.datasets import create_masking, generate_datasets, mask_dataset
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


def _compute_gradient_norms(model: nn.Module) -> dict[str, float]:
    """Compute per-parameter and total L2 gradient norms."""
    norms: dict[str, float] = {}
    all_grads: list[torch.Tensor] = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            g = param.grad.detach().float().flatten()
            norms[f"grad_norm/{name}"] = torch.linalg.norm(g).item()
            all_grads.append(g)
    norms["grad_norm/total"] = torch.linalg.norm(torch.cat(all_grads)).item() if all_grads else 0.0
    return norms


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


def _eval(model: nn.Module, loss_fn, loader) -> tuple[torch.Tensor, float]:
    device = get_device()
    all_losses = []
    correct = 0
    number = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        losses = loss_fn(logits, y)
        all_losses.append(losses.detach().cpu())

        labels = torch.argmax(logits, dim=1)
        correct += torch.sum(labels == y).item()
        number += x.size(0)
    all_losses_tensor = torch.cat(all_losses)
    return all_losses_tensor, correct / number


def evaluate(
    step: int,
    model: nn.Module,
    loss_fn,
    eval_train_loader,
    eval_test_loader,
) -> tuple[float, float, float, float]:
    with eval_mode(model):
        train_losses, train_accuracy = _eval(model, loss_fn, eval_train_loader)
        test_losses, test_accuracy = _eval(model, loss_fn, eval_test_loader)

        train_loss_mean = train_losses.mean().item()
        train_loss_std = train_losses.std().item()
        test_loss_mean = test_losses.mean().item()
        test_loss_std = test_losses.std().item()

        gradient_norms = _compute_gradient_norms(model)

        mlflow.log_metrics(
            {
                "validation.train.loss": train_loss_mean,
                "validation.train.loss_std": train_loss_std,
                "validation.train.accuracy": train_accuracy,
                "validation.test.loss": test_loss_mean,
                "validation.test.loss_std": test_loss_std,
                "validation.test.accuracy": test_accuracy,
                **gradient_norms,
            },
            step=step,
        )

        return train_loss_mean, test_loss_mean, train_accuracy, test_accuracy


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


def train_handle(cfg: TrainConfig | RestartConfig, optimization_steps: int) -> None:
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
    )
    eval_train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=config.batch_size, shuffle=False
    )
    eval_test_loader = torch.utils.data.DataLoader(
        test, batch_size=config.batch_size, shuffle=False
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
    step = cfg.checkpoint if restart else 0
    with tqdm(total=optimization_steps) as pbar:
        pbar.update(step)
        while step < optimization_steps:
            for x, y in train_loader:
                # Skip batches we've already processed in this epoch
                if restart and batch_offset > 0:
                    batch_offset -= 1
                    continue

                if step >= optimization_steps:
                    break

                if (
                    (step < 50)
                    or (step < LOG_FREQUENCY and step % 50 == 0)
                    or (step % LOG_FREQUENCY == 0)
                ):
                    train_loss_mean, test_loss_mean, train_accuracy, test_accuracy = evaluate(
                        step, model, loss_fn_eval, eval_train_loader, eval_test_loader
                    )
                    _log_optimizer_internals(optimizer, step)
                    save_model(model, optimizer, step)
                    pbar.set_description(
                        f"L: {train_loss_mean:1.1e}|{test_loss_mean:1.1e}. A: {train_accuracy * 100:2.1f}%|{test_accuracy * 100:2.1f}%"
                    )

                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = loss_fn(logits, y)
                loss.backward()

                optimizer.step()
                scheduler.step()

                step += 1
                pbar.update(1)

    logger.info("Saving results.")
    x, _ = next(iter(train_loader))
    evaluate(step, model, loss_fn_eval, eval_train_loader, eval_test_loader)
    save_model(model, optimizer, step)

    logger.info(f"Ending training: '{config.name}'")


def train(
    exp_name: str, total_steps: int, cfg: TrainConfig | RestartConfig, run_name: str | None = None
) -> str:
    run_name = run_name or (cfg.full_name if isinstance(cfg, TrainConfig) else cfg.run_id)

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
            train_handle(cfg, total_steps)
        except Exception as e:
            logger.error(f"Training failed with error: {e}", exc_info=True)
        finally:
            if log_handler.log_file_path and mlflow.active_run() is not None:
                mlflow.log_artifact(str(log_handler.log_file_path), "training_logs")
            logger.info("Training done")
    return returned_run_id
