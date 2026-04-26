import itertools
import os
import random
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.profiler
from pydantic import BaseModel
from tqdm import tqdm

from privacy_and_grokking.config import (
    TrainConfig,
)
from privacy_and_grokking.datasets import (
    GpuDataset,
    create_masking,
    generate_datasets,
    mask_dataset,
)
from privacy_and_grokking.metrics import evaluate
from privacy_and_grokking.models import create_model
from privacy_and_grokking.utils import (
    Logger,
    get_device,
    get_git_changes,
    set_all_seeds,
    setup_mlflow,
)

LOG_FREQUENCY = 1000
HEAVY_METRICS_LOG_FREQUENCY = LOG_FREQUENCY * 10


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

    logger.info("Preparing dataset.")
    keep_on_gpu = torch.cuda.is_available()
    train, test = generate_datasets(config=config.dataset)
    masking = create_masking(
        config=config.dataset_mask,
        num_samples=len(train),
        num_classes=train.num_classes,
    )
    train_subset = mask_dataset(masking, train, config.dataset_mask_idx)

    train_loader = torch.utils.data.DataLoader(
        GpuDataset(train_subset, device) if keep_on_gpu else train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
        # pin_memory=pin_memory,
        # num_workers=2,
        # persistent_workers=True
    )
    eval_train_loader = torch.utils.data.DataLoader(
        GpuDataset(train_subset, device) if keep_on_gpu else train_subset,
        batch_size=config.batch_size,
        shuffle=False,
        # pin_memory=pin_memory,
        # num_workers=0,
    )
    eval_test_loader = torch.utils.data.DataLoader(
        GpuDataset(test, device) if keep_on_gpu else test,
        batch_size=config.batch_size,
        shuffle=False,
        # pin_memory=pin_memory,
        # num_workers=0,
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
    optimizer = config.optimizer(params=model.parameters())

    regularizer_fn = None
    reg_val_iter = None
    reg_loss_fn = None
    noise_generator = None
    if config.regularizer is not None:
        regularizer_fn = config.regularizer()
        regularizer_fn.to(device)
        reg_loss_fn = config.loss(num_classes=train.num_classes, reduction="none")
        noise_generator = config.regularizer.create_noise_generator()
        if noise_generator is None:
            reg_val_iter = itertools.cycle(eval_test_loader)

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
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=100, warmup=10, active=10, repeat=5),
                on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler"),
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
            )
            prof.start()
            logger.info("Using profiler")

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
                    or (step < LOG_FREQUENCY and step % 100 == 0)
                    or (step % LOG_FREQUENCY == 0)
                ):
                    heavy_metrics = step % HEAVY_METRICS_LOG_FREQUENCY == 0
                    metrics = evaluate(
                        model=model,
                        step=step,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        key_prefix="eval",
                        train_loader=eval_train_loader,
                        test_loader=eval_test_loader,
                        compute_heavy_metrics=heavy_metrics,
                        last_step=False,
                    )
                    mlflow.log_metrics(metrics, step=step)

                    train_loss_mean = metrics[f"eval/train/loss/{config.loss.name}/mean"]
                    test_loss_mean = metrics[f"eval/test/loss/{config.loss.name}/mean"]
                    train_accuracy = metrics["eval/train/accuracy"]
                    test_accuracy = metrics["eval/test/accuracy"]

                    pbar.set_description(
                        f"L: {train_loss_mean:1.1e}|{test_loss_mean:1.1e}. A: {train_accuracy * 100:2.1f}%|{test_accuracy * 100:2.1f}%"
                    )

                if step % checkpoint_frequency == 0:
                    save_model(model, optimizer, step)

                if not keep_on_gpu:
                    x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(x)
                task_loss = loss_fn(logits, y)
                loss = task_loss

                reg_value = None
                if regularizer_fn is not None:
                    train_losses_per_sample = reg_loss_fn(logits, y)
                    # reduction="none" may return (B, C) for MSE or (B,) for CE;
                    # reduce to a true per-sample scalar (B,).
                    if train_losses_per_sample.dim() > 1:
                        train_losses_per_sample = train_losses_per_sample.mean(
                            dim=tuple(range(1, train_losses_per_sample.dim()))
                        )
                    if noise_generator is not None:
                        num_copies = config.regularizer.num_noisy_samples
                        val_losses_parts = []
                        for _ in range(num_copies):
                            x_noisy = noise_generator(x)
                            with torch.no_grad():
                                val_logits = model(x_noisy)
                                vl = reg_loss_fn(val_logits, y)
                                if vl.dim() > 1:
                                    vl = vl.mean(dim=tuple(range(1, vl.dim())))
                                val_losses_parts.append(vl)
                        val_losses_per_sample = torch.cat(val_losses_parts, dim=0)
                    else:
                        x_val, y_val = next(reg_val_iter)
                        if not keep_on_gpu:
                            x_val, y_val = x_val.to(device), y_val.to(device)
                        with torch.no_grad():
                            val_logits = model(x_val)
                            val_losses_per_sample = reg_loss_fn(val_logits, y_val)
                            if val_losses_per_sample.dim() > 1:
                                val_losses_per_sample = val_losses_per_sample.mean(
                                    dim=tuple(range(1, val_losses_per_sample.dim()))
                                )
                    reg_value = regularizer_fn(train_losses_per_sample, val_losses_per_sample)
                    loss = task_loss + config.regularizer.weight * reg_value

                loss.backward()

                mlflow.log_metrics(
                    {
                        "train/task_loss": task_loss.item(),
                        "train/total_loss": loss.item(),
                        **({
                            f"train/regularizer/{config.regularizer.name}": reg_value.item(),
                            f"train/regularizer/{config.regularizer.name}/weighted": (
                                config.regularizer.weight * reg_value
                            ).item(),
                        } if reg_value is not None else {}),
                    },
                    step=step,
                )

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

    evaluate(
        model=model,
        step=step,
        optimizer=optimizer,
        loss_fn=loss_fn,
        key_prefix="eval",
        train_loader=eval_train_loader,
        test_loader=eval_test_loader,
        compute_heavy_metrics=heavy_metrics,
        last_step=False,
    )
    save_model(model, optimizer, step)

    logger.info(f"Ending training: '{config.name}'")


def train(
    exp_name: str,
    total_steps: int,
    cfg: TrainConfig | RestartConfig,
    run_name: str | None = None,
    checkpoint_frequency: int | None = None,
) -> str:
    run_name = run_name or (cfg.name if isinstance(cfg, TrainConfig) else cfg.run_id)
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
