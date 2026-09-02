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
)
from privacy_and_grokking.metrics import MetricsConfig, evaluate
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
    load_all_to_gpu: bool = False,
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

    metrics_config: MetricsConfig = config.metrics
    log_frequency = metrics_config.log_frequency
    heavy_metrics_log_frequency = metrics_config.heavy_metrics_log_frequency

    device_name = get_device()
    device = torch.device(device_name)
    logger.info(f"Using device {device_name}", device=device_name)

    logger.info("Preparing dataset.")
    keep_on_gpu = torch.cuda.is_available()
    data_container = config.data()
    
    train_raw = data_container.train
    train_canary = data_container.train_canary
    if train_canary is not None:
        train_subset = torch.utils.data.ConcatDataset([train_raw, train_canary])
    else:
        train_subset = train_raw
        
    test = data_container.test

    mlflow.log_params(
        {
            "model_name": config.model.name,
            "loss_function": config.loss.name,
            "weight_decay": getattr(config.optimizer, "weight_decay", None),
            "initialization_scale": getattr(config.model, "initialization_scale", None),
            "learning_rate": getattr(config.optimizer, "lr", None),
            "optimizer": config.optimizer.name,
            "name": config.name,
            "train_size": len(train_subset),
            "batch_size": config.batch_size,
        }
    )

    def maybe_gpu_dataset(dataset):
        if keep_on_gpu and (load_all_to_gpu or len(dataset) < 5000):
            return GpuDataset(dataset, device)
        return dataset

    def get_dl_kwargs(dataset):
        is_gpu = isinstance(dataset, GpuDataset)
        return {
            "batch_size": config.batch_size if config.batch_size != -1 else len(train_subset),
            "num_workers": 0 if is_gpu else config.num_workers,
            "pin_memory": (config.num_workers > 0 and keep_on_gpu and not is_gpu),
            "persistent_workers": (config.num_workers > 0 and not is_gpu),
        }

    train_ds = maybe_gpu_dataset(train_subset)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
        **get_dl_kwargs(train_ds)
    )
    
    eval_train_ds = maybe_gpu_dataset(train_raw)
    eval_train_loader = torch.utils.data.DataLoader(
        eval_train_ds,
        shuffle=False,
        **get_dl_kwargs(eval_train_ds)
    )
    
    eval_test_ds = maybe_gpu_dataset(test)
    eval_test_loader = torch.utils.data.DataLoader(
        eval_test_ds,
        shuffle=False,
        **get_dl_kwargs(eval_test_ds)
    )
    
    eval_train_canary_loader = None
    if train_canary is not None:
        eval_train_canary_ds = maybe_gpu_dataset(train_canary)
        eval_train_canary_loader = torch.utils.data.DataLoader(
            eval_train_canary_ds,
            shuffle=False,
            **get_dl_kwargs(eval_train_canary_ds)
        )
        
    eval_test_canary_loader = None
    if data_container.test_canary is not None:
        eval_test_canary_ds = maybe_gpu_dataset(data_container.test_canary)
        eval_test_canary_loader = torch.utils.data.DataLoader(
            eval_test_canary_ds,
            shuffle=False,
            **get_dl_kwargs(eval_test_canary_ds)
        )
    
    epoch_log_frequency = None
    if metrics_config.log_every_n_epochs is not None:
        epoch_log_frequency = metrics_config.log_every_n_epochs * len(train_loader)
        logger.info(f"Adding epoch_log_frequency: {epoch_log_frequency} ({metrics_config.log_every_n_epochs} epochs)")
        
    epoch_heavy_log_frequency = None
    if metrics_config.heavy_log_every_n_epochs is not None:
        epoch_heavy_log_frequency = metrics_config.heavy_log_every_n_epochs * len(train_loader)
        logger.info(f"Adding epoch_heavy_log_frequency: {epoch_heavy_log_frequency} ({metrics_config.heavy_log_every_n_epochs} epochs)")

    batch_offset = cfg.checkpoint % len(train_loader) if restart else 0

    logger.info("Preparing model.")
    model = config.model(
        input_dim=data_container.input_shape,
        num_classes=data_container.num_classes,
    )
    model.to(device)

    logger.info("Preparing optimizer and loss function.")
    loss_fn = config.loss(num_classes=data_container.num_classes)
    optimizer = config.optimizer(params=model.parameters())

    regularizer_fn = None
    reg_loss_fn = None
    regularizer_cfg = config.regularizer
    if regularizer_cfg is not None:
        regularizer_fn = regularizer_cfg()
        reg_loss_fn = config.loss.model_copy(update={"reduction": "none"})(
            num_classes=data_container.num_classes
        )

    logger.info("Preparing seeds and defaults.")
    torch.set_default_dtype(torch.float32)
    torch.backends.cudnn.benchmark = True
    set_all_seeds(config.seed)

    if restart:
        load_model(model, optimizer, cfg.run_id, cfg.checkpoint, device)

    scheduler = config.scheduler(
        optimizer=optimizer,
        optimization_steps=optimization_steps,
        last_epoch=cfg.checkpoint if restart else -1,
    )

    norm_mean = None
    norm_std = None
    if data_container.normalization is not None:
        norm_mean = torch.tensor(data_container.normalization.mean, device=device).view(-1, 1, 1)
        norm_std = torch.tensor(data_container.normalization.std, device=device).view(-1, 1, 1)

    logger.info("Starting training loop.")
    model.train()
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

                steps_per_epoch = max(1, len(train_loader))
                current_epoch = step / steps_per_epoch

                if (
                    (step < 50)
                    or (step < log_frequency and step % 100 == 0)
                    or (step % log_frequency == 0)
                    or (epoch_log_frequency is not None and step % epoch_log_frequency == 0 and current_epoch <= 500)
                ):
                    heavy_metrics = (step % heavy_metrics_log_frequency == 0) or (
                            epoch_heavy_log_frequency is not None 
                            and step % epoch_heavy_log_frequency == 0
                        and current_epoch <= 500
                    )
                    metrics = evaluate(
                        model=model,
                        step=step,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        key_prefix="eval",
                        train_loader=eval_train_loader,
                        test_loader=eval_test_loader,
                        train_canary_loader=eval_train_canary_loader,
                        test_canary_loader=eval_test_canary_loader,
                        compute_heavy_metrics=heavy_metrics,
                        num_classes=data_container.num_classes,
                        metrics_config=metrics_config,
                        normalization=data_container.normalization,
                    )
                    model.train()
                    metrics["epoch"] = step / max(1, len(train_loader))
                    mlflow.log_metrics(metrics, step=step)

                    train_loss_mean = metrics[f"eval/train/loss/{config.loss.name}/mean"]
                    test_loss_mean = metrics[f"eval/test/loss/{config.loss.name}/mean"]
                    train_accuracy = metrics["eval/train/accuracy"]
                    test_accuracy = metrics["eval/test/accuracy"]

                    pbar.set_description(
                        f"L: {train_loss_mean:1.1e}|{test_loss_mean:1.1e}. A: {train_accuracy * 100:2.1f}%|{test_accuracy * 100:2.1f}%"
                    )

                should_checkpoint = False

                if config.checkpoint_frequency_step is not None and config.checkpoint_frequency_step > 0:
                    if step % config.checkpoint_frequency_step == 0:
                        should_checkpoint = True
                    if (
                        config.early_checkpoint_frequency_step is not None
                        and config.early_checkpoint_frequency_step > 0
                        and step <= config.early_checkpoint_step_limit
                        and step % config.early_checkpoint_frequency_step == 0
                    ):
                        should_checkpoint = True

                if config.checkpoint_frequency_epoch is not None and config.checkpoint_frequency_epoch > 0:
                    steps_per_epoch = max(1, len(train_loader))
                    current_epoch = step / steps_per_epoch
                    if current_epoch <= 500:
                        epoch_interval = config.checkpoint_frequency_epoch * steps_per_epoch
                        if step % epoch_interval == 0:
                            should_checkpoint = True

                if should_checkpoint:
                    save_model(model, optimizer, step)

                x, y = x.to(device), y.to(device)
                if norm_mean is not None:
                    x = (x - norm_mean) / norm_std
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                task_loss = loss_fn(logits, y)
                if config.loss.name == "mse" and config.model.name in ["modular_transformer"]:
                    loss = task_loss * data_container.num_classes
                else:
                    loss = task_loss

                reg_value = None
                if regularizer_fn is not None:
                    train_losses_per_sample = reg_loss_fn(logits, y)
                    # Collapse extra dims to get (B,) — e.g. MSE gives (B, C).
                    if train_losses_per_sample.dim() > 1:
                        extra = tuple(range(1, train_losses_per_sample.dim()))
                        train_losses_per_sample = train_losses_per_sample.mean(dim=extra)
                    reg_value = regularizer_fn(train_losses_per_sample)
                    loss = task_loss + reg_value

                loss.backward()
                optimizer.step()
                scheduler.step()

                if step % log_frequency == 0:
                    mlflow.log_metrics(
                        {
                            "train/task_loss": task_loss.item(),
                            "train/total_loss": loss.item(),
                            **(
                                {
                                    f"train/regularizer/{regularizer_cfg.name}": reg_value.item(),
                                }
                                if reg_value is not None
                                else {}
                            ),
                        },
                        step=step,
                    )

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
        train_canary_loader=eval_train_canary_loader,
        test_canary_loader=eval_test_canary_loader,
        compute_heavy_metrics=heavy_metrics,
        num_classes=data_container.num_classes,
        metrics_config=metrics_config,
        normalization=data_container.normalization,
    )
    save_model(model, optimizer, step)

    logger.info(f"Ending training: '{config.name}'")


def train(
    exp_name: str,
    total_steps: int,
    cfg: TrainConfig | RestartConfig,
    run_name: str | None = None,
    load_all_to_gpu: bool = False,
) -> str:
    run_name = run_name or (cfg.name if isinstance(cfg, TrainConfig) else cfg.run_id)

    setup_mlflow(exp_name)

    log_handler = Logger()
    run_id = cfg.run_id if isinstance(cfg, RestartConfig) else None
    with (
        log_handler as logger,
        mlflow.start_run(run_id=run_id, run_name=run_name, log_system_metrics=None) as mlflow_run,
    ):
        returned_run_id = mlflow_run.info.run_id
        try:
            logger.bind(run_id=returned_run_id)
            train_handle(cfg, total_steps, load_all_to_gpu=load_all_to_gpu)
        except Exception as e:
            logger.error(f"Training failed with error: {e}", exc_info=True)
        finally:
            if log_handler.log_file_path and mlflow.active_run() is not None:
                mlflow.log_artifact(str(log_handler.log_file_path), "training_logs")
            logger.info("Training done")
    return returned_run_id
