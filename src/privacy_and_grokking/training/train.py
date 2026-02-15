import json
import random
from collections.abc import Callable

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from pydantic import BaseModel
from pydantic_core import to_jsonable_python
from tqdm.auto import tqdm

from privacy_and_grokking.config import AdamW, LossConfig, TrainConfig
from privacy_and_grokking.datasets import create_masking, generate_datasets, mask_dataset
from privacy_and_grokking.logger import get_logger
from privacy_and_grokking.models import create_model
from privacy_and_grokking.path_keeper import get_path_keeper
from privacy_and_grokking.training.metrics import Metrics, ModeMetrics
from privacy_and_grokking.utils import eval_mode, get_device, set_all_seeds


def get_loss_fn(
    cfg: LossConfig, num_classes: int, device: torch.device
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    match cfg.name.lower():
        case "mse":
            one_hot = torch.eye(num_classes, num_classes).to(device)
            fn = nn.MSELoss()

            def loss(logits, labels: torch.Tensor) -> torch.Tensor:
                return fn(logits, one_hot[labels])

            return loss
        case "cross_entropy":
            return nn.CrossEntropyLoss()
        case _:
            raise ValueError(f"Unknown loss function: {cfg.name}")


def get_loss_fn_eval(
    cfg: LossConfig, num_classes: int, device: torch.device
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    match cfg.name.lower():
        case "mse":
            one_hot = torch.eye(num_classes, num_classes).to(device)
            fn = nn.MSELoss(reduction="none")

            def loss(logits, labels: torch.Tensor) -> torch.Tensor:
                return fn(logits, one_hot[labels]).mean(dim=1)

            return loss
        case "cross_entropy":
            return nn.CrossEntropyLoss(reduction="none")
        case _:
            raise ValueError(f"Unknown loss function: {cfg.name}")


def _eval(model: nn.Module, loss_fn, loader) -> tuple[float, float, float, pl.DataFrame]:
    device = get_device()

    all_losses = []
    correct = 0
    number = 0
    index_list = []
    logit_list = []
    label_list = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        losses = loss_fn(logits, y)
        all_losses.append(losses.detach().cpu())

        labels = torch.argmax(logits, dim=1)
        correct += torch.sum(labels == y).item()
        number += x.size(0)

        index_list.extend(range(number - x.size(0), number))
        label_list.append(y.detach().cpu().numpy())
        logit_list.append(logits.detach().cpu().numpy())

    all_losses_tensor = torch.cat(all_losses)
    loss_mean = all_losses_tensor.mean().item()
    loss_std = all_losses_tensor.std().item()

    df = pl.DataFrame(
        {
            "index": index_list,
            "correct_label": np.concatenate(label_list),
            **{
                f"logit_{i}": np.concatenate([logits[:, i] for logits in logit_list])
                for i in range(logit_list[0].shape[1])
            },
        }
    )

    return loss_mean, loss_std, (correct / number), df


def save_model(model: nn.Module, optimizer: torch.optim.Optimizer, x) -> None:
    pk = get_path_keeper()
    torch.save(model.state_dict(), pk.MODEL_TORCH)
    torch.save(optimizer.state_dict(), pk.OPTIMIZER)
    torch.onnx.export(model, x, pk.MODEL_ONNX, verbose=False)

    states = {
        "random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch-cuda": torch.cuda.get_rng_state_all(),
    }
    torch.save(states, pk.RNG_STATE)


def evaluate(
    step: int,
    model: nn.Module,
    x,
    optimizer,
    loss_fn,
    eval_train_loader,
    eval_test_loader,
) -> Metrics:
    pk = get_path_keeper()
    pk.set_params({"step": step})

    with eval_mode(model):
        train_loss, train_loss_std, train_accuracy, df_train = _eval(
            model, loss_fn, eval_train_loader
        )
        df_train = df_train.with_columns(pl.lit(step).alias("step"))
        test_loss, test_loss_std, test_accuracy, df_test = _eval(model, loss_fn, eval_test_loader)
        df_test = df_test.with_columns(pl.lit(step).alias("step"))

        all_layer = sum(torch.pow(p, 2).sum().item() for p in model.parameters())
        norm = float(np.sqrt(all_layer))
        last_layer = sum(torch.pow(p, 2).sum().item() for p in model.last_layer.parameters())
        last_layer_norm = float(np.sqrt(last_layer))

        metrics = Metrics(
            step=step,
            train=ModeMetrics(
                loss=train_loss,
                loss_std=train_loss_std,
                accuracy=train_accuracy,
            ),
            test=ModeMetrics(
                loss=test_loss,
                loss_std=test_loss_std,
                accuracy=test_accuracy,
            ),
            norm=norm,
            last_layer_norm=last_layer_norm,
        )

        with pk.TRAIN_METRICS.open("a") as f:
            f.write(metrics.model_dump_json() + "\n")
        df_train.write_parquet(pk.TRAIN_LOGITS)
        df_test.write_parquet(pk.TEST_LOGITS)
        save_model(model, optimizer, x)

    return metrics


def get_optimizer(cfg: AdamW, params) -> torch.optim.Optimizer:
    match cfg.name.lower():
        case "adamw":
            return torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        case _:
            raise ValueError(f"Unknown optimizer: {cfg.name}")


class RestartConfig(BaseModel):
    name: str
    checkpoint: int
    dataset_mask_idx: int

    @property
    def full_name(self) -> str:
        return f"{self.name}_{self.dataset_mask_idx}"


def train(cfg: TrainConfig | RestartConfig) -> None:
    logger = get_logger()
    pk = get_path_keeper()
    model_name = cfg.full_name
    pk.set_params({"model": model_name})

    if isinstance(cfg, RestartConfig):
        pk.set_params({"step": cfg.checkpoint})
        logger.info(f"Restarting training from checkpoint: '{model_name}' at step {cfg.checkpoint}")
        logger.warning(
            "Make sure you are using the same device as when the checkpoint was created."
        )
        restart = True
        config = TrainConfig.model_validate_json(pk.TRAIN_CONFIG.read_bytes())
    else:
        logger.info(f"Starting training: '{model_name}'")
        restart = False
        config = cfg

    # Config.
    logger.info("Training configuration.", extra={"config": config.model_dump()})
    if not restart:
        with pk.TRAIN_CONFIG.open("w") as f:
            json.dump(config.model_dump(), f, default=to_jsonable_python)

    # Settings.
    logger.info("Preparing seeds and defaults.")
    torch.set_default_dtype(torch.float32)
    if restart:
        states = torch.load(pk.RNG_STATE, weights_only=False)
        random.setstate(states["random"])
        np.random.set_state(states["numpy"])
        torch.set_rng_state(states["torch"])
        if torch.cuda.is_available() and states["torch-cuda"]:
            torch.cuda.set_rng_state_all(states["torch-cuda"])
    else:
        set_all_seeds(config.seed)
    device_name = get_device()
    device = torch.device(device_name)
    logger.info(f"Using device {device_name}", extra={"device": device_name})

    # Dataset
    logger.info("Preparing dataset.")
    train, test = generate_datasets(config=config.dataset)
    masking = create_masking(
        config=config.dataset_mask,
        num_samples=len(train),
        num_classes=train.num_classes,
    )
    train_subset = mask_dataset(masking, train, cfg.dataset_mask_idx)

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
    batch_offset = config.checkpoint % len(train_loader) if restart else 0

    # Model
    logger.info("Preparing model.")
    model = create_model(
        name=config.model,
        input_dim=train.input_shape,
        num_classes=train.num_classes,
        initialization_scale=config.initialization_scale,
    )
    model.to(device)
    if restart:
        model.load_state_dict(torch.load(pk.MODEL_TORCH, map_location=device, weights_only=False))

    # Optimizer and loss function
    logger.info("Preparing optimizer and loss function.")
    loss_fn = get_loss_fn(config.loss, train.num_classes, device)
    loss_fn_eval = get_loss_fn_eval(config.loss, train.num_classes, device)
    optimizer = get_optimizer(config.optimizer, model.parameters())
    if restart:
        optimizer.load_state_dict(torch.load(pk.OPTIMIZER, map_location=device, weights_only=False))

    # Training loop
    logger.info("Starting training loop.")
    step = config.checkpoint if restart else 0
    with tqdm(total=config.optimization_steps) as pbar:
        pbar.update(step)
        while step < config.optimization_steps:
            for x, y in train_loader:
                # Skip batches we've already processed in this epoch
                if restart and batch_offset > 0:
                    batch_offset -= 1
                    continue

                x, y = x.to(device), y.to(device)

                if step >= config.optimization_steps:
                    break

                if (
                    (step < 50)
                    or (step < config.log_frequency and step % 50 == 0)
                    or (step % config.log_frequency == 0)
                ):
                    metrics = evaluate(
                        step, model, x, optimizer, loss_fn_eval, eval_train_loader, eval_test_loader
                    )
                    pbar.set_description(
                        f"L: {metrics.train.loss:1.1e}|{metrics.test.loss:1.1e}. A: {metrics.train.accuracy * 100:2.1f}%|{metrics.test.accuracy * 100:2.1f}%"
                    )

                optimizer.zero_grad()
                logits = model(x)
                loss = loss_fn(logits, y)
                loss.backward()
                optimizer.step()

                step += 1
                pbar.update(1)
    logger.info("Training complete.")

    # Saving results
    logger.info("Saving results.")
    x, _ = next(iter(train_loader))
    evaluate(
        step, model, x.to(device), optimizer, loss_fn_eval, eval_train_loader, eval_test_loader
    )
    pk.set_params({"step": step})

    logger.info(f"Ending training: '{config.name}'")
