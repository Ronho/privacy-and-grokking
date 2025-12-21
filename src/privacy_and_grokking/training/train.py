import json
import numpy as np
import polars as pl
import random
import torch
import torch.nn as nn

from pydantic import BaseModel
from pydantic_core import to_jsonable_python
from typing import Callable
from torchvision import datasets, transforms
from tqdm.auto import tqdm
from ..config import TrainConfig, LossConfig, AdamW
from .metrics import Metrics, ModeMetrics
from ..datasets import get_dataset
from ..logger import get_logger
from ..models import create_model
from ..path_keeper import get_path_keeper
from ..utils import eval_mode, get_device, set_all_seeds

DTYPE = torch.float32

def _eval(model: nn.Module, loss_fn, loader) -> tuple[float, float, pl.DataFrame]:
    device = get_device()

    loss = 0
    correct = 0
    number = 0
    index_list = []
    logit_list = []
    label_list = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss += loss_fn(logits, y).item()
        labels = torch.argmax(logits, dim=1)
        correct += torch.sum(labels == y.to(device)).item()
        number += x.size(0)

        index_list.extend(range(number - x.size(0), number))
        label_list.append(y.detach().cpu().numpy())
        logit_list.append(logits.detach().cpu().numpy())

    df = pl.DataFrame({
        "index": index_list,
        "correct_label": np.concatenate(label_list),
        **{f"logit_{i}": np.concatenate([logits[:, i] for logits in logit_list]) for i in range(logit_list[0].shape[1])}
    })

    return (loss/number), (correct/number), df

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

def evaluate(step: int, model: nn.Module, x, optimizer, loss_fn, eval_train_loader, eval_test_loader) -> Metrics:
    pk = get_path_keeper()
    pk.set_params({"step": step})

    with eval_mode(model):
        train_loss, train_accuracy, df_train = _eval(model, loss_fn, eval_train_loader)
        df_train = df_train.with_columns(pl.lit(step).alias("step"))
        test_loss, test_accuracy, df_test = _eval(model, loss_fn, eval_test_loader)
        df_test = df_test.with_columns(pl.lit(step).alias("step"))

        all_layer = sum(torch.pow(p, 2).sum().item() for p in model.parameters())
        norm = float(np.sqrt(all_layer))
        last_layer = sum(torch.pow(p, 2).sum().item() for p in model.last_layer.parameters())
        last_layer_norm = float(np.sqrt(last_layer))

        metrics = Metrics(
            step=step,
            train=ModeMetrics(
                loss=train_loss,
                accuracy=train_accuracy,
            ),
            test=ModeMetrics(
                loss=test_loss,
                accuracy=test_accuracy,
            ),
            norm=norm,
            last_layer_norm=last_layer_norm,
        )
        df_train.write_parquet(pk.TRAIN_LOGITS)
        df_test.write_parquet(pk.TEST_LOGITS)
        save_model(
            model,
            optimizer,
            x
        )

    return metrics

def get_loss_fn(cfg: LossConfig, num_classes: int, device: torch.device) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
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
        
def get_optimizer(cfg: AdamW, params) -> torch.optim.Optimizer:
    match cfg.name.lower():
        case "adamw":
            return torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        case _:
            raise ValueError(f"Unknown optimizer: {cfg.name}")

class RestartConfig(BaseModel):
    name: str
    checkpoint: int

def train(cfg: TrainConfig | RestartConfig) -> None:
    logger = get_logger()
    pk = get_path_keeper()

    pk.set_params({"model": cfg.name})
    if isinstance(cfg, RestartConfig):
        pk.set_params({"step": cfg.checkpoint})
        logger.info(f"Restarting training from checkpoint: '{cfg.name}' at step {cfg.checkpoint}")
        logger.warning("Make sure you are using the same device as when the checkpoint was created.")
        restart = True
        config = TrainConfig.model_validate_json(pk.TRAIN_CONFIG.read_bytes())
    else:
        logger.info(f"Starting training: '{cfg.name}'")
        restart = False
        config = cfg
    
    # Config.
    logger.info("Training configuration.", extra={"config": config.model_dump()})
    if not restart:
        with pk.TRAIN_CONFIG.open("w") as f:
            json.dump(config.model_dump(), f, default=to_jsonable_python)

    # Settings.
    logger.info("Preparing seeds and defaults.")
    torch.set_default_dtype(DTYPE)
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
    train, _, test, input_shape, num_classes = get_dataset(
        name=config.dataset.name,
        train_ratio=config.dataset.train_ratio,
        train_size=config.dataset.train_size,
        canary=config.dataset.canary.name if config.dataset.canary is not None else None,
        **(config.dataset.canary.model_dump(exclude=["name"]) if config.dataset.canary is not None else {})
    )
    train_loader = torch.utils.data.DataLoader(train, batch_size=config.batch_size, shuffle=True)
    eval_train_loader = torch.utils.data.DataLoader(train, batch_size=config.batch_size, shuffle=False)
    eval_test_loader = torch.utils.data.DataLoader(test, batch_size=config.batch_size, shuffle=False)
    batch_offset = cfg.checkpoint % len(train_loader) if restart else 0

    # Model
    logger.info("Preparing model.")
    model = create_model(
        name=config.model,
        input_dim=input_shape,
        num_classes=num_classes,
        initialization_scale=config.initialization_scale
    )
    model.to(device)
    if restart:
        model.load_state_dict(torch.load(pk.MODEL_TORCH, map_location=device, weights_only=False))

    # Optimizer and loss function
    logger.info("Preparing optimizer and loss function.")
    loss_fn = get_loss_fn(config.loss, num_classes, device)
    optimizer = get_optimizer(config.optimizer, model.parameters())
    if restart:
        optimizer.load_state_dict(torch.load(pk.OPTIMIZER, map_location=device, weights_only=False))

    # Training loop
    logger.info("Starting training loop.")
    step = cfg.checkpoint if restart else 0
    data: list[Metrics] = []
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

                if (step < 30) or (step < 150 and step % 10 == 0) or step % config.log_frequency == 0:
                    metrics = evaluate(step, model, x, optimizer, loss_fn, eval_train_loader, eval_test_loader)
                    data.append(metrics)
                    pbar.set_description("L: {0:1.1e}|{1:1.1e}. A: {2:2.1f}%|{3:2.1f}%".format(
                        metrics.train.loss,
                        metrics.test.loss,
                        metrics.train.accuracy * 100,
                        metrics.test.accuracy * 100,
                    ))

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
    evaluate(step, model, x.to(device), optimizer, loss_fn, eval_train_loader, eval_test_loader)
    pk.set_params({"step": step})

    if restart:
        if pk.TRAIN_METRICS.exists():
            existing_data = json.loads(pk.TRAIN_METRICS.read_text())
            existing_metrics = [Metrics.model_validate(m) for m in existing_data]
        else:
            existing_metrics = []
        lookup = {m.step: m for m in existing_metrics}
        for m in data:
            lookup[m.step] = m
        data = sorted(lookup.values(), key=lambda m: m.step)

    with pk.TRAIN_METRICS.open("w") as f:
        json.dump(data, f, default=to_jsonable_python)
    logger.info(f"Ending training: '{config.name}'")
