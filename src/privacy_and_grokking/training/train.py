import json
import numpy as np
import polars as pl
import random
import torch
import torch.nn as nn

from pydantic_core import to_jsonable_python
from torchvision import datasets, transforms
from tqdm.auto import tqdm

from .metrics import Metrics, ModeMetrics
from .parameters import ModelArchitectures, Parameters
from ..logger import get_logger
from ..models import CNN, MLP
from ..path_keeper import get_path_keeper
from ..utils import eval_mode, get_device

DTYPE = torch.float32

def _eval(model: nn.Module, loss_fn, loader) -> tuple[float, float, pl.DataFrame]:
    device = get_device()
    one_hots = torch.eye(10, 10).to(device)

    loss = 0
    correct = 0
    number = 0
    index_list = []
    logit_list = []
    label_list = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss += loss_fn(logits, one_hots[y]).item()
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
        last_layer = sum(torch.pow(p, 2).sum().item() for p in model.model[-1].parameters())
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

def save_model(model: nn.Module, optimizer: torch.optim.Optimizer, x) -> None:
    pk = get_path_keeper()
    torch.save(model.state_dict(), pk.MODEL_TORCH)
    torch.save(optimizer.state_dict(), pk.OPTIMIZER)
    torch.onnx.export(model, x, pk.MODEL_ONNX, verbose=False)

def train(params: Parameters) -> None:
    logger = get_logger()
    pk = get_path_keeper()
    pk.set_params({"model": params.name})

    logger.info(f"Starting training: '{params.name}'", extra={"parameters": params.model_dump()})

    # Settings.
    logger.info("Preparing seeds and defaults.")
    torch.set_default_dtype(DTYPE)
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed_all(params.seed)
    random.seed(params.seed)
    np.random.seed(params.seed)
    device = get_device()

    # Dataset
    logger.info("Preparing dataset.")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)), # MNIST mean/std
    ])

    train = datasets.MNIST(
        root=pk.CACHE,
        train=True, 
        transform=transform,
        download=True
    )
    test = datasets.MNIST(
        root=pk.CACHE,
        train=False, 
        transform=transform,
        download=True
    )

    if params.sample_size:
        train_subset, _ = torch.utils.data.random_split(train, [params.sample_size, len(train) - params.sample_size])
    else:
        train_subset = train

    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=params.batch_size, shuffle=True)
    eval_train_loader = torch.utils.data.DataLoader(train_subset, batch_size=200, shuffle=False)
    eval_test_loader = torch.utils.data.DataLoader(test, batch_size=200, shuffle=False)

    # Model
    logger.info("Preparing model.")
    match (params.model):
        case ModelArchitectures.MLP:
            model = MLP()
        case ModelArchitectures.CNN:
            model = CNN()
        case _:
            raise ValueError(f"Unknown model architecture '{params.model}' specified.")
    model.to(device)

    if params.initialization_scale is not None:
        with torch.no_grad():
            for p in model.parameters():
                p.data *= params.initialization_scale

    # Optimizer and loss function
    logger.info("Preparing optimizer and loss function.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=params.optimizer.learning_rate, weight_decay=params.optimizer.weight_decay)
    loss_fn = nn.MSELoss()

    # Training loop
    logger.info("Starting training loop.")
    one_hots = torch.eye(10, 10).to(device)
    step = 0
    data: list[Metrics] = []
    with tqdm(total=params.optimization_steps) as pbar:
        while step < params.optimization_steps:
            for x, y in train_loader:
                if step >= params.optimization_steps:
                    break

                if (step < 30) or (step < 150 and step % 10 == 0) or step % params.log_frequency == 0:
                    metrics = evaluate(step, model, x, optimizer, loss_fn, eval_test_loader, eval_test_loader)
                    pbar.set_description("L: {0:1.1e}|{1:1.1e}. A: {2:2.1f}%|{3:2.1f}%".format(
                        metrics.train.loss,
                        metrics.test.loss,
                        metrics.train.accuracy * 100,
                        metrics.test.accuracy * 100,
                    ))

                optimizer.zero_grad()
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = loss_fn(logits, one_hots[y])
                loss.backward()
                optimizer.step()

                step += 1
                pbar.update(1)
    logger.info("Training complete.")

    # Saving results
    logger.info("Saving results.")
    x, _ = next(iter(train_loader))
    evaluate(step, model, x, optimizer, loss_fn, eval_train_loader, eval_test_loader)
    pk.set_params({"step": step})
    with pk.TRAIN_METRICS.open("w") as f:
        json.dump(data, f, default=to_jsonable_python)
    logger.info(f"Ending training: '{params.name}'")
