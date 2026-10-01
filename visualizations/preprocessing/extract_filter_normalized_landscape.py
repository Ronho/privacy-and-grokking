#!/usr/bin/env python3
"""extract_filter_normalized_landscape.py.

Evaluates the 2D loss landscape and weight-norm surface around a single trained checkpoint
using filter-wise normalized random perturbation directions (Li et al., NeurIPS 2018).

Filter-wise normalization eliminates scale-invariance distortions (e.g. ReLU scaling,
BatchNorm, LayerNorm) by scaling each slice of random Gaussian directions d1, d2
to match the Frobenius/L2 norm of the corresponding model weight filter/neuron:

    d_{l, j} <- (d_{l, j} / ||d_{l, j}||_F) * ||W_{l, j}||_F

The 2D loss surface is evaluated on the plane:
    f(alpha, beta) = L(theta* + alpha * d1 + beta * d2)
where (0, 0) corresponds exactly to the unperturbed checkpoint model theta*.

Required Inputs:
  - Checkpoint model weights (`checkpoints/{step}/model.pth`) from the local cache
    (`cache/runs/{run_id}/...`) or downloaded from the MLflow tracking server.
  - Training configuration (`training_config.json`) for data container and model architecture.

Generated Outputs:
  - cache/runs/{run_id}/filter_normalized/grid_{step_or_final}.json:
      2D loss landscape grid, weight norms, and metadata evaluated across the 2D plane.
"""

from __future__ import annotations

import io
import json
import urllib.request
from typing import Annotated

import numpy as np
import torch
import typer
from torch.utils.data import DataLoader, default_collate
from tqdm import tqdm

from privacy_and_grokking.artifacts.cache import (
    DEFAULT_TRACKING_URI,
    get_training_config,
)
from privacy_and_grokking.artifacts.local import (
    filter_normalized_grid_path,
    local_run_dir,
)
from privacy_and_grokking.config import TrainConfig
from privacy_and_grokking.metrics.filter_norm_landscape import (
    compute_perturbed_state_dict,
    compute_total_weight_norm,
    create_random_direction,
    filter_normalize_direction,
)
from privacy_and_grokking.utils import Logger

app = typer.Typer(pretty_exceptions_enable=False)


def get_available_checkpoints(run_id: str, tracking_uri: str) -> list[int]:
    """Retrieve sorted list of available checkpoint step integers for a run."""
    checkpoints_dir = local_run_dir(run_id) / "checkpoints"
    info_file = checkpoints_dir / "info.json"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    if info_file.exists():
        try:
            return sorted(json.loads(info_file.read_bytes())["steps"])
        except Exception:
            pass

    url = (
        f"{tracking_uri.rstrip('/')}/api/2.0/mlflow/artifacts/list?run_id={run_id}&path=checkpoints"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "privacy-and-grokking"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        files = data.get("files", [])
        steps = []
        for f in files:
            p = f.get("path", "")
            step_str = p.split("/")[-1]
            if step_str.isdigit():
                steps.append(int(step_str))
        if steps:
            presorted = sorted(set(steps))
            info_file.write_text(json.dumps({"steps": presorted}))
            return presorted
    except Exception as e:
        print(f"Warning: Failed to fetch checkpoints via REST ({e}). Fallback to local disk...")

    local_steps = []
    if checkpoints_dir.exists():
        for item in checkpoints_dir.iterdir():
            if item.is_dir() and item.name.isdigit() and (item / "model.pth").exists():
                local_steps.append(int(item.name))
    if local_steps:
        presorted = sorted(set(local_steps))
        info_file.write_text(json.dumps({"steps": presorted}))
        return presorted

    raise RuntimeError(f"No checkpoints found for run '{run_id}'.")


def get_checkpoint_weight(run_id: str, step: int, tracking_uri: str) -> dict[str, torch.Tensor]:
    """Load model checkpoint weights from local cache or download on demand."""
    model_file = local_run_dir(run_id) / "checkpoints" / str(step) / "model.pth"
    model_file.parent.mkdir(parents=True, exist_ok=True)
    if model_file.exists():
        return torch.load(model_file, map_location="cpu", weights_only=True)

    base_url = tracking_uri.rstrip("/")
    url = f"{base_url}/get-artifact?path=checkpoints/{step}/model.pth&run_uuid={run_id}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "privacy-and-grokking"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()

        model_file.write_bytes(data)
        buf = io.BytesIO(data)
        return torch.load(buf, map_location="cpu", weights_only=True)
    except Exception as e:
        raise RuntimeError(
            f"Failed to fetch checkpoint at step {step} for run {run_id} from {url}: {e}"
        ) from e


def safe_collate(batch):
    """Robust DataLoader collate function handling mixed tensor types."""
    if isinstance(batch[0], tuple) and len(batch[0]) == 2:
        xs, ys = zip(*batch, strict=True)
        if any(isinstance(y, torch.Tensor) for y in ys) and any(
            not isinstance(y, torch.Tensor) for y in ys
        ):
            ys = [y if isinstance(y, torch.Tensor) else torch.tensor(y) for y in ys]
            batch = list(zip(xs, ys, strict=True))
    return default_collate(batch)


@torch.no_grad()
def evaluate_loss_tensors(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_fn: torch.nn.Module,
    batch_size: int = 2000,
) -> float:
    """Evaluate loss on pre-batched dataset tensors."""
    model.eval()
    total_loss = 0.0
    n = len(y)
    for i in range(0, n, batch_size):
        bx = x[i : i + batch_size]
        by = y[i : i + batch_size]
        preds = model(bx)
        loss = loss_fn(preds, by)
        total_loss += float(loss.item()) * len(by)
    return total_loss / max(1, n)


def extract_filter_normalized_grid(
    run_id: str,
    step: int | None = None,
    resolution: int = 40,
    x_min: float = -1.0,
    x_max: float = 1.0,
    y_min: float = -1.0,
    y_max: float = 1.0,
    seed_x: int = 42,
    seed_y: int = 43,
    ignore_bias_and_bn: bool = False,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    force: bool = False,
) -> dict:
    """Computes filter-normalized 2D loss and weight-norm surface around a checkpoint."""
    grid_file = filter_normalized_grid_path(run_id, step if step is not None else "final")
    grid_file.parent.mkdir(parents=True, exist_ok=True)

    if grid_file.exists() and not force:
        try:
            data = json.loads(grid_file.read_bytes())
            if (
                data.get("resolution") == resolution
                and data.get("x_range") == [x_min, x_max]
                and data.get("y_range") == [y_min, y_max]
            ):
                print(f"Loading cached filter-normalized grid from {grid_file}...")
                return data
        except Exception:
            pass

    available_steps = get_available_checkpoints(run_id=run_id, tracking_uri=tracking_uri)
    target_step = step if step is not None else available_steps[-1]
    print(f"Extracting filter-normalized landscape for Run ID '{run_id}' at Step {target_step}...")

    base_weights = get_checkpoint_weight(run_id=run_id, step=target_step, tracking_uri=tracking_uri)

    # 1. Sample 2 Gaussian random directions
    raw_d1 = create_random_direction(base_weights, seed=seed_x)
    raw_d2 = create_random_direction(base_weights, seed=seed_y)

    # 2. Apply filter-wise normalization
    d1 = filter_normalize_direction(
        base_weights, raw_d1, ignore_bias_and_bn=ignore_bias_and_bn
    )
    d2 = filter_normalize_direction(
        base_weights, raw_d2, ignore_bias_and_bn=ignore_bias_and_bn
    )

    # 3. Setup Model and Dataset
    training_config: TrainConfig = get_training_config(run_id=run_id, tracking_uri=tracking_uri)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_container = training_config.data()
    model = training_config.model(
        input_dim=data_container.input_shape,
        num_classes=data_container.num_classes,
    ).to(device)

    loss_fn = training_config.loss(num_classes=data_container.num_classes)

    norm_mean = (
        torch.tensor(data_container.normalization.mean, device=device, dtype=torch.float32).view(
            -1, 1, 1
        )
        if data_container.normalization
        else None
    )
    norm_std = (
        torch.tensor(data_container.normalization.std, device=device, dtype=torch.float32).view(
            -1, 1, 1
        )
        if data_container.normalization
        else None
    )

    dataset_tensors = {}
    target_datasets = [("train", data_container.train), ("test", data_container.test)]
    if data_container.train_canary:
        target_datasets.append(("train_canary", data_container.train_canary))

    for name, ds in target_datasets:
        loader = DataLoader(
            ds,
            batch_size=len(ds),
            shuffle=False,
            collate_fn=safe_collate,
        )
        batch = next(iter(loader))
        bx, by = batch[0], batch[1]
        bx = bx.to(device, dtype=torch.float32)
        by = by.to(device)
        if norm_mean is not None and norm_std is not None and bx.dim() == 4:
            bx = (bx - norm_mean) / norm_std
        dataset_tensors[name] = (bx, by)

    # 4. Generate 2D Grid
    x_coords = np.linspace(x_min, x_max, resolution)
    y_coords = np.linspace(y_min, y_max, resolution)
    xx, yy = np.meshgrid(x_coords, y_coords)
    grid_points = np.stack([xx.ravel(), yy.ravel()], axis=1)

    print(
        f"Evaluating filter-normalized loss on {len(grid_points)} grid points "
        f"({resolution}x{resolution})..."
    )
    weight_norms: list[float] = []
    losses: dict[str, list[float]] = {name: [] for name in dataset_tensors}

    for pt in tqdm(grid_points, desc="Filter-norm grid evaluation"):
        alpha, beta = float(pt[0]), float(pt[1])
        perturbed_state = compute_perturbed_state_dict(
            base_weights, d1, alpha=alpha, d2=d2, beta=beta
        )
        w_norm = compute_total_weight_norm(perturbed_state)
        weight_norms.append(w_norm)

        # Move perturbed state to device
        state_on_device = {k: v.to(device) for k, v in perturbed_state.items()}
        model.load_state_dict(state_on_device)

        for name, (dx, dy) in dataset_tensors.items():
            l_val = evaluate_loss_tensors(model, dx, dy, loss_fn, batch_size=2000)
            losses[name].append(l_val)

    payload = {
        "run_id": run_id,
        "step": int(target_step),
        "resolution": int(resolution),
        "x_range": [float(x_min), float(x_max)],
        "y_range": [float(y_min), float(y_max)],
        "seed_x": int(seed_x),
        "seed_y": int(seed_y),
        "ignore_bias_and_bn": bool(ignore_bias_and_bn),
        "x_coords": x_coords.tolist(),
        "y_coords": y_coords.tolist(),
        "grid_points": grid_points.tolist(),
        "weight_norms": weight_norms,
        "losses": losses,
    }

    grid_file.write_text(json.dumps(payload))
    print(f"Saved filter-normalized grid to {grid_file} ({resolution}x{resolution} points)")
    return payload


@app.command()
def main(
    run_id: Annotated[
        str, typer.Option("--run-id", "-r", help="MLflow run ID to evaluate landscape for.")
    ],
    step: Annotated[
        int | None,
        typer.Option("--step", "-s", help="Checkpoint step to center around (default: final)."),
    ] = None,
    resolution: Annotated[
        int, typer.Option("--resolution", help="Points per grid axis.")
    ] = 40,
    x_min: Annotated[float, typer.Option("--x-min", help="Minimum alpha coordinate.")] = -1.0,
    x_max: Annotated[float, typer.Option("--x-max", help="Maximum alpha coordinate.")] = 1.0,
    y_min: Annotated[float, typer.Option("--y-min", help="Minimum beta coordinate.")] = -1.0,
    y_max: Annotated[float, typer.Option("--y-max", help="Maximum beta coordinate.")] = 1.0,
    seed_x: Annotated[int, typer.Option("--seed-x", help="RNG seed for direction 1.")] = 42,
    seed_y: Annotated[int, typer.Option("--seed-y", help="RNG seed for direction 2.")] = 43,
    ignore_bias_bn: Annotated[
        bool,
        typer.Option(
            "--ignore-bias-bn",
            help="Zero out perturbations for 1D bias/norm parameters.",
        ),
    ] = False,
    tracking_uri: Annotated[
        str, typer.Option("--tracking-uri", "-u", help="MLflow tracking URI.")
    ] = DEFAULT_TRACKING_URI,
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Force recomputation even if cached.")
    ] = False,
) -> None:
    """CLI to evaluate 2D filter-wise normalized loss landscape around a checkpoint."""
    Logger().setup()
    extract_filter_normalized_grid(
        run_id=run_id,
        step=step,
        resolution=resolution,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        seed_x=seed_x,
        seed_y=seed_y,
        ignore_bias_and_bn=ignore_bias_bn,
        tracking_uri=tracking_uri,
        force=force,
    )


if __name__ == "__main__":
    app()
