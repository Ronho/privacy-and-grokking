#!/usr/bin/env python3
"""extract_1d_interpolation.py.

Evaluates 1D linear parameter interpolation profiles between checkpoints
along an optimization trajectory (Li et al., NeurIPS 2018; Goodfellow et al., ICLR 2015).

Linear parameter interpolation computes:
    theta(alpha) = (1 - alpha) * theta_A + alpha * theta_B   for alpha in [alpha_min, alpha_max]

Evaluating the model directly on this 1D segment in high-dimensional space yields exact,
zero-variance loss and accuracy profiles without high-dimensional projection artifacts.

Required Inputs:
  - Checkpoints from MLflow or local cache (`cache/runs/{run_id}/checkpoints/{step}/model.pth`)
  - Training configuration (`training_config.json`)

Generated Outputs:
  - cache/runs/{run_id}/interpolation/interpolation_{start_step}_{end_step}.json (pair mode)
  - cache/runs/{run_id}/interpolation/trajectory_interpolation.json (trajectory mode)
"""

from __future__ import annotations

import io
import json
import sys
import urllib.request
from pathlib import Path
from typing import Annotated

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
    interpolation_1d_pair_path,
    interpolation_1d_path,
    local_run_dir,
)
from privacy_and_grokking.config import TrainConfig
from privacy_and_grokking.metrics.interpolation_1d import (
    compute_state_dict_l2_norm,
    linear_interpolate_state_dicts,
)
from privacy_and_grokking.utils import Logger

app = typer.Typer(pretty_exceptions_enable=False)


def get_available_checkpoints(run_id: str, tracking_uri: str) -> list[int]:
    """Retrieve available checkpoint step numbers for a run from cache or MLflow REST API."""
    checkpoints_dir = local_run_dir(run_id) / "checkpoints"
    info_file = checkpoints_dir / "info.json"
    if info_file.exists():
        try:
            data = json.loads(info_file.read_text())
            if "steps" in data and isinstance(data["steps"], list) and len(data["steps"]) > 0:
                return sorted(set(int(s) for s in data["steps"]))
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
            info_file.parent.mkdir(parents=True, exist_ok=True)
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
        info_file.parent.mkdir(parents=True, exist_ok=True)
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
def evaluate_tensors(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_fn: torch.nn.Module,
    batch_size: int = 2000,
) -> tuple[float, float]:
    """Evaluate loss and classification accuracy on pre-batched dataset tensors."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    n = len(y)
    for i in range(0, n, batch_size):
        bx = x[i : i + batch_size]
        by = y[i : i + batch_size]
        logits = model(bx)
        loss = loss_fn(logits, by)

        total_loss += float(loss.item()) * len(by)
        if logits.ndim > 1 and logits.size(-1) > 1 and by.ndim == 1:
            preds = logits.argmax(dim=-1)
            total_correct += int((preds == by).sum().item())
        elif logits.ndim == 1 or logits.size(-1) == 1:
            preds = (logits.squeeze() > 0.0).long()
            total_correct += int((preds == by).sum().item())

    avg_loss = total_loss / max(1, n)
    acc = total_correct / max(1, n)
    return avg_loss, acc


def extract_1d_pair_interpolation(
    run_id: str,
    start_step: int | None = None,
    end_step: int | None = None,
    alpha_min: float = -0.5,
    alpha_max: float = 1.5,
    num_points: int = 100,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    force: bool = False,
) -> dict:
    """Evaluates 1D linear parameter interpolation between two specific checkpoints.

    theta(alpha) = (1 - alpha) * theta_{start} + alpha * theta_{end}   for alpha in [alpha_min, alpha_max]
    """
    available_steps = get_available_checkpoints(run_id=run_id, tracking_uri=tracking_uri)
    s_start = start_step if start_step is not None else available_steps[0]
    s_end = end_step if end_step is not None else available_steps[-1]

    out_file = interpolation_1d_pair_path(run_id, s_start, s_end)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if out_file.exists() and not force:
        try:
            data = json.loads(out_file.read_bytes())
            if (
                data.get("alpha_min") == alpha_min
                and data.get("alpha_max") == alpha_max
                and data.get("num_points") == num_points
            ):
                print(f"Loading cached 1D pair interpolation from {out_file}...")
                return data
        except Exception:
            pass

    print(
        f"Extracting 1D linear interpolation for Run ID '{run_id}' between "
        f"Step {s_start} and Step {s_end} ({num_points} points in [{alpha_min}, {alpha_max}])..."
    )

    # 1. Setup Model and Dataset
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

    # 2. Load Checkpoint Weights
    state_a = get_checkpoint_weight(run_id=run_id, step=s_start, tracking_uri=tracking_uri)
    state_b = get_checkpoint_weight(run_id=run_id, step=s_end, tracking_uri=tracking_uri)

    alphas = np.linspace(alpha_min, alpha_max, num_points)
    samples = []

    for alpha in tqdm(alphas, desc=f"1D Interp ({s_start} -> {s_end})"):
        interp_state = linear_interpolate_state_dicts(state_a, state_b, float(alpha))
        model.load_state_dict(interp_state)

        tr_x, tr_y = dataset_tensors["train"]
        tr_loss, tr_acc = evaluate_tensors(model, tr_x, tr_y, loss_fn)

        te_x, te_y = dataset_tensors["test"]
        te_loss, te_acc = evaluate_tensors(model, te_x, te_y, loss_fn)

        weight_norm = compute_state_dict_l2_norm(interp_state)

        samples.append({
            "alpha": float(alpha),
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "test_loss": te_loss,
            "test_acc": te_acc,
            "weight_norm": weight_norm,
        })

    result_data = {
        "run_id": run_id,
        "start_step": s_start,
        "end_step": s_end,
        "alpha_min": alpha_min,
        "alpha_max": alpha_max,
        "num_points": num_points,
        "samples": samples,
    }

    out_file.write_text(json.dumps(result_data, indent=2))
    print(f"Saved 1D pair interpolation results to {out_file}")
    return result_data


def extract_1d_interpolation(
    run_id: str,
    points_per_segment: int = 10,
    stride: int = 1,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    force: bool = False,
) -> dict:
    """Evaluates 1D linear parameter interpolation along the entire checkpoint trajectory."""
    out_file = interpolation_1d_path(run_id)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if out_file.exists() and not force:
        try:
            data = json.loads(out_file.read_bytes())
            if (
                data.get("points_per_segment") == points_per_segment
                and data.get("stride") == stride
            ):
                print(f"Loading cached 1D interpolation from {out_file}...")
                return data
        except Exception:
            pass

    available_steps = get_available_checkpoints(run_id=run_id, tracking_uri=tracking_uri)
    if stride > 1:
        selected_steps = available_steps[::stride]
        if available_steps[-1] not in selected_steps:
            selected_steps.append(available_steps[-1])
        steps = sorted(set(selected_steps))
    else:
        steps = available_steps

    if len(steps) < 2:
        raise ValueError(f"Need at least 2 checkpoints to interpolate, found {len(steps)}.")

    print(
        f"Extracting 1D linear interpolation for Run ID '{run_id}' across {len(steps)} checkpoints "
        f"({len(steps)-1} segments, {points_per_segment} pts/segment)..."
    )

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

    samples = []
    checkpoint_evals = {}

    total_segments = len(steps) - 1
    pbar = tqdm(total=total_segments * points_per_segment + 1, desc="1D Interpolation")

    for seg_idx in range(total_segments):
        s_start = steps[seg_idx]
        s_end = steps[seg_idx + 1]

        state_a = get_checkpoint_weight(run_id=run_id, step=s_start, tracking_uri=tracking_uri)
        state_b = get_checkpoint_weight(run_id=run_id, step=s_end, tracking_uri=tracking_uri)

        is_last = seg_idx == total_segments - 1
        alphas = np.linspace(0.0, 1.0, points_per_segment, endpoint=is_last)

        for alpha in alphas:
            interp_state = linear_interpolate_state_dicts(state_a, state_b, float(alpha))
            model.load_state_dict(interp_state)

            tr_x, tr_y = dataset_tensors["train"]
            tr_loss, tr_acc = evaluate_tensors(model, tr_x, tr_y, loss_fn)

            te_x, te_y = dataset_tensors["test"]
            te_loss, te_acc = evaluate_tensors(model, te_x, te_y, loss_fn)

            weight_norm = compute_state_dict_l2_norm(interp_state)
            step_effective = float((1.0 - alpha) * s_start + alpha * s_end)

            sample = {
                "step": step_effective,
                "segment_idx": seg_idx,
                "alpha": float(alpha),
                "step_from": s_start,
                "step_to": s_end,
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "test_loss": te_loss,
                "test_acc": te_acc,
                "weight_norm": weight_norm,
            }
            samples.append(sample)

            if alpha == 0.0:
                checkpoint_evals[str(s_start)] = sample
            if is_last and alpha == 1.0:
                checkpoint_evals[str(s_end)] = sample

            pbar.update(1)

    pbar.close()

    result_data = {
        "run_id": run_id,
        "checkpoints": steps,
        "points_per_segment": points_per_segment,
        "stride": stride,
        "samples": samples,
        "checkpoint_evals": checkpoint_evals,
    }

    out_file.write_text(json.dumps(result_data, indent=2))
    print(f"Saved 1D interpolation results to {out_file} ({len(samples)} evaluated points)")
    return result_data


@app.command()
def main(
    run_id: Annotated[str, typer.Option("--run-id", help="Target MLflow Run ID.")],
    start_step: Annotated[
        int | None, typer.Option("--start-step", help="Initial checkpoint step.")
    ] = None,
    end_step: Annotated[
        int | None, typer.Option("--end-step", help="Final checkpoint step.")
    ] = None,
    alpha_min: Annotated[
        float, typer.Option("--alpha-min", help="Minimum alpha for pair interpolation.")
    ] = -0.5,
    alpha_max: Annotated[
        float, typer.Option("--alpha-max", help="Maximum alpha for pair interpolation.")
    ] = 1.5,
    num_points: Annotated[
        int, typer.Option("--num-points", help="Number of alpha evaluation points.")
    ] = 100,
    trajectory: Annotated[
        bool, typer.Option("--trajectory", help="Interpolate full multi-segment trajectory.")
    ] = False,
    points_per_segment: Annotated[
        int, typer.Option("--points-per-segment", help="Number of alpha points per segment.")
    ] = 10,
    stride: Annotated[
        int, typer.Option("--stride", help="Checkpoint sampling stride.")
    ] = 1,
    tracking_uri: Annotated[
        str, typer.Option("--tracking-uri", help="MLflow server URI.")
    ] = DEFAULT_TRACKING_URI,
    force: Annotated[
        bool, typer.Option("--force", help="Force recalculation even if cached.")
    ] = False,
) -> None:
    """CLI endpoint to extract 1D linear parameter interpolation."""
    Logger().setup()
    if trajectory:
        extract_1d_interpolation(
            run_id=run_id,
            points_per_segment=points_per_segment,
            stride=stride,
            tracking_uri=tracking_uri,
            force=force,
        )
    else:
        extract_1d_pair_interpolation(
            run_id=run_id,
            start_step=start_step,
            end_step=end_step,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            num_points=num_points,
            tracking_uri=tracking_uri,
            force=force,
        )


if __name__ == "__main__":
    app()
