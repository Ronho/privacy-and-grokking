#!/usr/bin/env python3
"""extract_jump_planes.py.

Extracts local 2D subspace planes and evaluates 2D loss landscapes around optimization jumps/slingshots.
For each detected jump (theta_pre -> theta_post):
  - u1 = (theta_pre - theta_prev) / ||theta_pre - theta_prev|| (incoming trajectory direction)
  - u2 = Gram-Schmidt orthonormalized jump vector (theta_post - theta_pre)
  - Evaluates the 2D loss surface f(x, y) = L(theta_pre + x * u1 + y * u2)
  - Projects the local trajectory checkpoints onto the (u1, u2) basis

This quantitatively and visually proves the orthogonal (approx 90-degree) direction change
undergone by the model during Slingshot / Edge-of-Stability instabilities.

Outputs:
  - cache/runs/{run_id}/jump_planes/jumps_2d_planes.json
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
    jump_planes_grid_path,
    local_run_dir,
)
from privacy_and_grokking.config import TrainConfig
from privacy_and_grokking.metrics.interpolation_1d import compute_state_dict_l2_norm
from privacy_and_grokking.utils import Logger
from visualizations.preprocessing.extract_1d_interpolation import (
    get_available_checkpoints,
    get_checkpoint_weight,
    safe_collate,
)

app = typer.Typer(pretty_exceptions_enable=False)


def state_dict_to_tensor(sd: dict[str, torch.Tensor]) -> torch.Tensor:
    """Flattens all floating point parameters of a state dict into a single 1D tensor."""
    tensors = [v.flatten().float() for v in sd.values() if torch.is_floating_point(v)]
    return torch.cat(tensors)


def tensor_to_state_dict(
    vec: torch.Tensor, template_sd: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Unflattens a 1D tensor back into a model state dict structure."""
    out = {}
    offset = 0
    for k, v in template_sd.items():
        if torch.is_floating_point(v):
            numel = v.numel()
            out[k] = vec[offset : offset + numel].view_as(v)
            offset += numel
        else:
            out[k] = v.clone()
    return out


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


def extract_jump_planes(
    run_id: str,
    window_steps: int = 500,
    grid_resolution: int = 35,
    margin_ratio: float = 0.35,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    force: bool = False,
) -> dict:
    """Evaluates 2D local subspace trajectory and loss landscapes for all detected jumps."""
    out_file = jump_planes_grid_path(run_id)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if out_file.exists() and not force:
        try:
            data = json.loads(out_file.read_bytes())
            if (
                data.get("grid_resolution") == grid_resolution
                and data.get("window_steps") == window_steps
            ):
                print(f"Loading cached jump planes from {out_file}...")
                return data
        except Exception:
            pass

    all_ckpts = get_available_checkpoints(run_id=run_id, tracking_uri=tracking_uri)

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

    # 2. Detect jumps
    # Load all checkpoints around candidate jumps or detect via interpolation cache
    from visualizations.images.plot_1d_interpolation import detect_trajectory_jumps

    jumps = detect_trajectory_jumps(run_id=run_id, tracking_uri=tracking_uri)
    print(f"Extracting 2D jump planes for {len(jumps)} detected jumps in run '{run_id}'...")

    jumps_data = []

    for jump_idx, (s_pre, s_post) in enumerate(jumps):
        print(f"\n--- Processing Jump {jump_idx+1}: Step {s_pre:,} -> {s_post:,} ---")
        idx_pre = all_ckpts.index(s_pre)
        idx_post = all_ckpts.index(s_post)

        # Window checkpoints: from s_pre - window_steps to s_post + window_steps
        step_min = max(0, s_pre - window_steps)
        step_max = min(all_ckpts[-1], s_post + window_steps)

        window_ckpt_steps = [s for s in all_ckpts if step_min <= s <= step_max]

        # Weights
        sd_pre = get_checkpoint_weight(run_id, s_pre, tracking_uri)
        sd_post = get_checkpoint_weight(run_id, s_post, tracking_uri)

        # Find closest checkpoint to s_pre - window_steps
        s_prev = min(window_ckpt_steps, key=lambda s: abs(s - (s_pre - window_steps)))
        # Find closest checkpoint to s_post + window_steps
        s_next = min(window_ckpt_steps, key=lambda s: abs(s - (s_post + window_steps)))

        sd_prev = get_checkpoint_weight(run_id, s_prev, tracking_uri)
        sd_next = get_checkpoint_weight(run_id, s_next, tracking_uri)

        w_pre = state_dict_to_tensor(sd_pre)
        w_post = state_dict_to_tensor(sd_post)
        w_prev = state_dict_to_tensor(sd_prev)
        w_next = state_dict_to_tensor(sd_next)

        # Construct Basis (u1, u2):
        # u1: incoming direction (from prev to pre)
        v1 = w_pre - w_prev
        norm_v1 = float(torch.norm(v1).item())
        u1 = v1 / (norm_v1 + 1e-12)

        # u2: Gram-Schmidt orthogonalized jump direction (from pre to post)
        v2 = w_post - w_pre
        norm_v2 = float(torch.norm(v2).item())
        proj_21 = float(torch.dot(v2, u1).item())
        v2_perp = v2 - proj_21 * u1
        norm_v2_perp = float(torch.norm(v2_perp).item())
        u2 = v2_perp / (norm_v2_perp + 1e-12)

        # Angle statistics
        cos_in_jump = float(torch.dot(v1, v2).item() / (norm_v1 * norm_v2 + 1e-12))
        turn_angle_deg = float(np.rad2deg(np.arccos(np.clip(cos_in_jump, -1.0, 1.0))))

        v3 = w_next - w_post
        norm_v3 = float(torch.norm(v3).item())
        cos_in_out = float(torch.dot(v1, v3).item() / (norm_v1 * norm_v3 + 1e-12))
        out_angle_deg = float(np.rad2deg(np.arccos(np.clip(cos_in_out, -1.0, 1.0))))

        cos_theta = float(torch.dot(w_pre, w_post).item() / (torch.norm(w_pre).item() * torch.norm(w_post).item()))
        theta_rot_deg = float(np.rad2deg(np.arccos(np.clip(cos_theta, -1.0, 1.0))))

        # Project trajectory checkpoints in window
        traj_points = []
        for s in window_ckpt_steps:
            sd_s = get_checkpoint_weight(run_id, s, tracking_uri)
            w_s = state_dict_to_tensor(sd_s)
            disp = w_s - w_pre
            x_val = float(torch.dot(disp, u1).item())
            y_val = float(torch.dot(disp, u2).item())
            traj_points.append({
                "step": s,
                "x": x_val,
                "y": y_val,
                "is_pre": s == s_pre,
                "is_post": s == s_post,
            })

        # Determine 2D grid bounds
        xs = [p["x"] for p in traj_points]
        ys = [p["y"] for p in traj_points]
        x_span = max(xs) - min(xs) if max(xs) > min(xs) else 10.0
        y_span = max(ys) - min(ys) if max(ys) > min(ys) else 10.0

        x_min = min(xs) - margin_ratio * x_span
        x_max = max(xs) + margin_ratio * x_span
        y_min = min(ys) - margin_ratio * y_span
        y_max = max(ys) + margin_ratio * y_span

        grid_x = np.linspace(x_min, x_max, grid_resolution)
        grid_y = np.linspace(y_min, y_max, grid_resolution)

        print(f"Evaluating {grid_resolution}x{grid_resolution} loss grid on Jump {jump_idx+1} plane...")
        loss_grid = np.zeros((grid_resolution, grid_resolution), dtype=np.float32)
        norm_grid = np.zeros((grid_resolution, grid_resolution), dtype=np.float32)

        tr_x, tr_y = dataset_tensors["train"]

        for i, gy in enumerate(tqdm(grid_y, desc=f"Grid J{jump_idx+1}", leave=False)):
            for j, gx in enumerate(grid_x):
                w_pt = w_pre + gx * u1 + gy * u2
                sd_pt = tensor_to_state_dict(w_pt, sd_pre)
                model.load_state_dict(sd_pt)
                l_val, _ = evaluate_tensors(model, tr_x, tr_y, loss_fn)
                loss_grid[i, j] = l_val
                norm_grid[i, j] = compute_state_dict_l2_norm(sd_pt)

        jumps_data.append({
            "jump_idx": jump_idx,
            "step_pre": s_pre,
            "step_post": s_post,
            "step_prev": s_prev,
            "step_next": s_next,
            "jump_distance": norm_v2,
            "turn_angle_deg": turn_angle_deg,
            "out_angle_deg": out_angle_deg,
            "theta_rot_deg": theta_rot_deg,
            "traj_points": traj_points,
            "grid_x": grid_x.tolist(),
            "grid_y": grid_y.tolist(),
            "loss_grid": loss_grid.tolist(),
            "norm_grid": norm_grid.tolist(),
        })

    result_data = {
        "run_id": run_id,
        "grid_resolution": grid_resolution,
        "window_steps": window_steps,
        "jumps": jumps_data,
    }

    out_file.write_text(json.dumps(result_data, indent=2))
    print(f"\nSaved all 2D jump planes to: {out_file}")
    return result_data


@app.command()
def main(
    run_id: Annotated[str, typer.Option("--run-id", help="Target MLflow Run ID.")],
    window_steps: Annotated[
        int, typer.Option("--window-steps", help="Checkpoint window around each jump.")
    ] = 500,
    grid_resolution: Annotated[
        int, typer.Option("--grid-resolution", help="Resolution of the 2D loss grid per jump.")
    ] = 35,
    tracking_uri: Annotated[
        str, typer.Option("--tracking-uri", help="MLflow server URI.")
    ] = DEFAULT_TRACKING_URI,
    force: Annotated[
        bool, typer.Option("--force", help="Force recalculation even if cached.")
    ] = False,
) -> None:
    """CLI endpoint to extract 2D subspace loss landscape planes across trajectory jumps."""
    Logger().setup()
    extract_jump_planes(
        run_id=run_id,
        window_steps=window_steps,
        grid_resolution=grid_resolution,
        tracking_uri=tracking_uri,
        force=force,
    )


if __name__ == "__main__":
    app()
