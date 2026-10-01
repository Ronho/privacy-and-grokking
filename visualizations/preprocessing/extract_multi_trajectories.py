#!/usr/bin/env python3
"""extract_multi_trajectories.py.

Extracts optimization trajectories across multiple checkpoint training runs and
computes a shared 2D loss and weight-norm landscape grid along the top joint
principal components.

Joint PCA captures the shared subspace explored by different initializations or
model seeds, allowing multiple trajectories to be overlaid and compared on the
same underlying loss landscape surface.

Required Inputs:
  - Checkpoint model weights (`checkpoints/{step}/model.pth`) from the local cache
    (`cache/runs/{run_id}/...`) or downloaded from the MLflow tracking server.
  - Training configuration (`training_config.json`) for data container and model architecture.

Generated Outputs:
  - cache/joint_trajectories/{group_id}/projection.json:
      Joint PCA projected coordinates for all runs, explained variance ratios, components,
      origin vector, and per-run trajectory metadata (init_scale, model_index).
  - cache/joint_trajectories/{group_id}/grid.json:
      Shared 2D loss and weight-norm landscape grid evaluated across the joint subspace.
"""

from __future__ import annotations

import hashlib
import io
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
import typer
from mlflow import MlflowClient
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, default_collate
from tqdm import tqdm

from privacy_and_grokking.artifacts.cache import (
    DEFAULT_TRACKING_URI,
    get_training_config,
)
from privacy_and_grokking.artifacts.local import (
    joint_trajectories_dir,
    joint_trajectory_grid_path,
    joint_trajectory_projection_path,
    local_run_dir,
)
from privacy_and_grokking.config import TrainConfig
from privacy_and_grokking.metrics.checkpoint_distance import (
    fetch_run_init_scale,
    fetch_run_model_index,
    fetch_run_name,
)
from privacy_and_grokking.utils import Logger

app = typer.Typer(pretty_exceptions_enable=False)


def generate_group_id(run_ids: list[str], prefix: str = "joint") -> str:
    """Generate a deterministic group identifier from a list of run IDs."""
    sorted_ids = sorted(run_ids)
    raw = "_".join(sorted_ids).encode("utf-8")
    short_hash = hashlib.md5(raw).hexdigest()[:10]
    return f"{prefix}_{len(run_ids)}runs_{short_hash}"


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
        print(f"Warning: Failed to fetch checkpoints via REST for '{run_id}' ({e}). Fallback to local disk...")

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


def collect_run_checkpoints(
    run_id: str,
    max_workers: int,
    tracking_uri: str,
) -> dict[int, dict[str, torch.Tensor]]:
    """Parallel download and loading of all checkpoint weights for a run."""
    steps = get_available_checkpoints(run_id=run_id, tracking_uri=tracking_uri)
    checkpoints: dict[int, dict[str, torch.Tensor]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(get_checkpoint_weight, run_id, step, tracking_uri): step
            for step in steps
        }
        for future in tqdm(as_completed(futures), total=len(steps), desc=f"Loading {run_id[:8]}"):
            step = futures[future]
            checkpoints[step] = future.result()

    return checkpoints


def flatten_weights(weights: dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten all floating-point parameter tensors into a single 1D tensor."""
    tensors = []
    for key in sorted(weights.keys()):
        val = weights[key]
        if isinstance(val, torch.Tensor) and val.is_floating_point():
            tensors.append(val.detach().cpu().reshape(-1).float())
    return torch.cat(tensors)


def unflatten_weights(
    flat_weights: torch.Tensor | np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Reconstruct a state dict from a flat parameter vector."""
    if isinstance(flat_weights, np.ndarray):
        flat_weights = torch.from_numpy(flat_weights)
    flat_weights = flat_weights.to(device=device, dtype=torch.float32)

    state_dict = model.state_dict()
    new_state_dict = {}
    offset = 0
    for key in sorted(state_dict.keys()):
        val = state_dict[key]
        if isinstance(val, torch.Tensor) and val.is_floating_point():
            numel = val.numel()
            param_tensor = flat_weights[offset : offset + numel].reshape(val.shape)
            new_state_dict[key] = param_tensor
            offset += numel
        else:
            new_state_dict[key] = val.to(device)
    return new_state_dict


def compute_joint_pca(
    all_checkpoints: dict[str, dict[int, dict[str, torch.Tensor]]],
    n_components: int,
    group_id: str,
    origin_mode: str = "mean_final",
    tracking_uri: str = DEFAULT_TRACKING_URI,
    force: bool = False,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Compute Joint PCA on multiple checkpoint trajectories.

    Args:
        all_checkpoints: Mapping from run_id to step-to-state_dict dictionary.
        n_components: Number of principal components to extract (>= 2).
        group_id: Unique group identifier.
        origin_mode: 'mean_final' (centroid of final minimizers),
                     'first_final' (final checkpoint of first run),
                     or 'global_mean' (mean of all checkpoints across all runs).
        tracking_uri: MLflow tracking server URI.
        force: If True, recompute even if cached.

    Returns:
        (run_coords_dict, explained_variance_ratio, components, origin)
    """
    projection_file = joint_trajectory_projection_path(group_id)
    projection_file.parent.mkdir(parents=True, exist_ok=True)

    if projection_file.exists() and not force:
        try:
            data = json.loads(projection_file.read_bytes())
            if (
                data.get("n_components", 0) >= n_components
                and data.get("origin_mode", "mean_final") == origin_mode
            ):
                print(f"Loading cached joint PCA projection from {projection_file}...")
                run_coords = {}
                for r_id, r_info in data["runs"].items():
                    run_coords[r_id] = np.array(r_info["pca_coords"])[:, :n_components]
                evr = np.array(data["explained_variance_ratio"])[:n_components]
                components = np.array(data["components"])[:n_components]
                origin = np.array(data["origin"])
                return run_coords, evr, components, origin
        except Exception:
            pass

    run_ids = list(all_checkpoints.keys())
    flat_trajectories: dict[str, tuple[list[int], torch.Tensor]] = {}

    for r in run_ids:
        ckpts = all_checkpoints[r]
        sorted_steps = sorted(ckpts.keys())
        w_list = [flatten_weights(ckpts[s]) for s in sorted_steps]
        flat_trajectories[r] = (sorted_steps, torch.stack(w_list))

    # Determine origin vector theta_ref
    if origin_mode == "mean_final":
        final_vectors = [flat_trajectories[r][1][-1].numpy() for r in run_ids]
        origin = np.mean(final_vectors, axis=0).astype(np.float64)
    elif origin_mode == "first_final":
        origin = flat_trajectories[run_ids[0]][1][-1].numpy().astype(np.float64)
    else:  # global_mean
        all_vecs = [flat_trajectories[r][1].numpy() for r in run_ids]
        origin = np.mean(np.concatenate(all_vecs, axis=0), axis=0).astype(np.float64)

    # Form joint displacement matrix
    displacement_blocks = []
    for r in run_ids:
        w_np = flat_trajectories[r][1].numpy().astype(np.float64)
        displacement_blocks.append(w_np - origin)

    joint_displacements = np.concatenate(displacement_blocks, axis=0)
    print(f"Joint displacement matrix shape: {joint_displacements.shape}")

    k = min(n_components, joint_displacements.shape[0], joint_displacements.shape[1])
    u, s, vt = np.linalg.svd(joint_displacements, full_matrices=False)
    components = vt[:k]  # (k, D)
    total_var = float(np.sum(s**2))
    evr = (s[:k] ** 2) / total_var if total_var > 0 else np.zeros(k)

    # Project individual runs onto joint (v1, v2)
    run_coords: dict[str, np.ndarray] = {}
    runs_payload: dict[str, dict] = {}

    for r in run_ids:
        steps, _ = flat_trajectories[r]
        disp = flat_trajectories[r][1].numpy().astype(np.float64) - origin
        coords = disp @ components.T  # (N_steps, k)
        run_coords[r] = coords

        runs_payload[r] = {
            "run_id": r,
            "run_name": fetch_run_name(r, tracking_uri=tracking_uri),
            "init_scale": fetch_run_init_scale(r, tracking_uri=tracking_uri),
            "model_index": fetch_run_model_index(r, tracking_uri=tracking_uri),
            "steps": steps,
            "pca_coords": coords.tolist(),
        }

    payload = {
        "group_id": group_id,
        "origin_mode": origin_mode,
        "n_components": int(k),
        "explained_variance_ratio": evr.tolist(),
        "components": components.tolist(),
        "origin": origin.tolist(),
        "runs": runs_payload,
    }

    projection_file.write_text(json.dumps(payload))
    print(f"Saved Joint PCA projection to {projection_file}")
    return run_coords, evr, components, origin


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
    """Evaluate loss on pre-batched tensors."""
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


def make_symlog_grid(
    vmin: float,
    vmax: float,
    n_points: int = 50,
    linthresh: float = 0.05,
) -> np.ndarray:
    """Generate symmetric log-spaced coordinates that densely cover the linear zero-region and geometric tails."""
    parts = []
    # Negative tail (geometric spacing from -abs(vmin) to -linthresh)
    if vmin < -linthresh:
        neg_n = max(n_points // 3, 10)
        parts.append(-np.geomspace(abs(vmin), linthresh, neg_n))

    # Linear central region around zero
    center_min = max(vmin, -linthresh)
    center_max = min(vmax, linthresh)
    center_n = max(n_points // 3, 12)
    parts.append(np.linspace(center_min, center_max, center_n))

    # Positive tail (geometric spacing from linthresh to vmax)
    if vmax > linthresh:
        pos_n = max(n_points // 3, 10)
        parts.append(np.geomspace(linthresh, vmax, pos_n))

    combined = np.unique(np.concatenate(parts))
    return np.sort(combined)


def joint_grid_search(
    run_coords: dict[str, np.ndarray],
    components: np.ndarray,
    origin: np.ndarray,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    dataset_tensors: dict[str, tuple[torch.Tensor, torch.Tensor]],
    group_id: str,
    resolution: int = 40,
    margin: float = 0.15,
    grid_spacing: str = "symlog",
    linthresh: float = 0.05,
    device: torch.device | None = None,
    force: bool = False,
) -> dict:
    """Evaluate loss and weight norm on a 2D grid spanning all projected trajectories."""
    if device is None:
        device = torch.device("cpu")

    grid_file = joint_trajectory_grid_path(group_id)
    grid_file.parent.mkdir(parents=True, exist_ok=True)

    if grid_file.exists() and not force:
        try:
            data = json.loads(grid_file.read_bytes())
            if (
                data.get("resolution") == resolution
                and data.get("margin") == margin
                and data.get("grid_spacing", "linear") == grid_spacing
            ):
                print(f"Loading cached joint grid from {grid_file}...")
                return data
        except Exception:
            pass

    all_coords = np.concatenate(list(run_coords.values()), axis=0)
    x = all_coords[:, 0]
    y = all_coords[:, 1]

    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())

    x_span = (x_max - x_min) if x_max > x_min else 1.0
    y_span = (y_max - y_min) if y_max > y_min else 1.0

    x_min_grid = x_min - margin * x_span
    x_max_grid = x_max + margin * x_span
    y_min_grid = y_min - margin * y_span
    y_max_grid = y_max + margin * y_span

    if grid_spacing == "symlog":
        x_coords = make_symlog_grid(x_min_grid, x_max_grid, resolution, linthresh=linthresh)
        y_coords = make_symlog_grid(y_min_grid, y_max_grid, resolution, linthresh=linthresh)
    else:
        x_coords = np.linspace(x_min_grid, x_max_grid, resolution)
        y_coords = np.linspace(y_min_grid, y_max_grid, resolution)

    xx, yy = np.meshgrid(x_coords, y_coords)
    grid_points = np.stack([xx.ravel(), yy.ravel()], axis=1)

    v0 = components[0]
    v1 = components[1]

    print(
        f"Evaluating joint landscape on {len(grid_points)} grid points ({len(y_coords)}x{len(x_coords)}, spacing='{grid_spacing}')..."
    )
    weight_norms: list[float] = []
    losses: dict[str, list[float]] = {name: [] for name in dataset_tensors}

    for pt in tqdm(grid_points, desc="Evaluating joint grid"):
        gx, gy = pt[0], pt[1]
        w = origin + gx * v0 + gy * v1
        norm = float(np.linalg.norm(w))
        weight_norms.append(norm)

        state = unflatten_weights(w, model, device)
        model.load_state_dict(state)

        for name, (dx, dy) in dataset_tensors.items():
            loss = evaluate_loss_tensors(model, dx, dy, loss_fn, batch_size=2000)
            losses[name].append(loss)

    grid_payload = {
        "group_id": group_id,
        "resolution": resolution,
        "margin": margin,
        "grid_spacing": grid_spacing,
        "linthresh": linthresh,
        "bounds": {
            "pc1": {"min": x_min, "max": x_max},
            "pc2": {"min": y_min, "max": y_max},
        },
        "grid_bounds": {
            "pc1": {"min": float(x_min_grid), "max": float(x_max_grid)},
            "pc2": {"min": float(y_min_grid), "max": float(y_max_grid)},
        },
        "x_coords": x_coords.tolist(),
        "y_coords": y_coords.tolist(),
        "grid_points": grid_points.tolist(),
        "weight_norms": weight_norms,
        "losses": losses,
    }

    grid_file.write_text(json.dumps(grid_payload))
    print(f"Saved joint grid to {grid_file} ({len(y_coords)}x{len(x_coords)} = {len(grid_points)} points, spacing='{grid_spacing}')")
    return grid_payload


@app.command()
def main(
    run_ids: Annotated[
        list[str],
        typer.Option(
            "--run-ids",
            "-r",
            help="List of MLflow run IDs to include in the joint trajectory landscape.",
        ),
    ],
    group_id: Annotated[
        str | None,
        typer.Option("--group-id", "-g", help="Custom group identifier for cached artifacts."),
    ] = None,
    tracking_uri: Annotated[
        str, typer.Option("--tracking-uri", "-u", help="MLflow tracking URI.")
    ] = DEFAULT_TRACKING_URI,
    n_components: Annotated[
        int, typer.Option("--n-components", "-k", help="Number of PCA components.")
    ] = 10,
    grid_resolution: Annotated[
        int, typer.Option("--grid-resolution", help="Points per grid axis.")
    ] = 50,
    grid_margin: Annotated[
        float, typer.Option("--grid-margin", help="Fractional margin beyond coordinates.")
    ] = 0.15,
    grid_spacing: Annotated[
        str,
        typer.Option(
            "--grid-spacing",
            help="Grid coordinate spacing mode: 'symlog' (denser near 0 and log tails) or 'linear'.",
        ),
    ] = "symlog",
    linthresh: Annotated[
        float,
        typer.Option("--linthresh", help="Linear threshold near 0 for symlog grid."),
    ] = 0.05,
    origin_mode: Annotated[
        str,
        typer.Option(
            "--origin-mode", "-o", help="Center origin mode: 'mean_final', 'first_final', or 'global_mean'."
        ),
    ] = "mean_final",
    max_workers: Annotated[
        int, typer.Option("--max-workers", "-w", help="Concurrent worker threads.")
    ] = 10,
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Force recomputation even if cached.")
    ] = False,
) -> None:
    """CLI entrypoint for extracting joint multi-run trajectories and loss landscape."""
    Logger().setup()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    target_runs: list[str] = []
    for item in run_ids:
        target_runs.extend([x.strip() for x in item.split(",") if x.strip()])

    if len(target_runs) < 2:
        raise typer.BadParameter("Please provide at least 2 run IDs for joint trajectory analysis.")

    gid = group_id or generate_group_id(target_runs)
    print(f"Processing {len(target_runs)} runs for group '{gid}'...")

    # 1. Collect checkpoints for all runs
    all_ckpts: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
    for r in target_runs:
        all_ckpts[r] = collect_run_checkpoints(
            run_id=r, max_workers=max_workers, tracking_uri=tracking_uri
        )

    # 2. Joint PCA
    run_coords, evr, components, origin = compute_joint_pca(
        all_checkpoints=all_ckpts,
        n_components=n_components,
        group_id=gid,
        origin_mode=origin_mode,
        tracking_uri=tracking_uri,
        force=force,
    )

    # 3. Build model architecture & evaluation dataset from reference run config
    ref_run_id = target_runs[0]
    config_dict = get_training_config(ref_run_id, tracking_uri=tracking_uri)
    train_cfg = TrainConfig.model_validate(config_dict)

    data_container = train_cfg.data()
    train_loader = DataLoader(
        data_container.train,
        batch_size=2000,
        shuffle=False,
        num_workers=0,
        collate_fn=safe_collate,
    )
    test_loader = DataLoader(
        data_container.test,
        batch_size=2000,
        shuffle=False,
        num_workers=0,
        collate_fn=safe_collate,
    )

    def load_tensors(loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
        xs, ys = [], []
        for bx, by in loader:
            xs.append(bx)
            ys.append(by)
        return torch.cat(xs, dim=0).to(device), torch.cat(ys, dim=0).to(device)

    dataset_tensors = {
        "train": load_tensors(train_loader),
        "test": load_tensors(test_loader),
    }

    model = train_cfg.model(
        input_dim=data_container.input_shape,
        num_classes=data_container.num_classes,
    ).to(device)
    loss_fn = train_cfg.loss(num_classes=data_container.num_classes)

    # 4. Evaluate joint 2D grid
    joint_grid_search(
        run_coords=run_coords,
        components=components,
        origin=origin,
        model=model,
        loss_fn=loss_fn,
        dataset_tensors=dataset_tensors,
        group_id=gid,
        resolution=grid_resolution,
        margin=grid_margin,
        grid_spacing=grid_spacing,
        linthresh=linthresh,
        device=device,
        force=force,
    )

    print(f"Joint multi-trajectory extraction complete for group '{gid}'.")


if __name__ == "__main__":
    app()
