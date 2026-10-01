#!/usr/bin/env python3
"""extract_trajectories.py.

Extracts optimization trajectories across checkpoint models and computes a 2D loss
and weight-norm landscape grid along the top principal components.

Supports centering the projection plane at the final converged checkpoint (recommended
by Li et al., 2018 to ensure the origin corresponds exactly to the final model) or at
the trajectory mean.

Required Inputs:
  - Checkpoint model weights (`checkpoints/{step}/model.pth`) from the local cache
    (`cache/runs/{run_id}/...`) or downloaded from the MLflow tracking server.
  - Training configuration (`training_config.json`) for data container and model architecture.

Generated Outputs:
  - cache/runs/{run_id}/trajectories/projection.json:
      PCA projected coordinates, variance ratios, components, and exact checkpoint losses.
  - cache/runs/{run_id}/trajectories/grid.json:
      2D loss landscape grid evaluated across the projection subspace.
"""

from __future__ import annotations

import io
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    local_run_dir,
    trajectory_grid_path,
    trajectory_projection_path,
)
from privacy_and_grokking.config import TrainConfig
from privacy_and_grokking.utils import Logger

app = typer.Typer(pretty_exceptions_enable=False)


def fetch_run(run_id: str, tracking_uri: str) -> tuple[str, str]:
    """Fetch run metadata from MLflow tracking server."""
    client = MlflowClient(tracking_uri)
    try:
        run = client.get_run(run_id)
        run_name = run.data.tags.get("mlflow.runName", run.info.run_name or run_id)
        experiment_id = run.info.experiment_id
        return run_name, experiment_id
    except Exception as e:
        raise RuntimeError(f"Could not fetch run '{run_id}' from MLflow ({e}).") from e


def get_available_checkpoints(run_id: str, tracking_uri: str) -> list[int]:
    """Retrieve sorted list of available checkpoint step integers for a run.

    Caches results in cache/runs/{run_id}/checkpoints/info.json.
    """
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

    # Fallback to local files if offline
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
        for future in tqdm(as_completed(futures), total=len(steps), desc="Loading checkpoints"):
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


def compute_pca(
    checkpoints: dict[int, dict[str, torch.Tensor]],
    n_components: int,
    identifier: str,
    center_mode: str = "final",
    force: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute PCA on checkpoint trajectories.

    Args:
        checkpoints: Mapping from step number to parameter state dict.
        n_components: Number of principal components to extract.
        identifier: MLflow run ID.
        center_mode: "final" to center at final checkpoint (Li et al., 2018 Sec. 7.2)
                     or "mean" for conventional PCA around arithmetic mean.
        force: If True, recomputes even if cached projection exists.

    Returns:
        (coords, explained_variance_ratio, components, origin)
    """
    projection_file = trajectory_projection_path(identifier)
    projection_file.parent.mkdir(parents=True, exist_ok=True)

    if projection_file.exists() and not force:
        try:
            data = json.loads(projection_file.read_bytes())
            if (
                data.get("n_components", 0) >= n_components
                and data.get("center_mode", "mean") == center_mode
            ):
                print(f"Loading cached PCA projection ({center_mode=}) from {projection_file}...")
                coords_list = [pt[:n_components] for pt in data["trajectories"]["pca_coords"]]
                coords = np.array(coords_list)
                evr = np.array(data["explained_variance_ratio"])[:n_components]
                components = np.array(data["components"])[:n_components]
                origin = np.array(data["mean"])
                return coords, evr, components, origin
        except Exception:
            pass

    sorted_steps = sorted(checkpoints.keys())
    flat_weights_list = [flatten_weights(checkpoints[step]) for step in sorted_steps]
    weight_matrix = torch.stack(flat_weights_list)  # (N, D)
    print(f"Weight matrix shape: {weight_matrix.shape} (checkpoints x parameters)")

    k = min(n_components, weight_matrix.shape[0], weight_matrix.shape[1])
    w_np = weight_matrix.numpy().astype(np.float64)

    if center_mode == "final":
        # Li et al. (2018) Section 7.2:
        # Displacement matrix relative to final minimizer theta_n:
        # At final step, M[n-1] == 0, so final coordinates are exactly (0, 0).
        final_weights = w_np[-1]
        origin = final_weights
        displacement_matrix = w_np - origin

        # SVD on displacements without subtracting mean
        u, s, vt = np.linalg.svd(displacement_matrix, full_matrices=False)
        components = vt[:k]  # (k, D)
        coords = displacement_matrix @ components.T  # (N, k)

        total_var = float(np.sum(s**2))
        evr = (s[:k] ** 2) / total_var if total_var > 0 else np.zeros(k)
        origin_step = sorted_steps[-1]
    else:
        # Standard PCA centered at arithmetic mean
        pca = PCA(n_components=k)
        coords = pca.fit_transform(w_np)
        components = pca.components_
        origin = pca.mean_
        evr = pca.explained_variance_ratio_
        origin_step = None

    trajectories: dict[str, list] = {"steps": [], "pca_coords": []}
    for step, coord in zip(sorted_steps, coords, strict=True):
        trajectories["steps"].append(step)
        trajectories["pca_coords"].append(
            coord.tolist() if hasattr(coord, "tolist") else list(coord)
        )

    payload = {
        "trajectory_id": identifier,
        "center_mode": center_mode,
        "origin_step": origin_step,
        "n_components": int(k),
        "explained_variance_ratio": evr.tolist(),
        "components": components.tolist(),
        "mean": origin.tolist(),  # Kept as "mean" for backward compatibility
        "trajectories": trajectories,
    }
    projection_file.write_text(json.dumps(payload))
    print(f"Saved PCA projection ({center_mode=}) to {projection_file}")

    return coords, evr, components, origin


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


def grid_search(
    coords: np.ndarray,
    components: np.ndarray,
    origin: np.ndarray,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    dataset_tensors: dict[str, tuple[torch.Tensor, torch.Tensor]],
    identifier: str,
    resolution: int = 40,
    margin: float = 0.1,
    device: torch.device | None = None,
    force: bool = False,
) -> dict:
    """Evaluate loss and weight norm on a 2D grid spanned by the first 2 principal components.

    Reconstruction formula: w(x, y) = origin + x * v0 + y * v1
    When centered at the final checkpoint, (0, 0) corresponds exactly to the final model.
    """
    if device is None:
        device = torch.device("cpu")

    grid_file = trajectory_grid_path(identifier)
    grid_file.parent.mkdir(parents=True, exist_ok=True)

    if grid_file.exists() and not force:
        try:
            data = json.loads(grid_file.read_bytes())
            if data.get("resolution") == resolution and data.get("margin") == margin:
                print(f"Loading cached grid with losses from {grid_file}...")
                return data
        except Exception:
            pass

    x = coords[:, 0]
    y = coords[:, 1]

    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())

    x_span = (x_max - x_min) if x_max > x_min else 1.0
    y_span = (y_max - y_min) if y_max > y_min else 1.0

    x_min_grid = x_min - margin * x_span
    x_max_grid = x_max + margin * x_span
    y_min_grid = y_min - margin * y_span
    y_max_grid = y_max + margin * y_span

    x_coords = np.linspace(x_min_grid, x_max_grid, resolution)
    y_coords = np.linspace(y_min_grid, y_max_grid, resolution)
    xx, yy = np.meshgrid(x_coords, y_coords)
    grid_points = np.stack([xx.ravel(), yy.ravel()], axis=1)

    v0 = components[0]
    v1 = components[1]

    print(f"Evaluating on {len(grid_points)} grid points ({resolution}x{resolution})...")
    weight_norms: list[float] = []
    losses: dict[str, list[float]] = {name: [] for name in dataset_tensors}

    for pt in tqdm(grid_points, desc="Grid evaluation"):
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
        "trajectory_id": identifier,
        "resolution": resolution,
        "margin": margin,
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
    print(f"Saved grid to {grid_file} ({resolution}x{resolution} = {len(grid_points)} points)")
    return grid_payload


@app.command()
def main(
    run_id: Annotated[
        str, typer.Option("--run-id", "-r", help="MLflow run ID to extract trajectories for.")
    ],
    tracking_uri: Annotated[
        str, typer.Option("--tracking-uri", "-u", help="MLflow tracking URI.")
    ] = DEFAULT_TRACKING_URI,
    n_components: Annotated[
        int, typer.Option("--n-components", "-k", help="Number of PCA components.")
    ] = 10,
    grid_resolution: Annotated[
        int, typer.Option("--grid-resolution", help="Points per grid axis.")
    ] = 40,
    grid_margin: Annotated[
        float, typer.Option("--grid-margin", help="Fractional margin beyond coordinates.")
    ] = 0.1,
    center_mode: Annotated[
        str,
        typer.Option(
            "--center-mode", "-c", help="PCA centering origin: 'final' (Li et al., 2018) or 'mean'."
        ),
    ] = "final",
    max_workers: Annotated[
        int, typer.Option("--max-workers", "-w", help="Concurrent worker threads.")
    ] = 10,
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Force recomputation even if cached.")
    ] = False,
) -> None:
    """Extract checkpoint trajectory and compute 2D loss landscape grid along top PCA directions."""
    Logger().setup()
    logger = Logger.get()
    logger.info(f"Extracting trajectories for Run ID: {run_id} (center_mode={center_mode})")

    training_config: TrainConfig = get_training_config(run_id=run_id, tracking_uri=tracking_uri)
    checkpoints = collect_run_checkpoints(
        run_id=run_id, max_workers=max_workers, tracking_uri=tracking_uri
    )

    logger.info(f"Computing PCA (n_components={n_components}, center_mode={center_mode})...")
    coords, evr, components, origin = compute_pca(
        checkpoints,
        n_components=n_components,
        identifier=run_id,
        center_mode=center_mode,
        force=force,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Instantiating dataset and model from config...")
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

    logger.info("Loading pre-normalized datasets onto device...")
    dataset_tensors = {}
    target_datasets = [("train", data_container.train), ("test", data_container.test)]
    if data_container.train_canary:
        target_datasets.append(("train_canary", data_container.train_canary))
    if data_container.test_canary:
        target_datasets.append(("test_canary", data_container.test_canary))

    for name, ds in target_datasets:
        loader = DataLoader(ds, batch_size=2000, shuffle=False, collate_fn=safe_collate)
        xs, ys = [], []
        for bx, by in loader:
            xs.append(bx)
            ys.append(by)
        x_all = torch.cat(xs).to(device=device, dtype=torch.float32)
        y_all = torch.cat(ys).to(device=device)
        if norm_mean is not None and norm_std is not None:
            x_all = (x_all - norm_mean) / norm_std
        dataset_tensors[name] = (x_all, y_all)

    logger.info("Evaluating exact checkpoint losses on train set...")
    x_train, y_train = dataset_tensors["train"]
    checkpoint_losses = []
    checkpoint_weight_norms = []
    for step in sorted(checkpoints.keys()):
        ckpt = checkpoints[step]
        model.load_state_dict(ckpt)
        loss_val = evaluate_loss_tensors(model, x_train, y_train, loss_fn, batch_size=2000)
        flat_w = flatten_weights(ckpt)
        norm_val = float(torch.norm(flat_w).item())
        checkpoint_losses.append(loss_val)
        checkpoint_weight_norms.append(norm_val)

    # Persist exact checkpoint losses inside projection.json
    proj_path = trajectory_projection_path(run_id)
    if proj_path.exists():
        proj_data = json.loads(proj_path.read_text())
        proj_data["trajectories"]["losses"] = checkpoint_losses
        proj_data["trajectories"]["weight_norms"] = checkpoint_weight_norms
        proj_path.write_text(json.dumps(proj_data))
        logger.info("Updated projection.json with exact checkpoint losses.")

    grid_search(
        coords=coords,
        components=components,
        origin=origin,
        model=model,
        loss_fn=loss_fn,
        dataset_tensors=dataset_tensors,
        identifier=run_id,
        resolution=grid_resolution,
        margin=grid_margin,
        device=device,
        force=force,
    )
    logger.info(f"Finished grid evaluation for run {run_id}.")


if __name__ == "__main__":
    app()
