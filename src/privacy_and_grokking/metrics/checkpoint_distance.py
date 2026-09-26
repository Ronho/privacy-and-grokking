"""Checkpoint distance metrics and trajectory comparisons.

Provides utilities to discover checkpoints, load parameter vectors, and compute
pairwise Euclidean distance matrices between optimization trajectories of models.
"""

from __future__ import annotations

import io
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from mlflow import MlflowClient

from privacy_and_grokking.artifacts.cache import DEFAULT_TRACKING_URI, get_artifact_json
from privacy_and_grokking.artifacts.local import (
    checkpoint_distance_cache_path,
    local_run_dir,
)


def fetch_run_name(run_id: str, tracking_uri: str = DEFAULT_TRACKING_URI) -> str:
    """Fetch human-readable run name from MLflow or local training_config.json cache.

    Falls back to run_id if metadata is unavailable.
    """
    try:
        cfg = get_artifact_json(run_id, "training_config.json", tracking_uri=tracking_uri)
        if "name" in cfg:
            return cfg["name"]
    except Exception:
        pass

    try:
        client = MlflowClient(tracking_uri)
        run = client.get_run(run_id)
        return run.data.tags.get("mlflow.runName", run.info.run_name or run_id)
    except Exception:
        return run_id


def get_available_checkpoints(
    run_id: str,
    tracking_uri: str = DEFAULT_TRACKING_URI,
) -> list[int]:
    """Retrieve sorted list of available checkpoint step integers for a run.

    Caches results in ``cache/runs/{run_id}/checkpoints/info.json`` to avoid repeated
    network requests to MLflow.
    """
    checkpoints_dir = local_run_dir(run_id) / "checkpoints"
    info_file = checkpoints_dir / "info.json"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    if info_file.exists():
        try:
            return sorted(json.loads(info_file.read_bytes())["steps"])
        except Exception:
            pass

    # Query MLflow REST API endpoint to list all checkpoint artifacts on the server
    base_uri = tracking_uri.rstrip("/")
    url = f"{base_uri}/api/2.0/mlflow/artifacts/list?run_id={run_id}&path=checkpoints"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "privacy-and-grokking"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        files = data.get("files", [])
        steps: list[int] = []
        for f in files:
            p = f.get("path", "")
            step_str = p.split("/")[-1]
            if step_str.isdigit():
                steps.append(int(step_str))
        if steps:
            sorted_steps = sorted(set(steps))
            info_file.write_text(json.dumps({"steps": sorted_steps}))
            return sorted_steps
    except Exception as err:
        print(f"Warning: Could not fetch checkpoints from MLflow for '{run_id}': {err}")

    # Fallback to local directories if previously downloaded / offline
    local_steps: list[int] = []
    if checkpoints_dir.exists():
        for item in checkpoints_dir.iterdir():
            if item.is_dir() and item.name.isdigit() and (item / "model.pth").exists():
                local_steps.append(int(item.name))
    if local_steps:
        sorted_steps = sorted(set(local_steps))
        info_file.write_text(json.dumps({"steps": sorted_steps}))
        return sorted_steps

    raise RuntimeError(f"No checkpoints found locally or on server for run '{run_id}'")


def load_checkpoint_vector(
    run_id: str,
    step: int,
    tracking_uri: str = DEFAULT_TRACKING_URI,
) -> torch.Tensor:
    """Load model checkpoint weights and flatten all parameters into a 1D tensor.

    Floating-point parameter tensors are concatenated into a single vector representation
    suitable for fast distance metric calculations.
    """
    model_file = local_run_dir(run_id) / "checkpoints" / str(step) / "model.pth"
    model_file.parent.mkdir(parents=True, exist_ok=True)

    if model_file.exists():
        state_dict = torch.load(model_file, map_location="cpu", weights_only=True)
    else:
        # Download from remote MLflow server via HTTP get-artifact endpoint
        base_uri = tracking_uri.rstrip("/")
        url = f"{base_uri}/get-artifact?path=checkpoints/{step}/model.pth&run_uuid={run_id}"
        req = urllib.request.Request(url, headers={"User-Agent": "privacy-and-grokking"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()


        # Atomically cache locally to prevent incomplete file writes during interrupted runs
        temp_file = model_file.with_name(f".{model_file.name}.tmp")
        temp_file.write_bytes(data)
        temp_file.replace(model_file)

        buf = io.BytesIO(data)
        state_dict = torch.load(buf, map_location="cpu", weights_only=True)

    # Flatten and concatenate all parameter tensors into a single float32 1D vector
    param_tensors = [
        param.detach().float().flatten()
        for param in state_dict.values()
        if isinstance(param, torch.Tensor) and param.is_floating_point()
    ]
    if not param_tensors:
        raise ValueError(
            f"No floating point parameters found in checkpoint for run {run_id} step {step}"
        )

    return torch.cat(param_tensors)


def load_run_trajectory_matrix(
    run_id: str,
    steps: list[int] | None = None,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    max_workers: int = 8,
) -> tuple[list[int], torch.Tensor]:
    """Load all checkpoint parameter vectors for a run into a 2D matrix (N, D).

    Uses a thread pool to download and process checkpoints concurrently.
    """
    if steps is None:
        steps = get_available_checkpoints(run_id, tracking_uri=tracking_uri)

    if not steps:
        raise ValueError(f"No checkpoints found for run_id '{run_id}'")

    print(f"Loading {len(steps)} checkpoints for run {run_id[:8]}...")
    vectors: dict[int, torch.Tensor] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_step = {
            executor.submit(load_checkpoint_vector, run_id, step, tracking_uri): step
            for step in steps
        }
        for future in as_completed(future_to_step):
            step = future_to_step[future]
            vectors[step] = future.result()

    # Stack ordered by checkpoint step
    ordered_vectors = [vectors[s] for s in steps]
    matrix = torch.stack(ordered_vectors, dim=0)  # Shape: (num_steps, num_params)
    return steps, matrix


def compute_run_pair_distance(
    run_id_1: str,
    run_id_2: str,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    max_workers: int = 8,
    force_recompute: bool = False,
) -> dict:
    """Compute and cache the pairwise Euclidean distance matrix between two runs.

    Returns the dictionary containing metadata and the computed distance matrix.
    """
    out_path = checkpoint_distance_cache_path(run_id_1, run_id_2)
    if out_path.exists() and not force_recompute:
        print(f"Loading cached checkpoint distance matrix from {out_path}...")
        return json.loads(out_path.read_text())

    steps_1, mat_1 = load_run_trajectory_matrix(
        run_id_1, tracking_uri=tracking_uri, max_workers=max_workers
    )
    if run_id_1 == run_id_2:
        steps_2, mat_2 = steps_1, mat_1
    else:
        steps_2, mat_2 = load_run_trajectory_matrix(
            run_id_2, tracking_uri=tracking_uri, max_workers=max_workers
        )

    # Compute pairwise Euclidean distance between all parameter vectors
    print(f"Computing Euclidean distance matrix ({len(steps_1)} x {len(steps_2)})...")
    dist_matrix = torch.cdist(mat_1, mat_2, p=2.0).numpy()

    run_name_1 = fetch_run_name(run_id_1, tracking_uri=tracking_uri)
    run_name_2 = fetch_run_name(run_id_2, tracking_uri=tracking_uri)

    result = {
        "run_id_1": run_id_1,
        "run_id_2": run_id_2,
        "run_name_1": run_name_1,
        "run_name_2": run_name_2,
        "steps_1": steps_1,
        "steps_2": steps_2,
        "matrix": dist_matrix.tolist(),
        "metric": "euclidean",
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Saved distance matrix to {out_path}")
    return result
