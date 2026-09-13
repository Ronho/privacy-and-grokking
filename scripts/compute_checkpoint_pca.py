#!/usr/bin/env python3
"""
compute_checkpoint_pca.py

Computes PCA on neural network checkpoint weights over time.
Downloads and caches checkpoints from an MLflow tracking server (e.g. http://localhost:5051)
or reads them from local disk, fits PCA on the weight trajectory, and caches the resulting
projections and explained variance metadata in cache/.
"""

import argparse
import io
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from mlflow import MlflowClient
from sklearn.decomposition import PCA
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_CACHE_DIR = PROJECT_DIR / "cache"
DEFAULT_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5051")


def get_available_checkpoints_via_api(tracking_uri: str, run_id: str) -> list[int]:
    """Queries MLflow REST API to list all checkpoint step integers for a given run."""
    base_uri = tracking_uri.rstrip("/")
    url = f"{base_uri}/api/2.0/mlflow/artifacts/list?run_id={run_id}&path=checkpoints"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "privacy-and-grokking-pca"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        files = data.get("files", [])
        steps = []
        for f in files:
            p = f.get("path", "")
            step_str = p.split("/")[-1]
            if step_str.isdigit():
                steps.append(int(step_str))
        return sorted(steps)
    except Exception as e:
        print(f"Warning: Failed to fetch artifacts via REST API ({e}). Attempting fallback...")
        return []


def get_available_checkpoints_local(search_dir: str | Path, run_id: str) -> list[int]:
    """Scans local filesystem for checkpoint steps."""
    search_path = Path(search_dir)
    # Direct candidate paths
    direct_candidates = [
        search_path / "downloaded_artifacts" / run_id / "checkpoints",
        search_path / "checkpoints" / run_id,
        search_path / run_id / "artifacts" / "checkpoints",
        search_path / run_id / "checkpoints",
    ]
    for cand in direct_candidates:
        if cand.is_dir():
            steps = [int(p.name) for p in cand.iterdir() if p.is_dir() and p.name.isdigit()]
            if steps:
                return sorted(steps)

    for root, _dirs, _ in os.walk(str(search_path)):
        if os.path.basename(root) == run_id:
            for sub in [os.path.join(root, "artifacts", "checkpoints"), os.path.join(root, "checkpoints")]:
                if os.path.isdir(sub):
                    steps = [int(p) for p in os.listdir(sub) if p.isdigit()]
                    if steps:
                        return sorted(steps)
    return []


def download_or_load_checkpoint(
    tracking_uri: str, run_id: str, step: int, cache_dir: Path
) -> dict[str, torch.Tensor]:
    """Loads checkpoint state_dict. Checks local cache first; if missing, downloads
    via MLflow /get-artifact HTTP endpoint and caches locally.
    """
    cached_ckpt_dir = cache_dir / "checkpoints" / run_id / str(step)
    cached_file = cached_ckpt_dir / "model.pth"

    if cached_file.is_file() and cached_file.stat().st_size > 0:
        return torch.load(cached_file, map_location="cpu", weights_only=True)

    alt_cached = cache_dir / "downloaded_artifacts" / run_id / "checkpoints" / str(step) / "model.pth"
    if alt_cached.is_file() and alt_cached.stat().st_size > 0:
        return torch.load(alt_cached, map_location="cpu", weights_only=True)

    # Download from HTTP tracking server
    base_uri = tracking_uri.rstrip("/")
    artifact_url = f"{base_uri}/get-artifact?path=checkpoints/{step}/model.pth&run_uuid={run_id}"

    try:
        req = urllib.request.Request(
            artifact_url, headers={"User-Agent": "privacy-and-grokking-pca"}
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()

        # Cache to disk for instant future loads
        cached_ckpt_dir.mkdir(parents=True, exist_ok=True)
        with open(cached_file, "wb") as f:
            f.write(data)

        buf = io.BytesIO(data)
        return torch.load(buf, map_location="cpu", weights_only=True)
    except Exception as e:
        raise RuntimeError(
            f"Failed to fetch checkpoint at step {step} for run {run_id} from {artifact_url}: {e}"
        ) from e


def extract_flattened_weights(state_dict: dict[str, torch.Tensor]) -> np.ndarray:
    """Deterministically flattens all floating point weight/bias parameters
    into a 1D numpy array.
    """
    float_tensors = [
        v.detach().cpu().flatten()
        for k, v in sorted(state_dict.items())
        if torch.is_tensor(v) and v.dtype.is_floating_point
    ]
    if not float_tensors:
        raise ValueError("No floating-point parameter tensors found in checkpoint state_dict!")
    return torch.cat(float_tensors).numpy()


def get_parameter_layout(
    state_dict: dict[str, torch.Tensor],
) -> tuple[list[str], list[list[int]], list[int], list[str]]:
    """Extracts metadata to map a flattened parameter vector back to state_dict."""
    param_names = []
    param_shapes = []
    param_numels = []
    param_dtypes = []
    for k, v in sorted(state_dict.items()):
        if torch.is_tensor(v) and v.dtype.is_floating_point:
            param_names.append(k)
            param_shapes.append(list(v.shape))
            param_numels.append(v.numel())
            param_dtypes.append(str(v.dtype).replace("torch.", ""))
    return param_names, param_shapes, param_numels, param_dtypes


def fetch_metric_map(
    client: MlflowClient, run_id: str, metric_keys: list[str]
) -> dict[str, dict[int, float]]:
    """Fetches full history of given metrics, returning {metric_key: {step: value}}."""
    result: dict[str, dict[int, float]] = {}
    for k in metric_keys:
        try:
            hist = client.get_metric_history(run_id, k)
            result[k] = {m.step: m.value for m in hist}
        except Exception:
            result[k] = {}
    return result


def find_best_matching_metric(step: int, step_to_value: dict[int, float]) -> float | None:
    """Finds metric value at step, or nearest available step if not exact."""
    if not step_to_value:
        return None
    if step in step_to_value:
        return step_to_value[step]
    # Nearest step
    closest_step = min(step_to_value.keys(), key=lambda s: abs(s - step))
    return step_to_value[closest_step]


def compute_checkpoint_pca(
    run_id: str,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    frequency: int | None = None,
    all_checkpoints: bool = True,
    min_step: int | None = None,
    max_step: int | None = None,
    n_components: int = 10,
    loss_metric: str = "auto",
    cache_dir: Path = DEFAULT_CACHE_DIR,
    output_path: Path | None = None,
    save_basis: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Main execution function: loads checkpoints, runs PCA, and saves cache files."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to MLflow tracking server: {tracking_uri}")
    client = MlflowClient(tracking_uri)
    run = None
    run_name = run_id

    try:
        run = client.get_run(run_id)
        run_name = run.data.tags.get("mlflow.runName", run.info.run_name or run_id)
        print(f"Found Run: '{run_name}' (ID: {run_id}, Status: {run.info.status})")
    except Exception as e:
        print(f"Warning: Could not fetch run '{run_id}' from MLflow ({e}). Continuing with local cache...")

    # 1. Discover available checkpoint steps
    print(f"Discovering checkpoints for run {run_id}...")
    steps = []
    if tracking_uri.startswith("http://") or tracking_uri.startswith("https://"):
        steps = get_available_checkpoints_via_api(tracking_uri, run_id)
    if not steps:
        steps = get_available_checkpoints_local(cache_dir, run_id)
    if not steps and not (tracking_uri.startswith("http://") or tracking_uri.startswith("https://")):
        steps = get_available_checkpoints_local(tracking_uri, run_id)

    if not steps:
        raise RuntimeError(f"No checkpoints found for run {run_id}!")

    print(f"Found {len(steps)} total checkpoints on server (min: {steps[0]}, max: {steps[-1]}).")

    # Filter by min_step and max_step
    if min_step is not None:
        steps = [s for s in steps if s >= min_step]
    if max_step is not None:
        steps = [s for s in steps if s <= max_step]

    if not steps:
        raise RuntimeError(
            f"No checkpoints found within range [min_step={min_step}, max_step={max_step}]!"
        )

    # 2. Filter steps by frequency if requested
    if not all_checkpoints and frequency is not None and frequency > 0:
        filtered_steps = [
            s for s in steps if s % frequency == 0 or s == steps[0] or s == steps[-1]
        ]
        print(f"Filtered to {len(filtered_steps)} checkpoints using frequency={frequency}.")
    else:
        filtered_steps = steps
        print(f"Using {len(filtered_steps)} checkpoints (range: {steps[0]} to {steps[-1]}).")

    # 3. Determine and fetch loss / metrics
    available_metrics = list(run.data.metrics.keys()) if run else []
    chosen_loss_metric = None

    if loss_metric != "auto":
        chosen_loss_metric = loss_metric
    else:
        # Priority list for default loss metrics based on run type
        if "mse" in run_name.lower():
            loss_candidates = [
                "eval/train/loss/mse/mean",
                "eval/train/loss/cross_entropy/mean",
                "train/total_loss",
                "train/task_loss",
                "train_loss",
                "loss",
            ]
        else:
            loss_candidates = [
                "eval/train/loss/cross_entropy/mean",
                "eval/train/loss/mse/mean",
                "train/total_loss",
                "train/task_loss",
                "train_loss",
                "loss",
            ]
        for cand in loss_candidates:
            if cand in available_metrics:
                chosen_loss_metric = cand
                break
        if not chosen_loss_metric:
            # Look for any metric with loss
            losses = [m for m in available_metrics if "loss" in m.lower()]
            chosen_loss_metric = losses[0] if losses else None

    print(f"Using loss metric: '{chosen_loss_metric}'")

    metric_keys_to_fetch = []
    if chosen_loss_metric:
        metric_keys_to_fetch.append(chosen_loss_metric)
    for aux in [
        "eval/test/loss/cross_entropy/mean",
        "eval/test/loss/mse/mean",
        "eval/train/accuracy",
        "eval/test/accuracy",
        "epoch",
    ]:
        if aux in available_metrics and aux not in metric_keys_to_fetch:
            metric_keys_to_fetch.append(aux)

    metric_histories = fetch_metric_map(client, run_id, metric_keys_to_fetch)
    loss_history = metric_histories.get(chosen_loss_metric, {}) if chosen_loss_metric else {}

    # 4. Load weights for each step
    print(f"Loading and vectorizing model weights across {len(filtered_steps)} checkpoints...")
    weight_vectors = []
    valid_steps = []
    step_losses = []
    step_epochs = []
    step_test_losses = []
    step_accuracies = []

    if "mse" in run_name.lower():
        test_loss_candidates = ["eval/test/loss/mse/mean", "eval/test/loss/cross_entropy/mean"]
    else:
        test_loss_candidates = ["eval/test/loss/cross_entropy/mean", "eval/test/loss/mse/mean"]

    test_loss_key = next(
        (k for k in test_loss_candidates if k in metric_histories),
        None,
    )
    train_acc_key = "eval/train/accuracy" if "eval/train/accuracy" in metric_histories else None

    sample_state_dict = None
    for step in tqdm(filtered_steps, desc="Processing checkpoints", unit="ckpt"):
        try:
            state_dict = download_or_load_checkpoint(tracking_uri, run_id, step, cache_dir)
            if sample_state_dict is None:
                sample_state_dict = state_dict
            w_flat = extract_flattened_weights(state_dict)
            weight_vectors.append(w_flat)
            valid_steps.append(step)

            # Match metrics
            l_val = find_best_matching_metric(step, loss_history)
            step_losses.append(l_val)

            ep_val = find_best_matching_metric(step, metric_histories.get("epoch", {}))
            step_epochs.append(ep_val if ep_val is not None else float(step))

            tl_val = (
                find_best_matching_metric(step, metric_histories.get(test_loss_key, {}))
                if test_loss_key
                else None
            )
            step_test_losses.append(tl_val)

            acc_val = (
                find_best_matching_metric(step, metric_histories.get(train_acc_key, {}))
                if train_acc_key
                else None
            )
            step_accuracies.append(acc_val)
        except Exception as e:
            print(f"\nWarning: Skipped checkpoint at step {step}: {e}")

    if len(weight_vectors) < 2:
        msg = f"Need at least 2 checkpoints to compute PCA, got {len(weight_vectors)}."
        raise RuntimeError(msg)

    # Stack weights: (N, D)
    weight_matrix = np.stack(weight_vectors)
    n_samples, n_features = weight_matrix.shape
    actual_n_components = min(n_components, n_samples, n_features)
    print(
        f"Weight matrix shape: {n_samples} checkpoints x {n_features:,} parameters.\n"
        f"Computing PCA ({actual_n_components} components)..."
    )

    # 5. Fit PCA
    pca = PCA(n_components=actual_n_components)
    projections = pca.fit_transform(weight_matrix)  # Shape (N, actual_n_components)

    explained_variance_ratio = pca.explained_variance_ratio_.tolist()
    singular_values = pca.singular_values_.tolist()
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_).tolist()

    print("\n--- PCA Explained Variance Summary ---")
    for idx, (ev, cum) in enumerate(
        zip(explained_variance_ratio, cumulative_variance, strict=False)
    ):
        print(f"  PC {idx + 1:2d}: {ev * 100:6.2f}% variance  (Cumulative: {cum * 100:6.2f}%)")
    print("--------------------------------------\n")

    # 6. Construct output DataFrame
    df_dict = {
        "run_id": run_id,
        "run_name": run_name,
        "step": valid_steps,
        "epoch": step_epochs,
        "loss": step_losses,
    }
    if any(tl is not None for tl in step_test_losses):
        df_dict["test_loss"] = step_test_losses
    if any(acc is not None for acc in step_accuracies):
        df_dict["accuracy"] = step_accuracies

    # Calculate exact Euclidean distances in full weight space
    dist_from_start = np.linalg.norm(weight_matrix - weight_matrix[0], axis=1)
    step_diffs = np.linalg.norm(weight_matrix[1:] - weight_matrix[:-1], axis=1)
    step_dist = np.concatenate([[0.0], step_diffs])

    df_dict["distance_from_start"] = dist_from_start
    df_dict["step_distance"] = step_dist

    for comp_idx in range(actual_n_components):
        df_dict[f"pc{comp_idx + 1}"] = projections[:, comp_idx]

    df = pd.DataFrame(df_dict)

    # Default output path
    output_path = cache_dir / f"{run_id}_pca.parquet" if output_path is None else Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Saved PCA projections ({len(df)} rows) to: {output_path}")

    # Save full loss history across all training steps if available
    if loss_history:
        loss_rows = [{"step": s, "loss": v} for s, v in sorted(loss_history.items())]
        df_loss_full = pd.DataFrame(loss_rows)
        if test_loss_key and test_loss_key in metric_histories:
            test_hist = metric_histories[test_loss_key]
            df_loss_full["test_loss"] = df_loss_full["step"].map(test_hist)
        loss_full_path = cache_dir / f"{run_id}_loss_full.parquet"
        df_loss_full.to_parquet(loss_full_path, index=False)
        print(f"Saved full loss history ({len(df_loss_full)} points) to: {loss_full_path}")

    # Metadata JSON
    meta_path = output_path.with_name(f"{output_path.stem}_meta.json")
    meta_info = {
        "run_id": run_id,
        "run_name": run_name,
        "n_checkpoints": n_samples,
        "n_parameters": n_features,
        "n_components": actual_n_components,
        "loss_metric": chosen_loss_metric,
        "explained_variance_ratio": explained_variance_ratio,
        "cumulative_variance": cumulative_variance,
        "singular_values": singular_values,
        "total_explained_variance_top2": sum(explained_variance_ratio[:2]),
        "total_explained_variance_all": sum(explained_variance_ratio),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=2)
    print(f"Saved PCA metadata to: {meta_path}")

    # Save PCA reconstruction basis
    if save_basis and sample_state_dict is not None:
        basis_path = output_path.with_name(f"{output_path.stem}_basis.npz")
        param_names, param_shapes, param_numels, param_dtypes = get_parameter_layout(sample_state_dict)
        np.savez_compressed(
            basis_path,
            components=pca.components_[:actual_n_components],
            mean=pca.mean_,
            explained_variance_ratio=np.array(explained_variance_ratio),
            singular_values=np.array(singular_values),
            param_names=np.array(param_names),
            param_shapes=np.array(param_shapes, dtype=object),
            param_numels=np.array(param_numels, dtype=np.int64),
            param_dtypes=np.array(param_dtypes),
        )
        print(f"Saved PCA reconstruction basis to: {basis_path}")

    return df, meta_info


def main():
    parser = argparse.ArgumentParser(
        description="Compute PCA trajectory on neural network model checkpoints from MLflow."
    )
    parser.add_argument(
        "--run-id",
        "-r",
        type=str,
        required=True,
        help="MLflow Run ID to process (e.g. b75317f9447346249f0b811acec427ce)",
    )
    parser.add_argument(
        "--uri",
        "-u",
        type=str,
        default=DEFAULT_TRACKING_URI,
        help=f"MLflow tracking URI (default: {DEFAULT_TRACKING_URI})",
    )
    parser.add_argument(
        "--frequency",
        "-f",
        type=int,
        default=None,
        help=(
            "Optional step frequency to downsample checkpoints (e.g. 1000). "
            "If omitted, all available checkpoints are used."
        ),
    )
    parser.add_argument(
        "--min-step",
        type=int,
        default=None,
        help="Optional minimum step number to consider (e.g. 10000 to exclude initial steps).",
    )
    parser.add_argument(
        "--max-step",
        type=int,
        default=None,
        help="Optional maximum step number to consider.",
    )
    parser.add_argument(
        "--n-components",
        "-k",
        type=int,
        default=10,
        help="Number of principal components to compute (default: 10).",
    )
    parser.add_argument(
        "--loss-metric",
        type=str,
        default="auto",
        help="Loss metric name to extract for point sizing (default: 'auto').",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=str(DEFAULT_CACHE_DIR),
        help="Directory to store checkpoint cache and PCA parquet output.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Custom output parquet file path (default: cache/<run_id>_pca.parquet).",
    )
    parser.add_argument(
        "--save-basis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save PCA basis vectors (PC1, PC2, mean, and parameter layout) to cache (default: True).",
    )

    args = parser.parse_args()

    compute_checkpoint_pca(
        run_id=args.run_id,
        tracking_uri=args.uri,
        frequency=args.frequency,
        all_checkpoints=(args.frequency is None),
        min_step=args.min_step,
        max_step=args.max_step,
        n_components=args.n_components,
        loss_metric=args.loss_metric,
        cache_dir=Path(args.cache_dir),
        output_path=Path(args.output) if args.output else None,
        save_basis=args.save_basis,
    )


if __name__ == "__main__":
    main()
