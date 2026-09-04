import argparse
import sys
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import requests
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Ensure privacy_and_grokking module can be imported
sys.path.append(str(Path(__file__).parent.parent / "src"))
from privacy_and_grokking.config import TrainConfig
from privacy_and_grokking.utils.logger import Logger


def parse_args():
    parser = argparse.ArgumentParser(description="Extract Info-RMIA Signals")
    parser.add_argument(
        "--experiment-name", type=str, required=True, help="Name of the MLflow experiment"
    )
    parser.add_argument("--model-name", type=str, required=True, help="Model name to filter for")
    parser.add_argument(
        "--tracking-uri", type=str, default="http://localhost:5051", help="MLflow tracking URI"
    )
    return parser.parse_args()


def validate_runs_differ_only_by_seed(runs_df):
    """Ensure runs differ only by seed related parameters."""
    if runs_df.empty:
        return True

    # Columns to ignore during equivalence check
    ignore_cols_prefixes = [
        "tags.",
        "metrics.",
        "run_id",
        "experiment_id",
        "status",
        "start_time",
        "end_time",
        "artifact_uri",
    ]
    ignore_exact_params = [
        "params.seed",
        "params.data.seed",
        "params.data.mask.seed",
        "params.data.mask.model_index",
    ]

    check_cols = []
    for col in runs_df.columns:
        if any(col.startswith(p) for p in ignore_cols_prefixes):
            continue
        if col in ignore_exact_params:
            continue
        check_cols.append(col)

    is_valid = True
    for col in check_cols:
        unique_vals = runs_df[col].dropna().unique()
        if len(unique_vals) > 1:
            print(f"Warning: Runs differ in parameter {col}: {unique_vals}")
            is_valid = False

    return is_valid


def build_datasets(cfg: TrainConfig):
    # Retrieve the datasets without mask
    container = cfg.data()
    train_dataset = container.train
    test_dataset = container.test

    # All Dataset: up to 500 train, up to 500 test.
    # Randomly sample. Equal amount.
    train_size = len(train_dataset)
    test_size = len(test_dataset)

    sample_size = min(train_size, test_size, 500)

    rng_all = np.random.default_rng(42)  # fixed seed for reproducibility across runs
    train_indices_all = rng_all.choice(train_size, sample_size, replace=False)
    test_indices_all = rng_all.choice(test_size, sample_size, replace=False)

    # Population Dataset: 1000 from test, non-overlapping with test_indices_all
    remaining_test_indices = np.setdiff1d(np.arange(test_size), test_indices_all)

    if len(remaining_test_indices) < 1000:
        print(
            f"Warning: Not enough remaining test samples for population dataset ({len(remaining_test_indices)} < 1000). Using all available."
        )
        pop_sample_size = len(remaining_test_indices)
    else:
        pop_sample_size = 1000

    pop_indices = rng_all.choice(remaining_test_indices, pop_sample_size, replace=False)

    # Create the actual tensors
    def extract_tensors(dataset, indices):
        xs = []
        ys = []
        for i in indices:
            x, y = dataset[i]
            xs.append(x)
            ys.append(y)
        return torch.stack(xs), torch.tensor(ys)

    train_x, train_y = extract_tensors(train_dataset, train_indices_all)
    test_x, test_y = extract_tensors(test_dataset, test_indices_all)

    all_x = torch.cat([train_x, test_x])
    all_y = torch.cat([train_y, test_y])

    pop_x, pop_y = extract_tensors(test_dataset, pop_indices)

    return all_x, all_y, pop_x, pop_y


def compute_signals_in_batches(model, x, y, device, norm_mean=None, norm_std=None, batch_size=256):
    all_probs = []
    dataset_size = x.size(0)
    for i in range(0, dataset_size, batch_size):
        batch_x = x[i : i + batch_size]
        batch_y = y[i : i + batch_size]

        if norm_mean is not None:
            batch_x = (batch_x - norm_mean) / norm_std

        with torch.no_grad():
            logits = model(batch_x)
            probs = F.softmax(logits, dim=1).gather(1, batch_y.view(-1, 1)).squeeze()

            # Handle the case where batch size is 1, causing squeeze() to return a scalar
            if probs.dim() == 0:
                probs = probs.unsqueeze(0)

            all_probs.append(probs.cpu().to(torch.float64))
    return torch.cat(all_probs)


def get_signals_for_step(
    step,
    run_ids,
    all_x,
    all_y,
    pop_x,
    pop_y,
    cfg,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    num_samples_all = all_x.size(0)
    num_samples_pop = pop_x.size(0)
    num_runs = len(run_ids)

    norm_mean = None
    norm_std = None
    if cfg.data().normalization is not None:
        norm_mean = torch.tensor(cfg.data().normalization.mean, device=device).view(-1, 1, 1)
        norm_std = torch.tensor(cfg.data().normalization.std, device=device).view(-1, 1, 1)

    all_signals = torch.zeros((num_samples_all, num_runs), dtype=torch.float64)
    pop_signals = torch.zeros((num_samples_pop, num_runs), dtype=torch.float64)

    for run_idx, run_id in enumerate(
        tqdm(run_ids, desc=f"Processing runs for step {step}", leave=False)
    ):
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                artifact_path = f"checkpoints/{step}/model.pth"
                model_path = Path(tmpdir) / "model.pth"
                tracking_uri = mlflow.get_tracking_uri()
                download_artifact(tracking_uri, run_id, artifact_path, str(model_path))

                if not model_path.exists():
                    # It's possible some runs didn't save for this step
                    continue

                model = cfg.model(
                    input_dim=cfg.data().input_shape, num_classes=cfg.data().num_classes
                )
                model.load_state_dict(
                    torch.load(model_path, map_location=device, weights_only=True)
                )
                model.to(device)
                model.eval()

                all_signals[:, run_idx] = compute_signals_in_batches(
                    model, all_x, all_y, device, norm_mean, norm_std
                )
                pop_signals[:, run_idx] = compute_signals_in_batches(
                    model, pop_x, pop_y, device, norm_mean, norm_std
                )
        except Exception as e:
            print(f"Error processing run {run_id} step {step}: {e}")

    return all_signals, pop_signals


def fetch_json_artifact(tracking_uri: str, run_id: str, artifact_path: str) -> dict:
    base_uri = tracking_uri.rstrip("/")
    url = f"{base_uri}/get-artifact?path={artifact_path}&run_uuid={run_id}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def download_artifact(tracking_uri: str, run_id: str, artifact_path: str, dst_path: str) -> None:
    base_uri = tracking_uri.rstrip("/")
    url = f"{base_uri}/get-artifact?path={artifact_path}&run_uuid={run_id}"
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dst_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def main():
    args = parse_args()
    Logger().setup()

    print(f"Connecting to MLflow server at {args.tracking_uri}...")
    mlflow.set_tracking_uri(args.tracking_uri)

    experiments = mlflow.search_experiments(filter_string=f"name = '{args.experiment_name}'")
    if not experiments:
        print(f"Error: Experiment '{args.experiment_name}' not found.")
        return

    experiment_ids = [exp.experiment_id for exp in experiments]

    print("Fetching runs...")
    runs_df = mlflow.search_runs(experiment_ids=experiment_ids)

    if runs_df.empty:
        print("No runs found for these experiments.")
        return

    if "tags.mlflow.runName" in runs_df.columns:
        filtered_runs = runs_df[runs_df["tags.mlflow.runName"] == args.model_name]
    else:
        print("Warning: 'tags.mlflow.runName' not found in MLflow. Skipping model filter.")
        filtered_runs = runs_df

    run_ids = filtered_runs["run_id"].tolist()
    print(f"Found {len(run_ids)} runs for model '{args.model_name}'.")

    if not validate_runs_differ_only_by_seed(filtered_runs):
        print(
            "Validation failed: Runs differ by more than just seed parameters. Proceeding anyway."
        )

    if not run_ids:
        print("No runs to process. Exiting.")
        return

    # Find a run with training_config.json to load the config
    cfg = None
    for r_id in run_ids:
        try:
            print(f"Trying runs:/{r_id}/training_config.json")
            config_dict = fetch_json_artifact(args.tracking_uri, r_id, "training_config.json")
            cfg = TrainConfig.model_validate(config_dict)
            print(f"Loaded config from run {r_id}")
            break
        except Exception as e:
            print(e)
            continue

    if cfg is None:
        print("Error: Could not find training_config.json in any of the runs.")
        return

    print("Building datasets...")
    all_x, all_y, pop_x, pop_y = build_datasets(cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_x, all_y = all_x.to(device), all_y.to(device)
    pop_x, pop_y = pop_x.to(device), pop_y.to(device)

    print(f"All dataset size: {all_x.size(0)}")
    print(f"Population dataset size: {pop_x.size(0)}")

    output_dir = Path(__file__).parent / "signals"
    output_dir.mkdir(exist_ok=True, parents=True)

    # Map column index in signal arrays to run_id
    torch.save(run_ids, output_dir / "run_ids.pt")
    print(f"Saved run_ids mapping to {output_dir / 'run_ids.pt'}")

    steps = list(range(0, 150001, 10000))
    for step in steps:
        print(f"\nProcessing step {step}...")
        all_signals, pop_signals = get_signals_for_step(
            step, run_ids, all_x, all_y, pop_x, pop_y, cfg, device
        )

        all_out = output_dir / f"all_signals_step_{step}.pt"
        pop_out = output_dir / f"population_signals_step_{step}.pt"

        torch.save(all_signals, all_out)
        torch.save(pop_signals, pop_out)
        print(f"Saved step {step} signals.")


if __name__ == "__main__":
    main()
