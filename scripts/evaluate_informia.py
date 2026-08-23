import argparse
import tempfile
import os
import json
from pathlib import Path
from typing import List, Dict, Any

import mlflow
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

from privacy_and_grokking.logger import Logger
from privacy_and_grokking.training.config import TrainConfig
from privacy_and_grokking.datasets.base import Normalization

# InfoRMIA functions
def get_rmia_out_signals(
    ref_signals: np.ndarray,
    ref_memberships: np.ndarray,
    target_model_idx: int,
    num_reference_models: int,
    offline_a: float,
) -> np.ndarray:
    mask = np.ones(ref_signals.shape[1], dtype=bool)
    mask[target_model_idx] = False
    
    ref_signals = ref_signals[:, mask]
    ref_memberships = ref_memberships[:, mask]

    non_members = ~ref_memberships
    out_signals = ref_signals * non_members
    if num_reference_models is None:
        num_reference_models = ref_signals.shape[1]
    if num_reference_models > 1:
        out_signals = -np.sort(-out_signals, axis=1)[:, :num_reference_models]
    else:
        if offline_a != 0:
            out_signals += ((ref_signals + offline_a - 1) / offline_a) * ref_memberships
        else:
            out_signals += ((ref_signals - 0.7) / 0.3) * ref_memberships
    return out_signals

def run_informia(
    target_model_idx: int,
    all_signals: np.ndarray,
    population_signals: np.ndarray,
    all_memberships: np.ndarray,
    num_reference_models: int,
    offline_a: float
    ) -> np.ndarray:
    target_signals = all_signals[:, target_model_idx]
    out_signals = get_rmia_out_signals(
        all_signals, all_memberships, target_model_idx, num_reference_models, offline_a
    )
    mean_out_x = np.mean(out_signals, axis=1)
    mean_x = (((1 + offline_a) / 2) * mean_out_x + ((1 - offline_a) / 2))
    mean_x = np.clip(mean_x, 1e-12, None)
    log_ratio_x = np.log(np.clip(target_signals.ravel() / mean_x, 1e-12, None))
    
    population_memberships = np.zeros_like(population_signals, dtype=bool)
    z_signals = population_signals[:, target_model_idx]
    z_out_signals = get_rmia_out_signals(
        population_signals, population_memberships, target_model_idx, num_reference_models, offline_a
    )
    mean_out_z = np.mean(z_out_signals, axis=1)
    mean_z = (((1 + offline_a) / 2) * mean_out_z + ((1 - offline_a) / 2))
    mean_z = np.clip(mean_z, 1e-12, None)
    prob_ratio_z = np.clip(z_signals.ravel() / mean_z, 1e-12, None)
    
    test_statistic = log_ratio_x - np.sum(mean_z * np.log(prob_ratio_z)) / mean_z.sum()
    return test_statistic

def tune_optimal_a(
    target_model_idx: int,
    reference_model_idx: int,
    all_signals: np.ndarray,
    population_signals: np.ndarray,
    all_memberships: np.ndarray,
    num_reference_models: int,
) -> float:
    best_a = 0.0
    best_auc = -1.0
    for a in np.arange(0.0, 1.1, 0.1):
        stats = run_informia(
            reference_model_idx, all_signals, population_signals, all_memberships, num_reference_models, float(a)
        )
        target_memberships = all_memberships[:, reference_model_idx]
        auc = roc_auc_score(target_memberships, stats)
        if auc > best_auc:
            best_auc = auc
            best_a = a
    return float(best_a)

# Helper for parsing MLflow
def get_ignored_params(run) -> Dict[str, Any]:
    params = run.data.params
    ignored = {'params.seed', 'params.data.seed', 'params.data.mask.seed', 'params.data.mask.model_index'}
    return {k: v for k, v in params.items() if k not in ignored}

def group_runs(runs_df) -> List[List[str]]:
    groups = {}
    for idx, run in runs_df.iterrows():
        ignored_params = get_ignored_params(run)
        # Sort keys to ensure deterministic tuple creation
        key = tuple(sorted(ignored_params.items()))
        if key not in groups:
            groups[key] = []
        groups[key].append(run.info.run_id)
    return list(groups.values())

def fetch_json_artifact(tracking_uri: str, run_id: str, artifact_path: str) -> dict:
    base_uri = tracking_uri.rstrip('/')
    url = f"{base_uri}/get-artifact?path={artifact_path}&run_uuid={run_id}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def download_artifact(tracking_uri: str, run_id: str, artifact_path: str, dst_path: str) -> None:
    base_uri = tracking_uri.rstrip('/')
    url = f"{base_uri}/get-artifact?path={artifact_path}&run_uuid={run_id}"
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dst_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

@torch.no_grad()
def compute_signals_in_batches(model, x, y, device, norm_mean=None, norm_std=None, batch_size=1024):
    all_probs = []
    dataset_size = x.size(0)
    for i in range(0, dataset_size, batch_size):
        batch_x = x[i:i+batch_size].to(device)
        batch_y = y[i:i+batch_size].to(device)
        if norm_mean is not None:
            batch_x = (batch_x - norm_mean) / norm_std
        logit, _ = model(batch_x, verbose=False)
        prob = F.softmax(logit, dim=1)
        true_prob = prob.gather(1, batch_y.view(-1, 1)).squeeze(1)
        all_probs.append(true_prob.cpu())
    return torch.cat(all_probs)

def extract_tensors(dataset, indices):
    xs, ys = [], []
    for i in indices:
        x, y = dataset[i]
        xs.append(x)
        ys.append(y)
    if not isinstance(xs[0], torch.Tensor):
        xs = [torch.tensor(x) for x in xs]
    if not isinstance(ys[0], torch.Tensor):
        ys = [torch.tensor(y) for y in ys]
    return torch.stack(xs), torch.stack(ys)

def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    CACHE_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "cache"))
    os.makedirs(CACHE_DIR, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--model-name", required=False, default=None, help="Optional model name to filter runs. If omitted, evaluates all models in the experiment.")
    parser.add_argument("--tracking-uri", default="http://localhost:5051")
    parser.add_argument("--output", "-o", default=None, help="Output Parquet file")
    parser.add_argument("--num-samples", type=int, default=500, help="Number of samples to evaluate on (per member/non-member class)")
    args = parser.parse_args()
    
    if args.output is None:
        args.output = os.path.join(CACHE_DIR, f"{args.experiment_name}_informia_results.parquet")
        
    Logger().setup()
    mlflow.set_tracking_uri(args.tracking_uri)
    
    experiments = mlflow.search_experiments(filter_string=f"name = '{args.experiment_name}'")
    if not experiments:
        print(f"Error: Experiment '{args.experiment_name}' not found.")
        return
    experiment_ids = [exp.experiment_id for exp in experiments]
    
    if args.model_name:
        runs_df = mlflow.search_runs(experiment_ids=experiment_ids, filter_string=f"params.model.name = '{args.model_name}'")
    else:
        runs_df = mlflow.search_runs(experiment_ids=experiment_ids)
        
    if runs_df.empty:
        print("No runs found.")
        return
        
    run_groups = group_runs(runs_df)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if os.path.exists(args.output):
        existing_df = pd.read_parquet(args.output)
    else:
        existing_df = pd.DataFrame()
        
    default_cache = os.path.join(CACHE_DIR, f"{args.experiment_name}_informia_cache.json")
    cache_file = os.environ.get("INFORMIA_CACHE_FILE", default_cache)
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cache = json.load(f)
    else:
        cache = {}

    def save_cache():
        with open(cache_file, "w") as f:
            json.dump(cache, f)

    all_results = []

    print(f"Experiment: {args.experiment_name} | Model: {args.model_name or 'ALL'}")
    print("="*60)
    print(f"Found {len(run_groups)} run groups based on parameters.")
    
    for group_idx, run_ids in enumerate(run_groups):
        if len(run_ids) < 2:
            print(f"Group {group_idx} has only {len(run_ids)} runs, skipping (need at least 2).")
            continue
            
        print(f"\nProcessing group {group_idx} with runs: {run_ids}")
        
        # 1. Fetch configs and build membership matrix
        member_indices_per_run = []
        base_cfg = None
        for r_id in run_ids:
            config_dict = fetch_json_artifact(args.tracking_uri, r_id, "training_config.json")
            cfg = TrainConfig.model_validate(config_dict)
            if base_cfg is None:
                base_cfg = cfg
            
            # Retrieve the full indices selected for this run
            final_subset = cfg.data().train
            canary_ds = final_subset.dataset
            orig_indices = set()
            for mask_idx in final_subset.indices:
                orig_idx = canary_ds.subset_indices[mask_idx]
                orig_indices.add(int(orig_idx))
            member_indices_per_run.append(orig_indices)
            
        # Use base_cfg to get the full raw datasets
        full_container = base_cfg.data.data()
        rng = np.random.default_rng(42)
        
        # Randomly select samples to evaluate on (to save time)
        N_train = len(full_container.train)
        eval_train_indices = rng.choice(N_train, min(N_train, args.num_samples * 2), replace=False)
        
        # Same for population dataset
        N_test = len(full_container.test)
        pop_indices = rng.choice(N_test, min(N_test, args.num_samples), replace=False)
        
        print(f"Extracting tensors for group {group_idx}...")
        train_x, train_y = extract_tensors(full_container.train, eval_train_indices)
        pop_x, pop_y = extract_tensors(full_container.test, pop_indices)
        
        all_x, all_y = train_x.to(device), train_y.to(device)
        pop_x, pop_y = pop_x.to(device), pop_y.to(device)
        
        # Build all_memberships matrix
        all_memberships = np.zeros((len(eval_train_indices), len(run_ids)), dtype=bool)
        for i, member_set in enumerate(member_indices_per_run):
            for j, orig_idx in enumerate(eval_train_indices):
                if orig_idx in member_set:
                    all_memberships[j, i] = True
                        
        norm_mean, norm_std = None, None
        if base_cfg.data.normalization == Normalization.BATCH:
            norm_mean = all_x.mean(dim=(0, 2, 3), keepdim=True)
            norm_std = all_x.std(dim=(0, 2, 3), keepdim=True)
                
        # Evaluate over steps
        steps = list(range(0, 150001, 10000))
        for step in steps:
            step_cache_keys = [f"group_{group_idx}_step_{step}_target_{target_idx}" for target_idx in range(len(run_ids))]
            if all(key in cache for key in step_cache_keys):
                print(f"Group {group_idx} step {step} fully cached. Skipping computation.")
                print(f"\nGroup {group_idx} | Step {step} (from cache)")
                for target_idx, key in enumerate(step_cache_keys):
                    res = cache[key]
                    print(f"  Target {target_idx} (Run {run_ids[target_idx]}) | Tuned a = {res['a']:.1f} | AUC = {res['auc']:.4f}")
                    all_results.append({
                        "group_idx": group_idx,
                        "step": step,
                        "target_idx": target_idx,
                        "run_id": run_ids[target_idx],
                        "a": res["a"],
                        "auc": res["auc"]
                    })
                continue

            print(f"Evaluating group {group_idx} step {step}...")
            all_signals = np.zeros((len(eval_train_indices), len(run_ids)), dtype=np.float32)
            pop_signals = np.zeros((len(pop_indices), len(run_ids)), dtype=np.float32)
            
            # Fetch models and get signals
            valid_runs = 0
            for i, r_id in enumerate(run_ids):
                try:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        model_path = Path(tmpdir) / "model.pth"
                        download_artifact(args.tracking_uri, r_id, f"checkpoints/{step}/model.pth", str(model_path))
                        
                        model = base_cfg.model(input_dim=full_container.input_shape, num_classes=full_container.num_classes)
                        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
                        model.to(device)
                        model.eval()
                        
                        all_signals[:, i] = compute_signals_in_batches(model, all_x, all_y, device, norm_mean, norm_std).numpy()
                        pop_signals[:, i] = compute_signals_in_batches(model, pop_x, pop_y, device, norm_mean, norm_std).numpy()
                        valid_runs += 1
                except Exception as e:
                    pass  # Expected if checkpoint doesn't exist yet
                    
            if valid_runs < 2:
                continue
                
            # Compute InfoRMIA
            print(f"\nGroup {group_idx} | Step {step}")
            num_ref_models = valid_runs - 1
            for target_idx in range(len(run_ids)):
                # Pick an arbitrary reference model to tune `a` that is not the target model
                ref_idx = (target_idx + 1) % len(run_ids)
                
                # Tune optimal a
                a = tune_optimal_a(target_idx, ref_idx, all_signals, pop_signals, all_memberships, num_ref_models)
                
                # Run InfoRMIA
                stats = run_informia(target_idx, all_signals, pop_signals, all_memberships, num_ref_models, a)
                auc = roc_auc_score(all_memberships[:, target_idx], stats)
                
                print(f"  Target {target_idx} (Run {run_ids[target_idx]}) | Tuned a = {a:.1f} | AUC = {auc:.4f}")
                
                cache[step_cache_keys[target_idx]] = {"a": float(a), "auc": float(auc)}
                all_results.append({
                    "group_idx": group_idx,
                    "step": step,
                    "target_idx": target_idx,
                    "run_id": run_ids[target_idx],
                    "a": float(a),
                    "auc": float(auc)
                })
            save_cache()
                
    if all_results:
        new_df = pd.DataFrame(all_results)
        if not existing_df.empty:
            final_df = pd.concat([existing_df, new_df], ignore_index=True)
            final_df = final_df.drop_duplicates(subset=["group_idx", "step", "run_id"], keep="last")
        else:
            final_df = new_df
        final_df.to_parquet(args.output)
        print(f"Saved {len(final_df)} results to {args.output}")

if __name__ == "__main__":
    main()
