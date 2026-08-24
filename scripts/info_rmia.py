from privacy_and_grokking.config import TrainConfig
import argparse
import mlflow
import os
import pandas as pd
import requests
import torch
import json
import hashlib
import numpy as np
from sklearn.metrics import auc, roc_curve

from collections import defaultdict

from privacy_and_grokking.utils.logger import Logger

NUM_MAX_SAMPLES = 500
NUM_MAX_CANARY_SAMPLES = 25
STEP_SIZE = 10_000
MAX_STEPS = 160_000
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "cache"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_rmia_out_signals(
    all_signals: torch.Tensor,
    all_memberships: torch.Tensor,
) -> torch.Tensor:
    """
    Get average prediction probability of samples over offline reference models (excluding the target model).

    Args:
        ref_signals (np.ndarray): Softmax value of all samples in all reference model.  Shape: (num_samples * num_models)
        ref_memberships (np.ndarray): Membership matrix for all reference models (if a sample is used for training a model).  Shape: (num_samples * num_models)
        num_reference_models (Optional[int]): Number of reference models used for the attack. Defaults to half reference models if None.
        offline_a (float): Coefficient offline_a is used to approximate p(x) using P_out in the offline setting.

    Returns:
        np.ndarray: Average softmax value for each sample over OUT reference models.
    """
    # Exclude target model (which is the last one, index -1)
    ref_signals = all_signals[:-1, :]
    ref_memberships = all_memberships[:-1, :]
    non_members = ~ref_memberships.to(torch.bool)
    out_signals = ref_signals * non_members
    return out_signals

def run_informia(
    all_signals: torch.Tensor,
    population_signals: torch.Tensor,
    all_memberships: torch.Tensor,
    offline_a: float
    ) -> torch.Tensor:
    """ Compute InfoRMIA membership scores.
    
    Args:
        all_signals: Softmax scores of all samples.
        population_signals: Softmax scores of population samples.
        all_memberships: Membership matrix for all models.
        offline_a: Offline correction coefficient used to approximate p(x) using P_out.
    
    Returns:
        Membership inference scores for all samples. Larger values indicate higher membership likelihood.
    """
    # Target model signals
    target_signals = all_signals[-1, :] # target/val model is -1
    out_signals = get_rmia_out_signals(all_signals, all_memberships) # (NUM_MODELS - 1, NUM_SAMPLES)
    
    mean_out_x = torch.mean(out_signals, dim=0) # P_out(x) - (NUM_SAMPLES,)
    mean_x = (  ((1 + offline_a) / 2) * mean_out_x + ((1 - offline_a) / 2) ) # Offline estimation of P(x) according to RMIA
    mean_x = torch.clamp(mean_x, min=1e-12)
    
    # log (p(x|theta) / p(x))
    log_ratio_x = torch.log(torch.clamp(target_signals.ravel() / mean_x, min=1e-12))
    population_memberships = torch.zeros_like(  population_signals, dtype=torch.bool, ) # population samples are OUT
    z_signals = population_signals[-1, :] # target/val model is -1
    z_out_signals = get_rmia_out_signals(  population_signals, population_memberships)
    mean_out_z = torch.mean(z_out_signals, dim=0)
    mean_z = (  ((1 + offline_a) / 2) * mean_out_z + ((1 - offline_a) / 2) )
    mean_z = torch.clamp(mean_z, min=1e-12)
    prob_ratio_z = torch.clamp(z_signals.ravel() / mean_z, min=1e-12)
    test_statistic = (  log_ratio_x - torch.sum(mean_z * torch.log(prob_ratio_z)) / mean_z.sum() )

    fpr_list, tpr_list, _ = roc_curve(
        all_memberships[-1, :].cpu().numpy(), test_statistic.cpu().numpy()
    )
    roc_auc = auc(fpr_list, tpr_list)
    metrics = {}
    metrics["auc"] = roc_auc

    for fpr in [0, 0.01, 0.05, 0.1]:
        valid_indices = np.where(fpr_list <= fpr)[0]

        if len(valid_indices) > 0:
            best_index = valid_indices[-1]
            metrics[f"tpr_at_{fpr}_fpr"] = tpr_list[best_index]
        else:
            metrics[f"tpr_at_{fpr}_fpr"] = 0.0

    return metrics

def fetch_json_artifact(tracking_uri: str, run_id: str, artifact_path: str, cache_path: str) -> dict:
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            config_dict = json.load(f)
        return config_dict

    base_uri = tracking_uri.rstrip('/')
    url = f"{base_uri}/get-artifact?path={artifact_path}&run_uuid={run_id}"
    response = requests.get(url)
    response.raise_for_status()
    result = response.json()
    with open(cache_path, "w") as f:
        json.dump(result, f)
    return result

def download_artifact(tracking_uri: str, run_id: str, artifact_path: str, cache_path: str) -> None:
    if os.path.exists(cache_path):
        return
    base_uri = tracking_uri.rstrip('/')
    url = f"{base_uri}/get-artifact?path={artifact_path}&run_uuid={run_id}"
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(cache_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

@torch.no_grad()
def compute_signals_in_batches(model, dataset, indices, device, norm_mean=None, norm_std=None, batch_size=1024):
    subset = torch.utils.data.Subset(dataset, indices.tolist())
    dataloader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False)
    
    if norm_mean is not None:
        norm_mean = torch.tensor(norm_mean, device=device).view(1, -1, 1, 1)
    if norm_std is not None:
        norm_std = torch.tensor(norm_std, device=device).view(1, -1, 1, 1)

    all_probs = []
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        if norm_mean is not None:
            batch_x = (batch_x - norm_mean) / norm_std
        logit = model(batch_x, verbose=False)
        prob = torch.functional.F.softmax(logit, dim=1)
        true_prob = prob.gather(1, batch_y.view(-1, 1)).squeeze(1)
        all_probs.append(true_prob.cpu())
    return torch.cat(all_probs) if all_probs else torch.tensor([])

def handle_group_runs(group_id, run_ids, cache_path, tracking_uri):
    group_hash = hashlib.sha256(str(group_id).encode('utf-8')).hexdigest()[:16]
    group_path = os.path.join(cache_path, group_hash)
    os.makedirs(group_path, exist_ok=True)
    rng = torch.Generator()
    
    if len(run_ids) < 3:
        print(f"Group has only {len(run_ids)} runs, skipping (need at least 3).")
        return False


    cfgs: dict[str, TrainConfig] = {}
    for r_id in run_ids:
        config_path = os.path.join(group_path, r_id, f"train_config.json")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        cfgs[r_id] = TrainConfig.model_validate(fetch_json_artifact(tracking_uri, r_id, "training_config.json", config_path))
    
    # Get Dataset for Target Model
    target_data_container = cfgs[run_ids[0]].data()
    has_canary = target_data_container.train_canary and target_data_container.test_canary
    target_in = torch.randperm(len(target_data_container.train), generator=rng)[:min(NUM_MAX_SAMPLES,len(target_data_container.train))]
    test_perm = torch.randperm(len(target_data_container.test), generator=rng)
    target_out = test_perm[:min(NUM_MAX_SAMPLES,len(target_data_container.test))]
    population_out = test_perm[min(NUM_MAX_SAMPLES,len(target_data_container.test)):min(NUM_MAX_SAMPLES,len(target_data_container.test))+NUM_MAX_SAMPLES]
    if has_canary:
        target_canary_in = torch.randperm(len(target_data_container.canary_train), generator=rng)[:min(NUM_MAX_CANARY_SAMPLES,len(target_data_container.canary_train))]
        test_canary_perm = torch.randperm(len(target_data_container.canary_test), generator=rng)
        target_canary_out = test_canary_perm[:min(NUM_MAX_CANARY_SAMPLES,len(target_data_container.canary_test))]
        population_canary_out = test_canary_perm[min(NUM_MAX_CANARY_SAMPLES,len(target_data_container.canary_test)):min(NUM_MAX_CANARY_SAMPLES,len(target_data_container.canary_test))+NUM_MAX_CANARY_SAMPLES]

    target_in_indices = set(torch.tensor(target_data_container.train.indices)[target_in].tolist())
    if has_canary:
        target_canary_in_indices = set(torch.tensor(target_data_container.canary_train.indices)[target_canary_in].tolist())
    
    target_memberships = torch.zeros(len(run_ids)-1, len(target_in))
    target_memberships[-1, :] = 1.0
    if has_canary:
        target_memberships_canary = torch.zeros(len(run_ids)-1, len(target_canary_in))
        target_memberships_canary[-1, :] = 1.0
    for idx, r_id in enumerate(run_ids[2:]):
        reference_data_container = cfgs[r_id].data()
        ref_in_indices = set(reference_data_container.train.indices)
        target_memberships[idx, :] = torch.tensor(
            [1.0 if val in ref_in_indices else 0.0 for val in torch.tensor(target_data_container.train.indices)[target_in].tolist()],
            dtype=torch.float
        )
        if has_canary:
            ref_canary_in_indices = set(reference_data_container.canary_train.indices)
            target_memberships_canary[idx, :] = torch.tensor(
                [1.0 if val in ref_canary_in_indices else 0.0 for val in torch.tensor(target_data_container.canary_train.indices)[target_canary_in].tolist()],
                dtype=torch.float
            )
        

    print(target_memberships.where(target_memberships == 1.0, 0.0))
    if has_canary:
        print(target_memberships_canary.where(target_memberships_canary == 1.0, 0.0))

    # Get Dataset for Validation Model
    val_data_container = cfgs[run_ids[1]].data()
    val_in = torch.randperm(len(val_data_container.train), generator=rng)[:min(NUM_MAX_SAMPLES,len(val_data_container.train))]
    val_test_perm = torch.randperm(len(val_data_container.test), generator=rng)
    val_out = val_test_perm[:min(NUM_MAX_SAMPLES,len(val_data_container.test))]
    val_population_out = val_test_perm[min(NUM_MAX_SAMPLES,len(val_data_container.test)):min(NUM_MAX_SAMPLES,len(val_data_container.test))+NUM_MAX_SAMPLES]
    if has_canary:
        val_canary_in = torch.randperm(len(val_data_container.canary_train), generator=rng)[:min(NUM_MAX_CANARY_SAMPLES,len(val_data_container.canary_train))]
        val_test_canary_perm = torch.randperm(len(val_data_container.canary_test), generator=rng)
        val_canary_out = val_test_canary_perm[:min(NUM_MAX_CANARY_SAMPLES,len(val_data_container.canary_test))]
        val_population_canary_out = val_test_canary_perm[min(NUM_MAX_CANARY_SAMPLES,len(val_data_container.canary_test)):min(NUM_MAX_CANARY_SAMPLES,len(val_data_container.canary_test))+NUM_MAX_CANARY_SAMPLES]

    val_in_indices = set(torch.tensor(val_data_container.train.indices)[val_in].tolist())
    if has_canary:
        val_canary_in_indices = set(torch.tensor(val_data_container.canary_train.indices)[val_canary_in].tolist())
    
    val_memberships = torch.zeros(len(run_ids)-1, len(val_in))
    val_memberships[-1, :] = 1.0
    if has_canary:
        val_memberships_canary = torch.zeros(len(run_ids)-1, len(val_canary_in))
        val_memberships_canary[-1, :] = 1.0
    for idx, r_id in enumerate(run_ids[2:]):
        reference_data_container = cfgs[r_id].data()
        ref_in_indices = set(reference_data_container.train.indices)
        val_memberships[idx, :] = torch.tensor(
            [1.0 if val in ref_in_indices else 0.0 for val in torch.tensor(val_data_container.train.indices)[val_in].tolist()],
            dtype=torch.float
        )
        if has_canary:
            ref_canary_in_indices = set(reference_data_container.canary_train.indices)
            val_memberships_canary[idx, :] = torch.tensor(
                [1.0 if val in ref_canary_in_indices else 0.0 for val in torch.tensor(val_data_container.canary_train.indices)[val_canary_in].tolist()],
                dtype=torch.float
            )

    metrics = []
    for step in range(0, MAX_STEPS, STEP_SIZE):
        models = {}
        for id in run_ids:
            model_path = os.path.join(group_path, id, f"{step}.pth")
            download_artifact(tracking_uri, id, f"checkpoints/{step}/model.pth", str(model_path))
            models[id] = model_path
        
        # Inference for target model data
        target_in_probs = torch.zeros(len(run_ids) - 1, len(target_in))
        target_out_probs = torch.zeros(len(run_ids) - 1, len(target_out))
        target_population_out_probs = torch.zeros(len(run_ids) - 1, len(population_out))
        if has_canary:
            target_canary_in_probs = torch.zeros(len(run_ids) - 1, len(target_canary_in))
            target_canary_out_probs = torch.zeros(len(run_ids) - 1, len(target_canary_out))
            target_canary_population_out_probs = torch.zeros(len(run_ids) - 1, len(population_canary_out))

        for idx, r_id in enumerate(run_ids[:-1]):
            if idx == 0:
                idx = -1
            elif idx == 1:
                continue
            else:
                idx -= 2

            reference_model_path = models[r_id]
            reference_model_config = cfgs[r_id]
            reference_model = reference_model_config.model(input_dim=target_data_container.input_shape, num_classes=target_data_container.num_classes)
            reference_model.load_state_dict(torch.load(reference_model_path, map_location=DEVICE, weights_only=True))
            reference_model.to(DEVICE)
            reference_model.eval()

            norm_mean = target_data_container.normalization.mean if target_data_container.normalization else None
            norm_std = target_data_container.normalization.std if target_data_container.normalization else None
            # Inference for reference model data
            target_in_probs[idx] = compute_signals_in_batches(
                reference_model, target_data_container.train, target_in, DEVICE, norm_mean, norm_std, batch_size=200
            )
            target_out_probs[idx] = compute_signals_in_batches(
                reference_model, target_data_container.test, target_out, DEVICE, norm_mean, norm_std, batch_size=200
            )
            target_population_out_probs[idx] = compute_signals_in_batches(
                reference_model, target_data_container.test, population_out, DEVICE, norm_mean, norm_std, batch_size=200
            )
            if has_canary:
                target_canary_in_probs[idx] = compute_signals_in_batches(
                    reference_model, target_data_container.canary_train, target_canary_in, DEVICE, norm_mean, norm_std, batch_size=200
                )
                target_canary_out_probs[idx] = compute_signals_in_batches(
                    reference_model, target_data_container.canary_test, target_canary_out, DEVICE, norm_mean, norm_std, batch_size=200
                )
                target_canary_population_out_probs[idx] = compute_signals_in_batches(
                    reference_model, target_data_container.canary_test, population_canary_out, DEVICE, norm_mean, norm_std, batch_size=200
                )

        # Inference for validation model data
        val_in_probs = torch.zeros(len(run_ids) - 1, len(val_in))
        val_out_probs = torch.zeros(len(run_ids) - 1, len(val_out))
        val_population_out_probs = torch.zeros(len(run_ids) - 1, len(val_population_out))
        if has_canary:
            val_canary_in_probs = torch.zeros(len(run_ids) - 1, len(val_canary_in))
            val_canary_out_probs = torch.zeros(len(run_ids) - 1, len(val_canary_out))
            val_canary_population_out_probs = torch.zeros(len(run_ids) - 1, len(val_population_canary_out))

        for idx, r_id in enumerate(run_ids[:-1]):
            if idx == 0:
                continue
            elif idx == 1:
                idx = -1
            else:
                idx -= 2

            reference_model_path = models[r_id]
            reference_model_config = cfgs[r_id]
            reference_model = reference_model_config.model(input_dim=val_data_container.input_shape, num_classes=val_data_container.num_classes)
            reference_model.load_state_dict(torch.load(reference_model_path, map_location=DEVICE, weights_only=True))
            reference_model.to(DEVICE)
            reference_model.eval()

            norm_mean = val_data_container.normalization.mean if val_data_container.normalization else None
            norm_std = val_data_container.normalization.std if val_data_container.normalization else None
            # Inference for reference model data
            val_in_probs[idx] = compute_signals_in_batches(
                reference_model, val_data_container.train, val_in, DEVICE, norm_mean, norm_std, batch_size=200
            )
            val_out_probs[idx] = compute_signals_in_batches(
                reference_model, val_data_container.test, val_out, DEVICE, norm_mean, norm_std, batch_size=200
            )
            val_population_out_probs[idx] = compute_signals_in_batches(
                reference_model, val_data_container.test, val_population_out, DEVICE, norm_mean, norm_std, batch_size=200
            )
            if has_canary:
                val_canary_in_probs[idx] = compute_signals_in_batches(
                    reference_model, val_data_container.canary_train, val_canary_in, DEVICE, norm_mean, norm_std, batch_size=200
                )
                val_canary_out_probs[idx] = compute_signals_in_batches(
                    reference_model, val_data_container.canary_test, val_canary_out, DEVICE, norm_mean, norm_std, batch_size=200
                )
                val_canary_population_out_probs[idx] = compute_signals_in_batches(
                    reference_model, val_data_container.canary_test, val_population_canary_out, DEVICE, norm_mean, norm_std, batch_size=200
                )

        # Get optimal a
        optimal_a = 0
        optimal_auc = -1
        for a in torch.arange(0.0, 1.1, 0.1):
            # Compute target privacy
            val_metrics = run_informia(
                all_signals=torch.cat([val_in_probs, val_out_probs], dim=1),
                population_signals=val_population_out_probs,
                all_memberships=torch.cat([val_memberships, torch.zeros_like(val_out_probs)], dim=1),
                offline_a=a
            )
            if val_metrics["auc"] > optimal_auc:
                optimal_auc = val_metrics["auc"]
                optimal_a = a

        # Compute target utiltiy
        target_metrics = run_informia(
            all_signals=torch.cat([target_in_probs, target_out_probs], dim=1),
            population_signals=target_population_out_probs,
            all_memberships=torch.cat([target_memberships, torch.zeros_like(target_out_probs)], dim=1),
            offline_a=optimal_a
        )
        target_metrics["step"] = step
        target_metrics["optimal_a"] = optimal_a
        target_metrics["optimal_a_auc"] = optimal_auc
        target_metrics["canary"] = False

        metrics.append(target_metrics)
        

        # Compute target utility
        if has_canary:
            optimal_canary_a = 0
            optimal_canary_auc = -1
            for a in torch.arange(0.0, 1.1, 0.1):
                # Compute target privacy
                val_metrics = run_informia(
                    all_signals=torch.cat([val_canary_in_probs, val_canary_out_probs], dim=1),
                    population_signals=val_canary_population_out_probs,
                    all_memberships=torch.cat([val_memberships_canary, torch.zeros_like(val_canary_out_probs)], dim=1),
                    offline_a=a
                )
                if val_metrics["auc"] > optimal_canary_auc:
                    optimal_canary_auc = val_metrics["auc"]
                    optimal_canary_a = a

            target_canary_metrics = run_informia(
                all_signals=torch.cat([target_canary_in_probs, target_canary_out_probs], dim=1),
                population_signals=target_canary_population_out_probs,
                all_memberships=torch.cat([target_memberships_canary, torch.zeros_like(target_canary_out_probs)], dim=1),
                offline_a=optimal_a
            )

            target_canary_metrics["step"] = step
            target_canary_metrics["optimal_a"] = optimal_canary_a
            target_canary_metrics["optimal_a_auc"] = optimal_canary_auc
            target_canary_metrics["canary"] = True

            metrics.append(target_canary_metrics)

    df = pd.DataFrame(metrics)
    df.to_csv(os.path.join(cache_path, "results.csv"), index=False)

def get_runs(args, existing_df):
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
        return False

    def get_ignored_params(run):
        params = {k: v for k, v in run.items() if isinstance(k, str) and k.startswith('params.')}
        ignored = {'params.seed', 'params.data.seed', 'params.data.mask.seed', 'params.data.mask.model_index'}
        return {k: v for k, v in params.items() if k not in ignored}

    groups = {}
    for idx, run in runs_df.iterrows():
        ignored_params = get_ignored_params(run)
        # Sort keys to ensure deterministic tuple creation
        key = tuple(sorted(ignored_params.items()))
        if key not in groups:
            groups[key] = []
        groups[key].append(run['run_id'])
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--model-name", required=False, default=None, help="Optional model name to filter runs. If omitted, evaluates all models in the experiment.")
    parser.add_argument("--tracking-uri", default="http://localhost:5051")
    parser.add_argument("--num-samples", type=int, default=500, help="Number of samples to evaluate on (per member/non-member class)")
    args = parser.parse_args()

    cache_path = os.path.join(CACHE_DIR, args.experiment_name)
        
    Logger().setup()
    mlflow.set_tracking_uri(args.tracking_uri)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_path = os.path.join(cache_path, "informia_results.parquet")
    if os.path.exists(output_path):
        existing_df = pd.read_parquet(output_path)
    else:
        existing_df = pd.DataFrame()

    groups = get_runs(args, existing_df)
    if not groups:
        print("No groups")
        return

    for group_id, run_ids in groups.items():
        handle_group_runs(group_id, run_ids, cache_path, args.tracking_uri)



if __name__ == "__main__":
    main()
