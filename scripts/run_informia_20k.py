"""Evaluates InfoRMIA for GROK_MNIST_CE_MLP at step 20k using strictly the implementation from scripts/info_rmia.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Import functions and constants directly from info_rmia.py
from info_rmia import (
    run_informia,
    compute_signals_in_batches,
    get_local_artifact_path,
    get_default_tracking_uri,
    get_default_mlruns_dir,
    NUM_MAX_SAMPLES,
    NUM_MAX_CANARY_SAMPLES,
    DEVICE,
)

from privacy_and_grokking.config import TrainConfig
from privacy_and_grokking.utils.logger import Logger

# Default 6-run paired stratified group for LABEL_NOISE_GROK_MNIST_CE_MLP
DEFAULT_GROUP_ID = "224987"
DEFAULT_RUN_IDS = [
    "1a89fa54721047528a9e6e7eb1762462",  # Target model (model_index: 0)
    "d5c6e9bd1e7f498a8ac703f4a3f090bc",  # Validation model (model_index: 1)
    "75b5d35903bc4c91a986a20b260eb134",  # Reference model 0 (model_index: 2)
    "898309a249f746a994e8e3b856394d27",  # Reference model 1 (model_index: 3)
    "011c029b14294251b04b901453d848cc",  # Reference model 2 (model_index: 4)
    "a21f123a84a44efdb642c24bfb2e22b8",  # Reference model 3 (model_index: 5)
]


def evaluate_informia_step(
    group_id: str = DEFAULT_GROUP_ID,
    run_ids: list[str] = DEFAULT_RUN_IDS,
    step: int = 20000,
    experiment_name: str = "canary-selection",
    mlruns_dir: str | None = None,
    experiment_id: str | None = "788515836770999869",
    seed: int = 42,
) -> dict:
    Logger().setup()
    rng = torch.Generator().manual_seed(seed)

    if mlruns_dir is None:
        tracking_uri = get_default_tracking_uri()
        mlruns_dir = get_default_mlruns_dir(tracking_uri)

    print(f"Device: {DEVICE}")
    print(f"Evaluating Group: '{group_id}' at step {step}")
    print(f"Target Run ID: {run_ids[0]}")
    print(f"Val Run ID:    {run_ids[1]}")
    print(f"Ref Run IDs:   {run_ids[2:]}")

    # 1. Load Configurations
    cfgs: dict[str, TrainConfig] = {}
    for r_id in run_ids:
        local_config_path = get_local_artifact_path(
            experiment_name, r_id, "training_config.json", mlruns_dir, experiment_id
        )
        with open(local_config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        cfgs[r_id] = TrainConfig.model_validate(config_dict)

    target_data_container = cfgs[run_ids[0]].data()
    val_data_container = cfgs[run_ids[1]].data()
    has_canary = bool(
        target_data_container.train_canary
        and target_data_container.test_canary
        and val_data_container.train_canary
        and val_data_container.test_canary
    )

    # 2. Sample Datasets (Exact implementation as in scripts/info_rmia.py)
    num_target_samples = min(NUM_MAX_SAMPLES, len(target_data_container.train), len(val_data_container.train))
    target_in = torch.randperm(len(target_data_container.train), generator=rng)[:num_target_samples]
    target_out = torch.randperm(len(val_data_container.train), generator=rng)[:num_target_samples]
    population_out = torch.randperm(len(target_data_container.test), generator=rng)[:min(NUM_MAX_SAMPLES, len(target_data_container.test))]

    if has_canary:
        num_canary_target = min(NUM_MAX_CANARY_SAMPLES, len(target_data_container.train_canary), len(val_data_container.train_canary))
        target_canary_in = torch.randperm(len(target_data_container.train_canary), generator=rng)[:num_canary_target]
        target_canary_out = torch.randperm(len(val_data_container.train_canary), generator=rng)[:num_canary_target]
        population_canary_out = torch.randperm(len(target_data_container.test_canary), generator=rng)[:min(NUM_MAX_CANARY_SAMPLES, len(target_data_container.test_canary))]

    num_val_samples = min(NUM_MAX_SAMPLES, len(val_data_container.train), len(target_data_container.train))
    val_in = torch.randperm(len(val_data_container.train), generator=rng)[:num_val_samples]
    val_out = torch.randperm(len(target_data_container.train), generator=rng)[:num_val_samples]
    val_population_out = torch.randperm(len(val_data_container.test), generator=rng)[:min(NUM_MAX_SAMPLES, len(val_data_container.test))]

    if has_canary:
        num_canary_val = min(NUM_MAX_CANARY_SAMPLES, len(val_data_container.train_canary), len(target_data_container.train_canary))
        val_canary_in = torch.randperm(len(val_data_container.train_canary), generator=rng)[:num_canary_val]
        val_canary_out = torch.randperm(len(target_data_container.train_canary), generator=rng)[:num_canary_val]
        val_population_canary_out = torch.randperm(len(val_data_container.test_canary), generator=rng)[:min(NUM_MAX_CANARY_SAMPLES, len(val_data_container.test_canary))]

    # 3. Membership Matrices (Exact implementation as in scripts/info_rmia.py)
    ref_train_indices = {}
    ref_canary_indices = {}
    for r_id in run_ids[2:]:
        ref_container = cfgs[r_id].data()
        ref_train_indices[r_id] = set(ref_container.train.indices)
        if has_canary and ref_container.train_canary is not None:
            ref_canary_indices[r_id] = set(ref_container.train_canary.indices)

    # Target memberships
    target_in_indices = torch.tensor(target_data_container.train.indices)[target_in].tolist()
    target_out_indices = torch.tensor(val_data_container.train.indices)[target_out].tolist()

    target_memberships = torch.zeros(len(run_ids) - 1, len(target_in))
    target_memberships[-1, :] = 1.0  # target model is member for target_in
    target_out_memberships = torch.zeros(len(run_ids) - 1, len(target_out))
    target_out_memberships[-1, :] = 0.0  # target model is non-member for target_out

    for idx, r_id in enumerate(run_ids[2:]):
        ref_in = ref_train_indices[r_id]
        target_memberships[idx, :] = torch.tensor(
            [1.0 if val in ref_in else 0.0 for val in target_in_indices], dtype=torch.float
        )
        target_out_memberships[idx, :] = torch.tensor(
            [1.0 if val in ref_in else 0.0 for val in target_out_indices], dtype=torch.float
        )

    if has_canary:
        target_canary_in_indices = torch.tensor(target_data_container.train_canary.indices)[target_canary_in].tolist()
        target_canary_out_indices = torch.tensor(val_data_container.train_canary.indices)[target_canary_out].tolist()

        target_memberships_canary = torch.zeros(len(run_ids) - 1, len(target_canary_in))
        target_memberships_canary[-1, :] = 1.0
        target_out_memberships_canary = torch.zeros(len(run_ids) - 1, len(target_canary_out))
        target_out_memberships_canary[-1, :] = 0.0

        for idx, r_id in enumerate(run_ids[2:]):
            ref_canary_in = ref_canary_indices[r_id]
            target_memberships_canary[idx, :] = torch.tensor(
                [1.0 if val in ref_canary_in else 0.0 for val in target_canary_in_indices], dtype=torch.float
            )
            target_out_memberships_canary[idx, :] = torch.tensor(
                [1.0 if val in ref_canary_in else 0.0 for val in target_canary_out_indices], dtype=torch.float
            )

    # Validation memberships
    val_in_indices = torch.tensor(val_data_container.train.indices)[val_in].tolist()
    val_out_indices = torch.tensor(target_data_container.train.indices)[val_out].tolist()

    val_memberships = torch.zeros(len(run_ids) - 1, len(val_in))
    val_memberships[-1, :] = 1.0
    val_out_memberships = torch.zeros(len(run_ids) - 1, len(val_out))
    val_out_memberships[-1, :] = 0.0

    for idx, r_id in enumerate(run_ids[2:]):
        ref_in = ref_train_indices[r_id]
        val_memberships[idx, :] = torch.tensor(
            [1.0 if val in ref_in else 0.0 for val in val_in_indices], dtype=torch.float
        )
        val_out_memberships[idx, :] = torch.tensor(
            [1.0 if val in ref_in else 0.0 for val in val_out_indices], dtype=torch.float
        )

    if has_canary:
        val_canary_in_indices = torch.tensor(val_data_container.train_canary.indices)[val_canary_in].tolist()
        val_canary_out_indices = torch.tensor(target_data_container.train_canary.indices)[val_canary_out].tolist()

        val_memberships_canary = torch.zeros(len(run_ids) - 1, len(val_canary_in))
        val_memberships_canary[-1, :] = 1.0
        val_out_memberships_canary = torch.zeros(len(run_ids) - 1, len(val_canary_out))
        val_out_memberships_canary[-1, :] = 0.0

        for idx, r_id in enumerate(run_ids[2:]):
            ref_canary_in = ref_canary_indices[r_id]
            val_memberships_canary[idx, :] = torch.tensor(
                [1.0 if val in ref_canary_in else 0.0 for val in val_canary_in_indices], dtype=torch.float
            )
            val_out_memberships_canary[idx, :] = torch.tensor(
                [1.0 if val in ref_canary_in else 0.0 for val in val_canary_out_indices], dtype=torch.float
            )

    # 4. Load Models at Checkpoint Step
    models = {}
    for r_id in run_ids:
        model_path = get_local_artifact_path(
            experiment_name, r_id, f"checkpoints/{step}/model.pth", mlruns_dir, experiment_id
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint at step {step} not found: {model_path}")
        models[r_id] = model_path

    # 5. Inference for Target Data
    print("Computing probabilities on target data...")
    target_in_probs = torch.zeros(len(run_ids) - 1, len(target_in))
    target_out_probs = torch.zeros(len(run_ids) - 1, len(target_out))
    target_population_out_probs = torch.zeros(len(run_ids) - 1, len(population_out))
    if has_canary:
        target_canary_in_probs = torch.zeros(len(run_ids) - 1, len(target_canary_in))
        target_canary_out_probs = torch.zeros(len(run_ids) - 1, len(target_canary_out))
        target_canary_population_out_probs = torch.zeros(len(run_ids) - 1, len(population_canary_out))

    for enum_idx, r_id in enumerate(run_ids):
        if enum_idx == 0:
            idx = -1
        elif enum_idx == 1:
            continue
        else:
            idx = enum_idx - 2

        ref_model = cfgs[r_id].model(
            input_dim=target_data_container.input_shape, num_classes=target_data_container.num_classes
        )
        ref_model.load_state_dict(torch.load(models[r_id], map_location=DEVICE, weights_only=True))
        ref_model.to(DEVICE)
        ref_model.eval()

        norm_mean = target_data_container.normalization.mean if target_data_container.normalization else None
        norm_std = target_data_container.normalization.std if target_data_container.normalization else None

        target_in_probs[idx] = compute_signals_in_batches(
            ref_model, target_data_container.train, target_in, DEVICE, norm_mean, norm_std, batch_size=200
        )
        target_out_probs[idx] = compute_signals_in_batches(
            ref_model, val_data_container.train, target_out, DEVICE, norm_mean, norm_std, batch_size=200
        )
        target_population_out_probs[idx] = compute_signals_in_batches(
            ref_model, target_data_container.test, population_out, DEVICE, norm_mean, norm_std, batch_size=200
        )
        if has_canary:
            target_canary_in_probs[idx] = compute_signals_in_batches(
                ref_model, target_data_container.train_canary, target_canary_in, DEVICE, norm_mean, norm_std, batch_size=200
            )
            target_canary_out_probs[idx] = compute_signals_in_batches(
                ref_model, val_data_container.train_canary, target_canary_out, DEVICE, norm_mean, norm_std, batch_size=200
            )
            target_canary_population_out_probs[idx] = compute_signals_in_batches(
                ref_model, target_data_container.test_canary, population_canary_out, DEVICE, norm_mean, norm_std, batch_size=200
            )

    # 6. Inference for Validation Data
    print("Computing probabilities on validation data...")
    val_in_probs = torch.zeros(len(run_ids) - 1, len(val_in))
    val_out_probs = torch.zeros(len(run_ids) - 1, len(val_out))
    val_population_out_probs = torch.zeros(len(run_ids) - 1, len(val_population_out))
    if has_canary:
        val_canary_in_probs = torch.zeros(len(run_ids) - 1, len(val_canary_in))
        val_canary_out_probs = torch.zeros(len(run_ids) - 1, len(val_canary_out))
        val_canary_population_out_probs = torch.zeros(len(run_ids) - 1, len(val_population_canary_out))

    for enum_idx, r_id in enumerate(run_ids):
        if enum_idx == 0:
            continue
        elif enum_idx == 1:
            idx = -1
        else:
            idx = enum_idx - 2

        ref_model = cfgs[r_id].model(
            input_dim=val_data_container.input_shape, num_classes=val_data_container.num_classes
        )
        ref_model.load_state_dict(torch.load(models[r_id], map_location=DEVICE, weights_only=True))
        ref_model.to(DEVICE)
        ref_model.eval()

        norm_mean = val_data_container.normalization.mean if val_data_container.normalization else None
        norm_std = val_data_container.normalization.std if val_data_container.normalization else None

        val_in_probs[idx] = compute_signals_in_batches(
            ref_model, val_data_container.train, val_in, DEVICE, norm_mean, norm_std, batch_size=200
        )
        val_out_probs[idx] = compute_signals_in_batches(
            ref_model, target_data_container.train, val_out, DEVICE, norm_mean, norm_std, batch_size=200
        )
        val_population_out_probs[idx] = compute_signals_in_batches(
            ref_model, val_data_container.test, val_population_out, DEVICE, norm_mean, norm_std, batch_size=200
        )
        if has_canary:
            val_canary_in_probs[idx] = compute_signals_in_batches(
                ref_model, val_data_container.train_canary, val_canary_in, DEVICE, norm_mean, norm_std, batch_size=200
            )
            val_canary_out_probs[idx] = compute_signals_in_batches(
                ref_model, target_data_container.train_canary, val_canary_out, DEVICE, norm_mean, norm_std, batch_size=200
            )
            val_canary_population_out_probs[idx] = compute_signals_in_batches(
                ref_model, val_data_container.test_canary, val_population_canary_out, DEVICE, norm_mean, norm_std, batch_size=200
            )

    # 7. Tune offline_a on Validation Model (Exact run_informia call)
    print("Tuning offline_a with run_informia on validation model...")
    optimal_a = 0.0
    optimal_auc = -1.0
    val_a_sweep = {}
    for a in torch.arange(0.0, 1.1, 0.1):
        a_float = float(a)
        val_m = run_informia(
            all_signals=torch.cat([val_in_probs, val_out_probs], dim=1),
            population_signals=val_population_out_probs,
            all_memberships=torch.cat([val_memberships, val_out_memberships], dim=1),
            offline_a=a_float,
        )
        val_a_sweep[a_float] = val_m["auc"]
        if val_m["auc"] > optimal_auc:
            optimal_auc = val_m["auc"]
            optimal_a = a_float

    # 8. Compute Target Metrics with run_informia
    target_metrics = run_informia(
        all_signals=torch.cat([target_in_probs, target_out_probs], dim=1),
        population_signals=target_population_out_probs,
        all_memberships=torch.cat([target_memberships, target_out_memberships], dim=1),
        offline_a=optimal_a,
    )
    target_metrics["step"] = step
    target_metrics["optimal_a"] = float(optimal_a)
    target_metrics["optimal_a_auc"] = float(optimal_auc)
    target_metrics["canary"] = False
    target_metrics["group_id"] = group_id
    target_metrics["target_run_id"] = run_ids[0]

    # 9. Canary Tuning and Target Evaluation (if applicable)
    canary_metrics = None
    if has_canary:
        print("Tuning offline_a with run_informia on canary validation data...")
        optimal_canary_a = 0.0
        optimal_canary_auc = -1.0
        val_canary_a_sweep = {}
        for a in torch.arange(0.0, 1.1, 0.1):
            a_float = float(a)
            val_cm = run_informia(
                all_signals=torch.cat([val_canary_in_probs, val_canary_out_probs], dim=1),
                population_signals=val_canary_population_out_probs,
                all_memberships=torch.cat([val_memberships_canary, val_out_memberships_canary], dim=1),
                offline_a=a_float,
            )
            val_canary_a_sweep[a_float] = val_cm["auc"]
            if val_cm["auc"] > optimal_canary_auc:
                optimal_canary_auc = val_cm["auc"]
                optimal_canary_a = a_float

        canary_metrics = run_informia(
            all_signals=torch.cat([target_canary_in_probs, target_canary_out_probs], dim=1),
            population_signals=target_canary_population_out_probs,
            all_memberships=torch.cat([target_memberships_canary, target_out_memberships_canary], dim=1),
            offline_a=optimal_canary_a,
        )
        canary_metrics["step"] = step
        canary_metrics["optimal_a"] = float(optimal_canary_a)
        canary_metrics["optimal_a_auc"] = float(optimal_canary_auc)
        canary_metrics["canary"] = True
        canary_metrics["group_id"] = group_id
        canary_metrics["target_run_id"] = run_ids[0]

    return {
        "step": step,
        "target_non_canary": target_metrics,
        "target_canary": canary_metrics,
        "val_a_sweep_non_canary": val_a_sweep,
        "val_a_sweep_canary": val_canary_a_sweep if has_canary else None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run InfoRMIA at a specific step strictly using info_rmia.py implementation."
    )
    parser.add_argument("--step", type=int, default=20000, help="Step to evaluate (default: 20000)")
    parser.add_argument("--group-id", type=str, default=DEFAULT_GROUP_ID, help="Group ID to evaluate")
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to save results as JSON")
    args = parser.parse_args()

    results = evaluate_informia_step(
        group_id=args.group_id,
        step=args.step,
    )

    print("\n" + "=" * 80)
    print(f"InfoRMIA Evaluation Result at Step {args.step} (Using scripts/info_rmia.py)")
    print("=" * 80)
    nc = results["target_non_canary"]
    print(f"NON-CANARY:")
    print(f"  Optimal a (tuned on Val): {nc['optimal_a']:.1f} (Val AUC: {nc['optimal_a_auc']:.4f})")
    print(f"  Target AUC:               {nc['auc']:.4f}")
    print(f"  TPR @ 1% FPR:             {nc.get('tpr_at_0.01_fpr', 0.0):.4f}")
    print(f"  TPR @ 5% FPR:             {nc.get('tpr_at_0.05_fpr', 0.0):.4f}")
    print(f"  TPR @ 10% FPR:            {nc.get('tpr_at_0.1_fpr', 0.0):.4f}")

    if results["target_canary"]:
        c = results["target_canary"]
        print(f"\nCANARY (Label Noise):")
        print(f"  Optimal a (tuned on Val): {c['optimal_a']:.1f} (Val AUC: {c['optimal_a_auc']:.4f})")
        print(f"  Target AUC:               {c['auc']:.4f}")
        print(f"  TPR @ 1% FPR:             {c.get('tpr_at_0.01_fpr', 0.0):.4f}")
        print(f"  TPR @ 5% FPR:             {c.get('tpr_at_0.05_fpr', 0.0):.4f}")
        print(f"  TPR @ 10% FPR:            {c.get('tpr_at_0.1_fpr', 0.0):.4f}")
    print("=" * 80 + "\n")

    output_path = args.output_json or os.path.join(
        PROJECT_ROOT, "plots", "informia_vs_loss_overlap", f"informia_step_{args.step}_metrics.json"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()
