import argparse
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score
def get_rmia_out_signals(
    ref_signals: np.ndarray,
    ref_memberships: np.ndarray,
    target_model_idx: int,
    num_reference_models: int,
    offline_a: float,
) -> np.ndarray:
    """
    Get average prediction probability of samples over offline reference models (excluding the target model).

    Args:
        ref_signals (np.ndarray): Softmax value of all samples in all reference model.  Shape: (num_samples * num_models)
        ref_memberships (np.ndarray): Membership matrix for all reference models (if a sample is used for training a model).  Shape: (num_samples * num_models)
        target_model_idx (int): Index of the target model to exclude from reference models.
        num_reference_models (Optional[int]): Number of reference models used for the attack. Defaults to half reference models if None.
        offline_a (float): Coefficient offline_a is used to approximate p(x) using P_out in the offline setting.

    Returns:
        np.ndarray: Average softmax value for each sample over OUT reference models.
    """
    # Exclude target model
    mask = np.ones(ref_signals.shape[1], dtype=bool)
    mask[target_model_idx] = False
    
    ref_signals = ref_signals[:, mask]
    ref_memberships = ref_memberships[:, mask]

    non_members = ~ref_memberships
    out_signals = ref_signals * non_members
    # Sort the signals such that only the non-zero signals (out signals) for each sample are kept
    if num_reference_models is None:
        num_reference_models = ref_signals.shape[1]
    if num_reference_models > 1:
        out_signals = -np.sort(-out_signals, axis=1)[:, :num_reference_models]
    else:
        # Derive according to ((1+a)P_out + (1-a))/2 = P(x) = (P_out + P_in)/2
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
    """Compute InfoRMIA membership scores.
    
    Args:
        target_model_idx: Index of the target model.
        all_signals: Softmax scores of all samples.
        population_signals: Softmax scores of population samples.
        all_memberships: Membership matrix for all models.
        num_reference_models: Number of reference models.
        offline_a: Offline correction coefficient used to approximate p(x) using P_out.
        
    Returns:
        Membership inference scores for all samples.
        Larger values indicate higher membership likelihood.
    """ 
    # Target model signals
    target_signals = all_signals[:, target_model_idx]
    out_signals = get_rmia_out_signals(
        all_signals,
        all_memberships,
        target_model_idx,
        num_reference_models,
        offline_a
    )
    mean_out_x = np.mean(out_signals, axis=1) # P_out(x)
    mean_x = (
        ((1 + offline_a) / 2) * mean_out_x
        + ((1 - offline_a) / 2)
    ) # Offline estimation of P(x) according to RMIA
    mean_x = np.clip(mean_x, 1e-12, None)
    # log (p(x|theta) / p(x))
    log_ratio_x = np.log(np.clip(target_signals.ravel() / mean_x, 1e-12, None))
    population_memberships = np.zeros_like(
        population_signals,
        dtype=bool
    ) # population samples are OUT

    z_signals = population_signals[:, target_model_idx]
    z_out_signals = get_rmia_out_signals(
        population_signals,
        population_memberships,
        target_model_idx,
        num_reference_models,
        offline_a
    )
    
    mean_out_z = np.mean(z_out_signals, axis=1)
    mean_z = (
        ((1 + offline_a) / 2) * mean_out_z
        + ((1 - offline_a) / 2)
    )
    mean_z = np.clip(mean_z, 1e-12, None)
    prob_ratio_z = np.clip(z_signals.ravel() / mean_z, 1e-12, None)
    test_statistic = (
        log_ratio_x
        - np.sum(mean_z * np.log(prob_ratio_z))
        / mean_z.sum()
    )

    return test_statistic

def evaluate_informia_from_files(
    step: int,
    target_model_idx: int,
    all_memberships: np.ndarray,
    num_reference_models: int,
    offline_a: float,
    signals_dir: str | Path = "signals"
) -> np.ndarray:
    """Wrapper to load signals from files and compute InfoRMIA."""
    signals_dir = Path(signals_dir)
    
    all_signals_path = signals_dir / f"all_signals_step_{step}.pt"
    pop_signals_path = signals_dir / f"population_signals_step_{step}.pt"
    
    if not all_signals_path.exists() or not pop_signals_path.exists():
        raise FileNotFoundError(f"Signal files for step {step} not found in {signals_dir}")
        
    all_signals = torch.load(all_signals_path, weights_only=True).numpy()
    population_signals = torch.load(pop_signals_path, weights_only=True).numpy()
    
    return run_informia(
        target_model_idx=target_model_idx,
        all_signals=all_signals,
        population_signals=population_signals,
        all_memberships=all_memberships,
        num_reference_models=num_reference_models,
        offline_a=offline_a
    )

def tune_optimal_a(
    step: int,
    target_model_idx: int,
    reference_model_idx: int,
    all_memberships: np.ndarray,
    num_reference_models: int,
    signals_dir: str | Path = "signals"
) -> float:
    """Finds the optimal offline_a parameter between 0 and 1 in steps of 0.1
    using a non-target reference model as a surrogate target."""
    best_a = 0.0
    best_auc = -1.0
    
    print(f"\nTuning optimal 'a' using reference model {reference_model_idx} as surrogate target...")
    for a in np.arange(0.0, 1.1, 0.1):
        stats = evaluate_informia_from_files(
            step=step,
            target_model_idx=reference_model_idx,
            all_memberships=all_memberships,
            num_reference_models=num_reference_models,
            offline_a=float(a),
            signals_dir=signals_dir
        )
        target_memberships = all_memberships[:, reference_model_idx]
        auc = roc_auc_score(target_memberships, stats)
        print(f"  a = {a:.1f} -> AUC = {auc:.4f}")
        
        if auc > best_auc:
            best_auc = auc
            best_a = a
            
    print(f"Optimal 'a' found: {best_a:.1f} with AUC: {best_auc:.4f}\n")
    return float(best_a)


def main():
    parser = argparse.ArgumentParser(description="Evaluate InfoRMIA")
    parser.add_argument("--step", type=int, required=True, help="Checkpoint step to evaluate")
    parser.add_argument("--target-model-idx", type=int, default=0, help="Index of the target model")
    parser.add_argument("--num-reference-models", type=int, required=True, help="Number of reference models")
    parser.add_argument("--offline-a", type=float, default=None, help="Offline estimation coefficient 'a'. Required unless tuning.")
    parser.add_argument("--tune-a-ref-model", type=int, default=None, help="Index of non-target reference model to tune 'a' on. Overrides --offline-a.")
    parser.add_argument("--signals-dir", type=str, default="signals", help="Directory containing signal files")
    parser.add_argument("--memberships-file", type=str, default=None, help="Path to membership matrix (.npy). If omitted, defaults to first half being members for all models.")
    parser.add_argument("--output-file", type=str, default="informia_results.npy", help="Output file for test statistics")
    
    args = parser.parse_args()
    
    signals_dir = Path(args.signals_dir)
    all_signals_path = signals_dir / f"all_signals_step_{args.step}.pt"
    if not all_signals_path.exists():
        raise FileNotFoundError(f"Missing {all_signals_path}")
        
    all_signals = torch.load(all_signals_path, weights_only=True).numpy()
    num_samples, num_models = all_signals.shape
    
    if args.memberships_file:
        all_memberships = np.load(args.memberships_file)
    else:
        print("Warning: No memberships file provided. Assuming first half of samples are members for all models.")
        all_memberships = np.zeros((num_samples, num_models), dtype=bool)
        all_memberships[:num_samples//2, :] = True
        
    if args.tune_a_ref_model is not None:
        if args.tune_a_ref_model == args.target_model_idx:
            raise ValueError("Tuning reference model cannot be the target model itself!")
        args.offline_a = tune_optimal_a(
            step=args.step,
            target_model_idx=args.target_model_idx,
            reference_model_idx=args.tune_a_ref_model,
            all_memberships=all_memberships,
            num_reference_models=args.num_reference_models,
            signals_dir=args.signals_dir
        )
    elif args.offline_a is None:
        raise ValueError("You must provide either --offline-a or --tune-a-ref-model")

    print(f"Evaluating InfoRMIA for step {args.step} on target model {args.target_model_idx} with a={args.offline_a}...")
    stats = evaluate_informia_from_files(
        step=args.step,
        target_model_idx=args.target_model_idx,
        all_memberships=all_memberships,
        num_reference_models=args.num_reference_models,
        offline_a=args.offline_a,
        signals_dir=args.signals_dir
    )
    print("\nTest statistics:")
    print(f"Summary -> Mean: {stats.mean():.4f}, Min: {stats.min():.4f}, Max: {stats.max():.4f}")
    
    target_memberships = all_memberships[:, args.target_model_idx]
    auc = roc_auc_score(target_memberships, stats)
    fpr, tpr, _ = roc_curve(target_memberships, stats)
    
    print("\n--- Metrics ---")
    print(f"AUC: {auc:.4f}")
    
    fpr_targets = [0.001, 0.01, 0.05, 0.1]
    for target in fpr_targets:
        tpr_val = np.interp(target, fpr, tpr)
        print(f"TPR @ {target*100:g}% FPR: {tpr_val:.4f}")
        
    np.save(args.output_file, stats)
    print(f"Saved InfoRMIA test statistics to {args.output_file}")

if __name__ == "__main__":
    main()