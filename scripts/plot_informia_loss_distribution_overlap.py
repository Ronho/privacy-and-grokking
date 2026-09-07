import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.metrics import auc, roc_curve
import torch
import torch.nn.functional as F

from privacy_and_grokking.config import TrainConfig
from privacy_and_grokking.utils.logger import Logger

SCRIPT_DIR = Path(__file__).parent.resolve()
WORKSPACE_ROOT = SCRIPT_DIR.parent
DEFAULT_MLRUNS_DIR = WORKSPACE_ROOT / "cache" / "mlruns"
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "plots" / "informia_vs_loss_overlap"

# Canonical 6 runs for LABEL_NOISE_GROK_MNIST_CE_MLP in canary-selection
DEFAULT_RUN_IDS = [
    "1a89fa54721047528a9e6e7eb1762462",  # model_index 0 (Target)
    "d5c6e9bd1e7f498a8ac703f4a3f090bc",  # model_index 1 (Validation)
    "75b5d35903bc4c91a986a20b260eb134",  # model_index 2 (Ref 1)
    "898309a249f746a994e8e3b856394d27",  # model_index 3 (Ref 2)
    "011c029b14294251b04b901453d848cc",  # model_index 4 (Ref 3)
    "a21f123a84a44efdb642c24bfb2e22b8",  # model_index 5 (Ref 4)
]


def compute_distribution_overlap(x1: np.ndarray, x2: np.ndarray, num_bins: int = 100) -> float:
    """Computes Overlapping Coefficient (OVL) via histogram intersection:
    OVL = sum(min(hist1, hist2) * bin_width) in [0, 1].
    0 = complete separation, 1 = identical distributions.
    """
    if len(x1) == 0 or len(x2) == 0:
        return 0.0
    val_min = min(np.min(x1), np.min(x2))
    val_max = max(np.max(x1), np.max(x2))
    if np.isclose(val_min, val_max):
        return 1.0

    bins = np.linspace(val_min, val_max, num_bins + 1)
    h1, _ = np.histogram(x1, bins=bins, density=True)
    h2, _ = np.histogram(x2, bins=bins, density=True)
    bin_widths = np.diff(bins)
    overlap = float(np.sum(np.minimum(h1, h2) * bin_widths))
    return min(1.0, max(0.0, overlap))


def plot_hist_and_kde(
    ax: plt.Axes,
    in_data: np.ndarray,
    out_data: np.ndarray,
    num_bins: int,
    in_color: str = "#2563eb",
    out_color: str = "#ef4444",
    in_label: str = "Train (IN)",
    out_label: str = "Test (OUT)",
    fixed_range: tuple[float, float] | None = None,
    clip_percentiles: tuple[float, float] = (0.0, 99.5),
    show_kde: bool = True,
):
    """Plots high-resolution histograms, fitted smooth KDEs, and shades the overlap."""
    if fixed_range is not None:
        val_min, val_max = fixed_range
    else:
        val_min = min(float(np.percentile(in_data, clip_percentiles[0])), float(np.percentile(out_data, clip_percentiles[0])))
        val_max = max(float(np.percentile(in_data, clip_percentiles[1])), float(np.percentile(out_data, clip_percentiles[1])))
    if np.isclose(val_min, val_max):
        val_max = val_min + 1.0

    bins = np.linspace(val_min, val_max, num_bins + 1)

    # High-resolution stepfilled histograms
    ax.hist(
        in_data, bins=bins, density=True, alpha=0.35, color=in_color, label=in_label, edgecolor=in_color, linewidth=0.5
    )
    ax.hist(
        out_data, bins=bins, density=True, alpha=0.35, color=out_color, label=out_label, edgecolor=out_color, linewidth=0.5
    )

    if show_kde and len(in_data) > 3 and len(out_data) > 3:
        try:
            kde_x = np.linspace(val_min, val_max, 400)
            kde_in = gaussian_kde(in_data, bw_method="scott")(kde_x)
            kde_out = gaussian_kde(out_data, bw_method="scott")(kde_x)
            ax.plot(kde_x, kde_in, color=in_color, linewidth=1.8)
            ax.plot(kde_x, kde_out, color=out_color, linewidth=1.8)

            overlap_y = np.minimum(kde_in, kde_out)
            ax.fill_between(kde_x, overlap_y, color="#4b5563", alpha=0.22, hatch="//", label="Overlap (KDE)")
        except Exception:
            pass


def compute_auc(in_scores: np.ndarray, out_scores: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Computes ROC AUC given member (in) and non-member (out) scores."""
    labels = np.concatenate([np.ones_like(in_scores), np.zeros_like(out_scores)])
    scores = np.concatenate([in_scores, out_scores])
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = float(auc(fpr, tpr))
    return roc_auc, fpr, tpr


import sys
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from info_rmia import get_rmia_mean_out_signals


def compute_informia_test_statistic(
    all_signals: torch.Tensor,
    population_signals: torch.Tensor,
    all_memberships: torch.Tensor,
    offline_a: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes InfoRMIA test statistic, numerator, and denominator for all samples.
    Returns: (test_statistic, target_signals, mean_x)
    """
    target_signals = all_signals[-1, :]
    mean_out_x = get_rmia_mean_out_signals(all_signals, all_memberships)

    mean_x = ((1.0 + offline_a) / 2.0) * mean_out_x + ((1.0 - offline_a) / 2.0)
    mean_x = torch.clamp(mean_x, min=1e-12)

    log_ratio_x = torch.log(torch.clamp(target_signals.ravel() / mean_x, min=1e-12))

    if population_signals.numel() == 0:
        expectation = 0.0
    else:
        population_memberships = torch.zeros_like(population_signals, dtype=torch.bool)
        z_signals = population_signals[-1, :]
        mean_out_z = get_rmia_mean_out_signals(population_signals, population_memberships)

        mean_z = ((1.0 + offline_a) / 2.0) * mean_out_z + ((1.0 - offline_a) / 2.0)
        mean_z = torch.clamp(mean_z, min=1e-12)
        prob_ratio_z = torch.clamp(z_signals.ravel() / mean_z, min=1e-12)
        expectation = torch.sum(mean_z * torch.log(prob_ratio_z)) / mean_z.sum()

    test_statistic = log_ratio_x - expectation
    return test_statistic, target_signals, mean_x


@torch.no_grad()
def compute_signals_and_losses_in_batches(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    indices: torch.Tensor,
    device: str,
    norm_mean: list[float] | None = None,
    norm_std: list[float] | None = None,
    batch_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes true-class probabilities p(y|x) and Cross-Entropy losses."""
    subset = torch.utils.data.Subset(dataset, indices.tolist())
    dataloader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False)

    mean_t = torch.tensor(norm_mean, device=device).view(1, -1, 1, 1) if norm_mean else None
    std_t = torch.tensor(norm_std, device=device).view(1, -1, 1, 1) if norm_std else None

    ce_criterion = torch.nn.CrossEntropyLoss(reduction="none")
    all_probs = []
    all_losses = []

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        if mean_t is not None and std_t is not None:
            batch_x = (batch_x - mean_t) / std_t
        logits = model(batch_x, verbose=False)
        probs = F.softmax(logits, dim=1)
        true_prob = probs.gather(1, batch_y.view(-1, 1)).squeeze(1)
        loss = ce_criterion(logits, batch_y)

        all_probs.append(true_prob.cpu())
        all_losses.append(loss.cpu())

    probs_cat = torch.cat(all_probs) if all_probs else torch.tensor([])
    losses_cat = torch.cat(all_losses) if all_losses else torch.tensor([])
    return probs_cat, losses_cat


def resolve_artifact_path(mlruns_dir: Path, exp_id: str, run_id: str, subpath: str) -> Path:
    candidates = [
        mlruns_dir / "canary-selection" / run_id / "artifacts" / subpath,
        mlruns_dir / exp_id / run_id / "artifacts" / subpath,
        mlruns_dir / run_id / "artifacts" / subpath,
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(description="Analyze InfoRMIA vs Loss attack distributions and overlap.")
    parser.add_argument("--step", type=int, default=20000, help="Checkpoint step to evaluate (default: 20000)")
    parser.add_argument("--mlruns-dir", type=str, default=str(DEFAULT_MLRUNS_DIR), help="Path to mlruns directory")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Directory to save plots")
    parser.add_argument("--experiment-id", type=str, default="788515836770999869", help="Experiment ID")
    parser.add_argument("--num-samples", type=int, default=500, help="Max non-canary samples (matching info_rmia.py)")
    parser.add_argument("--num-canary-samples", type=int, default=25, help="Max canary samples (matching info_rmia.py)")
    parser.add_argument("--num-bins", type=int, default=80, help="Number of bins for high-resolution histograms (default: 80)")
    parser.add_argument("--use-full-canaries", action="store_true", help="Evaluate all available canaries")
    args = parser.parse_args()

    Logger().setup()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    mlruns_dir = Path(args.mlruns_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_ids = DEFAULT_RUN_IDS
    print(f"Target Run ID: {run_ids[0]}")
    print(f"Val Run ID:    {run_ids[1]}")
    print(f"Ref Run IDs:   {run_ids[2:]}")

    # 1. Load Configurations
    cfgs: dict[str, TrainConfig] = {}
    for r_id in run_ids:
        cfg_path = resolve_artifact_path(mlruns_dir, args.experiment_id, r_id, "training_config.json")
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfgs[r_id] = TrainConfig.model_validate(json.load(f))

    # 2. Setup Data Containers & Sample Indices
    rng = torch.Generator().manual_seed(42)
    target_dc = cfgs[run_ids[0]].data()
    val_dc = cfgs[run_ids[1]].data()

    def get_splits(num_samples: int, max_samples: int) -> tuple[int, int]:
        if num_samples >= 2 * max_samples:
            return max_samples, max_samples
        return num_samples // 2, num_samples - (num_samples // 2)

    # Target indices
    t_train_len = len(target_dc.train)
    t_test_len = len(target_dc.test)
    num_t_in = min(args.num_samples, t_train_len)
    target_in = torch.randperm(t_train_len, generator=rng)[:num_t_in]

    t_test_perm = torch.randperm(t_test_len, generator=rng)
    out_size, pop_size = get_splits(t_test_len, args.num_samples)
    target_out = t_test_perm[:out_size]
    target_pop_out = t_test_perm[out_size : out_size + pop_size]

    # Canaries indices
    has_canary = target_dc.train_canary is not None and target_dc.test_canary is not None
    if not has_canary:
        raise ValueError("Target run does not contain canaries!")

    c_train_len = len(target_dc.train_canary)
    c_test_len = len(target_dc.test_canary)
    c_max = c_train_len if args.use_full_canaries else args.num_canary_samples
    target_c_in = torch.randperm(c_train_len, generator=rng)[:min(c_max, c_train_len)]

    t_c_test_perm = torch.randperm(c_test_len, generator=rng)
    c_out_size, c_pop_size = get_splits(c_test_len, c_max)
    target_c_out = t_c_test_perm[:c_out_size]
    target_c_pop_out = t_c_test_perm[c_out_size : c_out_size + c_pop_size]

    # Memberships matrix: rows = [ref_0, ..., ref_k, target_model]
    target_memberships = torch.zeros(len(run_ids) - 1, len(target_in))
    target_memberships[-1, :] = 1.0
    target_c_memberships = torch.zeros(len(run_ids) - 1, len(target_c_in))
    target_c_memberships[-1, :] = 1.0

    target_train_indices_raw = torch.tensor(target_dc.train.indices)[target_in].tolist()
    target_c_indices_raw = torch.tensor(target_dc.train_canary.indices)[target_c_in].tolist()

    for idx, r_id in enumerate(run_ids[2:]):
        ref_dc = cfgs[r_id].data()
        ref_in_set = set(ref_dc.train.indices)
        target_memberships[idx, :] = torch.tensor(
            [1.0 if v in ref_in_set else 0.0 for v in target_train_indices_raw], dtype=torch.float
        )
        ref_c_set = set(ref_dc.train_canary.indices)
        target_c_memberships[idx, :] = torch.tensor(
            [1.0 if v in ref_c_set else 0.0 for v in target_c_indices_raw], dtype=torch.float
        )

    # Validation indices & memberships
    v_train_len = len(val_dc.train)
    v_test_len = len(val_dc.test)
    num_v_in = min(args.num_samples, v_train_len)
    val_in = torch.randperm(v_train_len, generator=rng)[:num_v_in]

    v_test_perm = torch.randperm(v_test_len, generator=rng)
    v_out_size, v_pop_size = get_splits(v_test_len, args.num_samples)
    val_out = v_test_perm[:v_out_size]
    val_pop_out = v_test_perm[v_out_size : v_out_size + v_pop_size]

    val_c_train_len = len(val_dc.train_canary)
    val_c_test_len = len(val_dc.test_canary)
    val_c_in = torch.randperm(val_c_train_len, generator=rng)[:min(c_max, val_c_train_len)]

    v_c_test_perm = torch.randperm(val_c_test_len, generator=rng)
    vc_out_size, vc_pop_size = get_splits(val_c_test_len, c_max)
    val_c_out = v_c_test_perm[:vc_out_size]
    val_c_pop_out = v_c_test_perm[vc_out_size : vc_out_size + vc_pop_size]

    val_memberships = torch.zeros(len(run_ids) - 1, len(val_in))
    val_memberships[-1, :] = 1.0
    val_c_memberships = torch.zeros(len(run_ids) - 1, len(val_c_in))
    val_c_memberships[-1, :] = 1.0

    val_train_indices_raw = torch.tensor(val_dc.train.indices)[val_in].tolist()
    val_c_indices_raw = torch.tensor(val_dc.train_canary.indices)[val_c_in].tolist()

    for idx, r_id in enumerate(run_ids[2:]):
        ref_dc = cfgs[r_id].data()
        ref_in_set = set(ref_dc.train.indices)
        val_memberships[idx, :] = torch.tensor(
            [1.0 if v in ref_in_set else 0.0 for v in val_train_indices_raw], dtype=torch.float
        )
        ref_c_set = set(ref_dc.train_canary.indices)
        val_c_memberships[idx, :] = torch.tensor(
            [1.0 if v in ref_c_set else 0.0 for v in val_c_indices_raw], dtype=torch.float
        )

    # 3. Load Models at Step
    print(f"\nLoading checkpoints at step {args.step}...")
    models = {}
    for r_id in run_ids:
        ckpt_path = resolve_artifact_path(mlruns_dir, args.experiment_id, r_id, f"checkpoints/{args.step}/model.pth")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        cfg = cfgs[r_id]
        m = cfg.model(input_dim=target_dc.input_shape, num_classes=target_dc.num_classes)
        m.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        m.to(device)
        m.eval()
        models[r_id] = m

    # 4. Compute Signals & Losses on Target Model
    print("Computing target model signals and losses...")
    norm_mean = target_dc.normalization.mean if target_dc.normalization else None
    norm_std = target_dc.normalization.std if target_dc.normalization else None

    # Target model is run_ids[0]
    target_model = models[run_ids[0]]
    t_in_probs, t_in_losses = compute_signals_and_losses_in_batches(
        target_model, target_dc.train, target_in, device, norm_mean, norm_std
    )
    t_out_probs, t_out_losses = compute_signals_and_losses_in_batches(
        target_model, target_dc.test, target_out, device, norm_mean, norm_std
    )
    t_c_in_probs, t_c_in_losses = compute_signals_and_losses_in_batches(
        target_model, target_dc.train_canary, target_c_in, device, norm_mean, norm_std
    )
    t_c_out_probs, t_c_out_losses = compute_signals_and_losses_in_batches(
        target_model, target_dc.test_canary, target_c_out, device, norm_mean, norm_std
    )

    # 5. Compute Signals for All Reference Models
    print("Computing reference model predictions...")
    t_in_signals_all = torch.zeros(len(run_ids) - 1, len(target_in))
    t_out_signals_all = torch.zeros(len(run_ids) - 1, len(target_out))
    t_pop_signals_all = torch.zeros(len(run_ids) - 1, len(target_pop_out))

    t_c_in_signals_all = torch.zeros(len(run_ids) - 1, len(target_c_in))
    t_c_out_signals_all = torch.zeros(len(run_ids) - 1, len(target_c_out))
    t_c_pop_signals_all = torch.zeros(len(run_ids) - 1, len(target_c_pop_out))

    for enum_idx, r_id in enumerate(run_ids):
        if enum_idx == 1:  # skip validation model for target inference
            continue
        idx = -1 if enum_idx == 0 else enum_idx - 2
        m = models[r_id]

        p_in, _ = compute_signals_and_losses_in_batches(m, target_dc.train, target_in, device, norm_mean, norm_std)
        p_out, _ = compute_signals_and_losses_in_batches(m, target_dc.test, target_out, device, norm_mean, norm_std)
        p_pop, _ = compute_signals_and_losses_in_batches(m, target_dc.test, target_pop_out, device, norm_mean, norm_std)
        t_in_signals_all[idx] = p_in
        t_out_signals_all[idx] = p_out
        t_pop_signals_all[idx] = p_pop

        p_cin, _ = compute_signals_and_losses_in_batches(m, target_dc.train_canary, target_c_in, device, norm_mean, norm_std)
        p_cout, _ = compute_signals_and_losses_in_batches(m, target_dc.test_canary, target_c_out, device, norm_mean, norm_std)
        p_cpop, _ = compute_signals_and_losses_in_batches(m, target_dc.test_canary, target_c_pop_out, device, norm_mean, norm_std)
        t_c_in_signals_all[idx] = p_cin
        t_c_out_signals_all[idx] = p_cout
        t_c_pop_signals_all[idx] = p_cpop

    # Compute Signals for Validation Model (used to tune a)
    print("Computing validation model predictions for tuning 'a'...")
    v_norm_mean = val_dc.normalization.mean if val_dc.normalization else None
    v_norm_std = val_dc.normalization.std if val_dc.normalization else None

    v_in_signals_all = torch.zeros(len(run_ids) - 1, len(val_in))
    v_out_signals_all = torch.zeros(len(run_ids) - 1, len(val_out))
    v_pop_signals_all = torch.zeros(len(run_ids) - 1, len(val_pop_out))

    v_c_in_signals_all = torch.zeros(len(run_ids) - 1, len(val_c_in))
    v_c_out_signals_all = torch.zeros(len(run_ids) - 1, len(val_c_out))
    v_c_pop_signals_all = torch.zeros(len(run_ids) - 1, len(val_c_pop_out))

    for enum_idx, r_id in enumerate(run_ids):
        if enum_idx == 0:  # skip target model for validation inference
            continue
        idx = -1 if enum_idx == 1 else enum_idx - 2
        m = models[r_id]

        p_in, _ = compute_signals_and_losses_in_batches(m, val_dc.train, val_in, device, v_norm_mean, v_norm_std)
        p_out, _ = compute_signals_and_losses_in_batches(m, val_dc.test, val_out, device, v_norm_mean, v_norm_std)
        p_pop, _ = compute_signals_and_losses_in_batches(m, val_dc.test, val_pop_out, device, v_norm_mean, v_norm_std)
        v_in_signals_all[idx] = p_in
        v_out_signals_all[idx] = p_out
        v_pop_signals_all[idx] = p_pop

        p_cin, _ = compute_signals_and_losses_in_batches(m, val_dc.train_canary, val_c_in, device, v_norm_mean, v_norm_std)
        p_cout, _ = compute_signals_and_losses_in_batches(m, val_dc.test_canary, val_c_out, device, v_norm_mean, v_norm_std)
        p_cpop, _ = compute_signals_and_losses_in_batches(m, val_dc.test_canary, val_c_pop_out, device, v_norm_mean, v_norm_std)
        v_c_in_signals_all[idx] = p_cin
        v_c_out_signals_all[idx] = p_cout
        v_c_pop_signals_all[idx] = p_cpop

    # 6. Tune Optimal a on Validation Model
    print("Tuning offline_a on validation model...")
    val_all_signals = torch.cat([v_in_signals_all, v_out_signals_all], dim=1)
    val_all_mem = torch.cat([val_memberships, torch.zeros_like(v_out_signals_all)], dim=1)

    opt_a_non_canary = 0.0
    opt_auc_non_canary = -1.0
    for a_cand in np.arange(0.0, 1.05, 0.1):
        stats, _, _ = compute_informia_test_statistic(
            all_signals=val_all_signals,
            population_signals=v_pop_signals_all,
            all_memberships=val_all_mem,
            offline_a=float(a_cand),
        )
        auc_cand, _, _ = compute_auc(stats[:len(val_in)].numpy(), stats[len(val_in):].numpy())
        if auc_cand > opt_auc_non_canary:
            opt_auc_non_canary = auc_cand
            opt_a_non_canary = float(a_cand)

    val_c_all_signals = torch.cat([v_c_in_signals_all, v_c_out_signals_all], dim=1)
    val_c_all_mem = torch.cat([val_c_memberships, torch.zeros_like(v_c_out_signals_all)], dim=1)

    opt_a_canary = 0.0
    opt_auc_canary = -1.0
    for a_cand in np.arange(0.0, 1.05, 0.1):
        stats, _, _ = compute_informia_test_statistic(
            all_signals=val_c_all_signals,
            population_signals=v_c_pop_signals_all,
            all_memberships=val_c_all_mem,
            offline_a=float(a_cand),
        )
        auc_cand, _, _ = compute_auc(stats[:len(val_c_in)].numpy(), stats[len(val_c_in):].numpy())
        if auc_cand > opt_auc_canary:
            opt_auc_canary = auc_cand
            opt_a_canary = float(a_cand)

    print(f"Optimal 'a' Non-Canary: {opt_a_non_canary:.1f} (Val AUC: {opt_auc_non_canary:.4f})")
    print(f"Optimal 'a' Canary:     {opt_a_canary:.1f} (Val AUC: {opt_auc_canary:.4f})")

    # 7. Compute Loss-based Attack Metrics & Overlap
    # Loss Attack score = -CE_loss = log(prob)
    t_in_loss_scores = -t_in_losses.numpy()
    t_out_loss_scores = -t_out_losses.numpy()
    auc_loss_nc, fpr_loss_nc, tpr_loss_nc = compute_auc(t_in_loss_scores, t_out_loss_scores)
    ovl_loss_nc = compute_distribution_overlap(t_in_losses.numpy(), t_out_losses.numpy())
    ovl_score_nc = compute_distribution_overlap(t_in_loss_scores, t_out_loss_scores)

    t_c_in_loss_scores = -t_c_in_losses.numpy()
    t_c_out_loss_scores = -t_c_out_losses.numpy()
    auc_loss_canary, fpr_loss_canary, tpr_loss_canary = compute_auc(t_c_in_loss_scores, t_c_out_loss_scores)
    ovl_loss_canary = compute_distribution_overlap(t_c_in_losses.numpy(), t_c_out_losses.numpy())
    ovl_score_canary = compute_distribution_overlap(t_c_in_loss_scores, t_c_out_loss_scores)

    # 8. Compute InfoRMIA across sweep of 'a' values
    target_all_signals = torch.cat([t_in_signals_all, t_out_signals_all], dim=1)
    target_all_mem = torch.cat([target_memberships, torch.zeros_like(t_out_signals_all)], dim=1)

    target_c_all_signals = torch.cat([t_c_in_signals_all, t_c_out_signals_all], dim=1)
    target_c_all_mem = torch.cat([target_c_memberships, torch.zeros_like(t_c_out_signals_all)], dim=1)

    a_sweep = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    informia_results = {"non_canary": {}, "canary": {}}

    for a_val in a_sweep:
        # Non-canary
        stats_nc, num_nc, den_nc = compute_informia_test_statistic(
            target_all_signals, t_pop_signals_all, target_all_mem, a_val
        )
        in_stats_nc = stats_nc[:len(target_in)].numpy()
        out_stats_nc = stats_nc[len(target_in):].numpy()
        auc_nc, _, _ = compute_auc(in_stats_nc, out_stats_nc)
        ovl_nc = compute_distribution_overlap(in_stats_nc, out_stats_nc)

        informia_results["non_canary"][a_val] = {
            "auc": auc_nc,
            "ovl": ovl_nc,
            "in_scores": in_stats_nc,
            "out_scores": out_stats_nc,
            "num": num_nc.numpy(),
            "den": den_nc.numpy(),
        }

        # Canary
        stats_c, num_c, den_c = compute_informia_test_statistic(
            target_c_all_signals, t_c_pop_signals_all, target_c_all_mem, a_val
        )
        in_stats_c = stats_c[:len(target_c_in)].numpy()
        out_stats_c = stats_c[len(target_c_in):].numpy()
        auc_c, _, _ = compute_auc(in_stats_c, out_stats_c)
        ovl_c = compute_distribution_overlap(in_stats_c, out_stats_c)

        informia_results["canary"][a_val] = {
            "auc": auc_c,
            "ovl": ovl_c,
            "in_scores": in_stats_c,
            "out_scores": out_stats_c,
            "num": num_c.numpy(),
            "den": den_c.numpy(),
        }

    # Evaluate at optimal a values as well
    stats_nc_opt, _, _ = compute_informia_test_statistic(
        target_all_signals, t_pop_signals_all, target_all_mem, opt_a_non_canary
    )
    in_nc_opt = stats_nc_opt[:len(target_in)].numpy()
    out_nc_opt = stats_nc_opt[len(target_in):].numpy()
    auc_nc_opt, _, _ = compute_auc(in_nc_opt, out_nc_opt)
    ovl_nc_opt = compute_distribution_overlap(in_nc_opt, out_nc_opt)

    stats_c_opt, _, _ = compute_informia_test_statistic(
        target_c_all_signals, t_c_pop_signals_all, target_c_all_mem, opt_a_canary
    )
    in_c_opt = stats_c_opt[:len(target_c_in)].numpy()
    out_c_opt = stats_c_opt[len(target_c_in):].numpy()
    auc_c_opt, _, _ = compute_auc(in_c_opt, out_c_opt)
    ovl_c_opt = compute_distribution_overlap(in_c_opt, out_c_opt)

    print("\n" + "=" * 80)
    print(f"RESULTS SUMMARY AT STEP {args.step} (LABEL_NOISE_GROK_MNIST_CE_MLP)")
    print("=" * 80)
    print(f"Loss Attack CE (Non-Canary): AUC = {auc_loss_nc:.4f}, Overlap (OVL) = {ovl_score_nc:.4f}")
    print(f"Loss Attack CE (Canary):     AUC = {auc_loss_canary:.4f}, Overlap (OVL) = {ovl_score_canary:.4f}")
    print(f"InfoRMIA Non-Canary (opt a={opt_a_non_canary:.1f}): AUC = {auc_nc_opt:.4f}, Overlap (OVL) = {ovl_nc_opt:.4f}")
    print(f"InfoRMIA Canary     (opt a={opt_a_canary:.1f}): AUC = {auc_c_opt:.4f}, Overlap (OVL) = {ovl_c_opt:.4f}")
    print("-" * 80)
    print("InfoRMIA Non-Canary Sweep over 'a':")
    for a_val in a_sweep:
        res = informia_results["non_canary"][a_val]
        print(f"  a = {a_val:3.1f}: AUC = {res['auc']:.4f}, Overlap = {res['ovl']:.4f}")
    print("-" * 80)
    print("InfoRMIA Canary Sweep over 'a':")
    for a_val in a_sweep:
        res = informia_results["canary"][a_val]
        print(f"  a = {a_val:3.1f}: AUC = {res['auc']:.4f}, Overlap = {res['ovl']:.4f}")
    print("=" * 80 + "\n")

    # =========================================================================
    # FIGURE 1: LOSS ATTACK DISTRIBUTIONS (Step 20000)
    # =========================================================================
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
    nb = args.num_bins

    # Subplot (0, 0): Non-Canary CE Loss
    ax = axes[0, 0]
    in_l = t_in_losses.numpy()
    out_l = t_out_losses.numpy()
    plot_hist_and_kde(
        ax, in_l, out_l, num_bins=nb, in_color="#10b981", out_color="#ef4444",
        in_label=f"Train (IN) [N={len(in_l)}]", out_label=f"Test (OUT) [N={len(out_l)}]"
    )
    ax.set_title(f"(a) Non-Canary CE Loss (High-Res Bins={nb})\nAUC: {auc_loss_nc:.4f} | Overlap (OVL): {ovl_loss_nc:.4f}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Cross-Entropy Loss", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.legend(frameon=True, facecolor="white", fontsize=9)

    # Subplot (0, 1): Canary CE Loss
    ax = axes[0, 1]
    in_cl = t_c_in_losses.numpy()
    out_cl = t_c_out_losses.numpy()
    nb_c = max(30, nb // 2)
    plot_hist_and_kde(
        ax, in_cl, out_cl, num_bins=nb_c, in_color="#10b981", out_color="#ef4444",
        in_label=f"Canary Train (IN) [N={len(in_cl)}]", out_label=f"Canary Test (OUT) [N={len(out_cl)}]"
    )
    ax.set_title(f"(b) Canary CE Loss (Label Noise, Bins={nb_c})\nAUC: {auc_loss_canary:.4f} | Overlap (OVL): {ovl_loss_canary:.4f}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Cross-Entropy Loss", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.legend(frameon=True, facecolor="white", fontsize=9)

    # Subplot (1, 0): Non-Canary Confidence p(y|x)
    ax = axes[1, 0]
    in_p = t_in_probs.numpy()
    out_p = t_out_probs.numpy()
    plot_hist_and_kde(
        ax, in_p, out_p, num_bins=nb, in_color="#2563eb", out_color="#f97316",
        in_label="Train (IN)", out_label="Test (OUT)", fixed_range=(0.0, 1.0)
    )
    ax.set_title(f"(c) Non-Canary Confidence p(y|x) (Bins={nb})\nMean IN: {in_p.mean():.3f} | Mean OUT: {out_p.mean():.3f}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Softmax Probability p(y|x)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.legend(frameon=True, facecolor="white", fontsize=9)

    # Subplot (1, 1): Canary Confidence p(y|x)
    ax = axes[1, 1]
    in_cp = t_c_in_probs.numpy()
    out_cp = t_c_out_probs.numpy()
    plot_hist_and_kde(
        ax, in_cp, out_cp, num_bins=nb_c, in_color="#2563eb", out_color="#f97316",
        in_label="Canary Train (IN)", out_label="Canary Test (OUT)", fixed_range=(0.0, 1.0)
    )
    ax.set_title(f"(d) Canary Confidence p(y|x) (Label Noise, Bins={nb_c})\nMean IN: {in_cp.mean():.3f} | Mean OUT: {out_cp.mean():.3f}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Softmax Probability p(y|x)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.legend(frameon=True, facecolor="white", fontsize=9)

    fig.suptitle(f"High-Resolution Loss Attack Distributions & Separation for GROK_MNIST_CE_MLP (Step {args.step})", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    loss_plot_path = output_dir / f"loss_distributions_step_{args.step}.png"
    plt.savefig(loss_plot_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / f"loss_distributions_step_{args.step}.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved loss distribution plot: {loss_plot_path}")

    # =========================================================================
    # FIGURE 2: InfoRMIA DISTRIBUTIONS SWEEP OVER a (Step 20000)
    # =========================================================================
    fig, axes = plt.subplots(2, 6, figsize=(24, 8), sharey=False)

    for col_idx, a_val in enumerate(a_sweep):
        # Row 0: Non-canary
        ax0 = axes[0, col_idx]
        res_nc = informia_results["non_canary"][a_val]
        in_sc = res_nc["in_scores"]
        out_sc = res_nc["out_scores"]
        plot_hist_and_kde(
            ax0, in_sc, out_sc, num_bins=nb, in_color="#2563eb", out_color="#ef4444",
            in_label="Train (IN)", out_label="Test (OUT)"
        )
        opt_flag = " [OPT]" if np.isclose(a_val, opt_a_non_canary) else ""
        ax0.set_title(f"a = {a_val:.1f}{opt_flag}\nAUC: {res_nc['auc']:.4f} | Overlap: {res_nc['ovl']:.3f}", fontsize=10, fontweight="bold")
        if col_idx == 0:
            ax0.set_ylabel("Non-Canary\nDensity", fontsize=10, fontweight="bold")
        ax0.tick_params(labelsize=8)

        # Row 1: Canary
        ax1 = axes[1, col_idx]
        res_c = informia_results["canary"][a_val]
        in_csc = res_c["in_scores"]
        out_csc = res_c["out_scores"]
        plot_hist_and_kde(
            ax1, in_csc, out_csc, num_bins=nb_c, in_color="#059669", out_color="#f59e0b",
            in_label="Canary IN", out_label="Canary OUT"
        )
        c_opt_flag = " [OPT]" if np.isclose(a_val, opt_a_canary) else ""
        ax1.set_title(f"a = {a_val:.1f}{c_opt_flag}\nAUC: {res_c['auc']:.4f} | Overlap: {res_c['ovl']:.3f}", fontsize=10, fontweight="bold")
        if col_idx == 0:
            ax1.set_ylabel("Canary (Label Noise)\nDensity", fontsize=10, fontweight="bold")
        ax1.set_xlabel("InfoRMIA Score", fontsize=9)
        ax1.tick_params(labelsize=8)

    axes[0, 0].legend(loc="upper right", fontsize=8)
    axes[1, 0].legend(loc="upper right", fontsize=8)

    fig.suptitle(f"High-Resolution InfoRMIA Score Distributions across offline_a values (GROK_MNIST_CE_MLP Step {args.step})", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()
    informia_plot_path = output_dir / f"informia_distributions_sweep_a_step_{args.step}.png"
    plt.savefig(informia_plot_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / f"informia_distributions_sweep_a_step_{args.step}.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved InfoRMIA sweep distribution plot: {informia_plot_path}")

    # =========================================================================
    # FIGURE 3: DIAGNOSTICS & COMPARISON (Step 20000)
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Subplot 0: AUC vs 'a' curve
    ax0 = axes[0]
    aucs_nc = [informia_results["non_canary"][a]["auc"] for a in a_sweep]
    aucs_c = [informia_results["canary"][a]["auc"] for a in a_sweep]

    ax0.plot(a_sweep, aucs_nc, marker="s", color="#2563eb", linewidth=2, label="InfoRMIA (Non-Canary)")
    ax0.plot(a_sweep, aucs_c, marker="o", color="#059669", linewidth=2, linestyle="--", label="InfoRMIA (Canary)")
    ax0.axhline(auc_loss_nc, color="#dc2626", linestyle="-.", linewidth=2, label=f"Loss Attack CE Non-Canary ({auc_loss_nc:.3f})")
    ax0.axhline(auc_loss_canary, color="#d97706", linestyle=":", linewidth=2, label=f"Loss Attack CE Canary ({auc_loss_canary:.3f})")
    ax0.axhline(0.5, color="#9ca3af", linestyle=":", label="Random (0.5)")

    # Mark optimal a points
    ax0.scatter([opt_a_non_canary], [auc_nc_opt], color="#1d4ed8", s=120, zorder=5, marker="*", label=f"Tuned Non-Canary a={opt_a_non_canary:.1f}")
    ax0.scatter([opt_a_canary], [auc_c_opt], color="#047857", s=120, zorder=5, marker="*", label=f"Tuned Canary a={opt_a_canary:.1f}")

    ax0.set_xlabel("offline_a Parameter", fontsize=11)
    ax0.set_ylabel("Attack ROC AUC", fontsize=11)
    ax0.set_title("(a) Attack AUC vs offline_a", fontsize=12, fontweight="bold")
    ax0.set_ylim(0.4, 1.0)
    ax0.legend(frameon=True, fontsize=9, loc="lower right")

    # Subplot 1: Scatter of Numerator vs Denominator (Non-Canary at optimal a)
    ax1 = axes[1]
    res_opt_nc = informia_results["non_canary"][opt_a_non_canary]
    t_in_p_arr = t_in_probs.numpy()
    t_out_p_arr = t_out_probs.numpy()
    den_in_arr = res_opt_nc["den"][:len(target_in)]
    den_out_arr = res_opt_nc["den"][len(target_in):]

    ax1.scatter(den_in_arr, t_in_p_arr, alpha=0.5, color="#2563eb", s=18, label="Train (IN)")
    ax1.scatter(den_out_arr, t_out_p_arr, alpha=0.5, color="#ef4444", s=18, label="Test (OUT)")
    ax1.plot([0, 1], [0, 1], color="#6b7280", linestyle="--", linewidth=1.2, label="p(target) = p_hat(offline)")
    ax1.set_xlabel(r"Denominator $\hat{p}(x)$ [Reference Model Prior]", fontsize=11)
    ax1.set_ylabel(r"Numerator $p(y|x)$ [Target Model]", fontsize=11)
    ax1.set_title(f"(b) Target Signal vs. Reference Prior\n(Non-Canary at a={opt_a_non_canary:.1f})", fontsize=12, fontweight="bold")
    ax1.legend(frameon=True, fontsize=9)

    # Subplot 2: Reference Model P_out(x) Distribution
    ax2 = axes[2]
    # Mean P_out for train vs test
    p_out_x_train = get_rmia_mean_out_signals(target_all_signals[:, :len(target_in)], target_all_mem[:, :len(target_in)]).numpy()
    p_out_x_test = get_rmia_mean_out_signals(target_all_signals[:, len(target_in):], target_all_mem[:, len(target_in):]).numpy()
    plot_hist_and_kde(
        ax2, p_out_x_train, p_out_x_test, num_bins=nb, in_color="#2563eb", out_color="#ef4444",
        in_label="Train Samples (IN)", out_label="Test Samples (OUT)", fixed_range=(0.0, 1.0)
    )
    ax2.set_xlabel(r"Reference Model Mean Out-Probability $P_{out}(x)$", fontsize=11)
    ax2.set_ylabel("Density", fontsize=11)
    ax2.set_title(r"(c) Distribution of Reference Signal $P_{out}(x)$" + f"\nOverlap: {compute_distribution_overlap(p_out_x_train, p_out_x_test):.3f}", fontsize=12, fontweight="bold")
    ax2.legend(frameon=True, fontsize=9)

    plt.tight_layout()
    diag_plot_path = output_dir / f"informia_vs_loss_diagnostics_step_{args.step}.png"
    plt.savefig(diag_plot_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / f"informia_vs_loss_diagnostics_step_{args.step}.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved diagnostics plot: {diag_plot_path}")

    # 9. Save JSON Summary Metrics
    json_summary = {
        "step": args.step,
        "run_ids": {
            "target": run_ids[0],
            "validation": run_ids[1],
            "references": run_ids[2:],
        },
        "loss_attack_ce": {
            "non_canary": {"auc": auc_loss_nc, "overlap_loss": ovl_loss_nc, "overlap_score": ovl_score_nc},
            "canary": {"auc": auc_loss_canary, "overlap_loss": ovl_loss_canary, "overlap_score": ovl_score_canary},
        },
        "informia_tuned": {
            "non_canary": {"optimal_a": opt_a_non_canary, "auc": auc_nc_opt, "overlap": ovl_nc_opt},
            "canary": {"optimal_a": opt_a_canary, "auc": auc_c_opt, "overlap": ovl_c_opt},
        },
        "informia_sweep": {
            "non_canary": {str(a): {"auc": informia_results["non_canary"][a]["auc"], "ovl": informia_results["non_canary"][a]["ovl"]} for a in a_sweep},
            "canary": {str(a): {"auc": informia_results["canary"][a]["auc"], "ovl": informia_results["canary"][a]["ovl"]} for a in a_sweep},
        },
    }
    json_path = output_dir / f"summary_metrics_step_{args.step}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=2)
    print(f"Saved JSON metrics summary: {json_path}")


if __name__ == "__main__":
    main()
