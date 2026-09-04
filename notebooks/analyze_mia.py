import argparse
import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from privacy_and_grokking.config import TrainConfig
from privacy_and_grokking.metrics.evaluate import _process_loader
from privacy_and_grokking.metrics.mia import margin_distance_lf
from privacy_and_grokking.metrics.roc import compute_roc_metrics_single_step
from privacy_and_grokking.utils import Logger, get_device, set_all_seeds


def load_config(run_id):
    client = mlflow.tracking.MlflowClient()
    local_path = client.download_artifacts(run_id, "training_config.json")
    with open(local_path) as f:
        return TrainConfig.model_validate_json(f.read())


def get_dataloaders(config):
    data_container = config.data()
    train_subset = data_container.train
    test = data_container.test

    keep_on_gpu = torch.cuda.is_available()
    device = torch.device(get_device())

    def maybe_gpu_dataset(dataset):
        if keep_on_gpu and len(dataset) < 5000:
            from privacy_and_grokking.datasets import GpuDataset

            return GpuDataset(dataset, device)
        return dataset

    def get_loader_kwargs(dataset):
        if "GpuDataset" in str(type(dataset)):
            return {"num_workers": 0, "pin_memory": False}
        return {"num_workers": 4, "pin_memory": True}

    train_ds = maybe_gpu_dataset(train_subset)
    test_ds = maybe_gpu_dataset(test)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=False, **get_loader_kwargs(train_ds)
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=config.batch_size, shuffle=False, **get_loader_kwargs(test_ds)
    )
    return train_loader, test_loader, data_container


def _list_checkpoint_steps(run_id: str) -> list[int]:
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(run_id, path="checkpoints")
    steps = []
    for artifact in artifacts:
        parts = artifact.path.split("/")
        if len(parts) >= 2 and parts[0] == "checkpoints":
            candidate = parts[1]
            if candidate.isdigit():
                steps.append(int(candidate))
    return sorted(set(steps))


def extract_features_and_weights(model, loader, data_container):
    results, activations, labels, _ = _process_loader(
        model,
        loader,
        compute_mm=False,
        last_step=True,
        collect_features=True,
        normalization=data_container.normalization,
    )

    last_linear_name = None
    last_linear_weight = None
    last_linear_bias = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_linear_name = name
            last_linear_weight = module.weight.detach().cpu()
            last_linear_bias = module.bias.detach().cpu() if module.bias is not None else None

    feats = None
    if last_linear_name and f"{last_linear_name}.input" in activations:
        feats = activations[f"{last_linear_name}.input"]
        if feats.ndim > 2:
            feats = feats.reshape(feats.size(0), -1)

    return feats, labels, last_linear_weight, last_linear_bias, results


def margin_distance_true_mean(features, labels, train_feats, train_labels, w, b):
    dists = {}
    num_classes = w.shape[0]
    device = features.device

    if b is None:
        b = torch.zeros(num_classes, device=w.device)

    w = w.to(device)
    b = b.to(device)

    for c in range(num_classes):
        mask_c = labels == c
        train_mask_c = train_labels == c
        if mask_c.sum() == 0 or train_mask_c.sum() == 0:
            continue

        f_sub = features[mask_c].float()
        f_train_c = train_feats[train_mask_c].float()

        def get_all_margins(f_in):
            margins = []
            for k in range(num_classes):
                if k == c:
                    continue
                w_diff = w[c] - w[k]
                b_diff = b[c] - b[k]
                norm_w = torch.norm(w_diff, p=2)
                if norm_w == 0:
                    m = torch.zeros(f_in.shape[0], device=device)
                else:
                    m = (torch.matmul(f_in, w_diff) + b_diff) / norm_w
                margins.append(m.unsqueeze(1))
            if len(margins) == 0:
                return torch.zeros((f_in.shape[0], 0), device=device)
            return torch.cat(margins, dim=1)

        # Calculate true mean from training data
        true_mean = f_train_c.mean(dim=0, keepdim=True)

        true_margin_mean = get_all_margins(true_mean)[0]
        sample_margins = get_all_margins(f_sub)

        dists[c] = torch.norm(sample_margins - true_margin_mean, dim=1).cpu()

    return dists


def calculate_margin_profile(f_in, w, b, c):
    num_classes = w.shape[0]
    margins = []
    device = w.device

    # Check if b is None (can happen if model has no bias)
    has_b = b is not None

    for k in range(num_classes):
        if k == c:
            continue
        w_diff = w[c] - w[k]
        b_diff = b[c] - b[k] if has_b else 0.0
        norm_w = torch.norm(w_diff, p=2)
        if norm_w == 0:
            m = torch.zeros(f_in.shape[0], device=device)
        else:
            m = (torch.matmul(f_in.float(), w_diff.float()) + b_diff) / norm_w
        margins.append(m.unsqueeze(1))

    if len(margins) == 0:
        return torch.zeros((f_in.shape[0], 0), device=device)
    return torch.cat(margins, dim=1)


def margin_distance_lf(features, labels, w, b, pool_mean_norm):
    """
    Calculate the L2 distance of each sample's margin profile to the Proxy Mean's margin profile.
    """
    num_classes = w.shape[0]
    dists = {}

    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            continue

        sample_features = features[mask]
        sample_margins = calculate_margin_profile(sample_features, w, b, c)

        # Calculate the Proxy Mean's margin profile
        w_proxy_c = w[c] / torch.norm(w[c]) * pool_mean_norm
        proxy_margins = calculate_margin_profile(w_proxy_c.unsqueeze(0), w, b, c)

        dists[c] = torch.norm(sample_margins - proxy_margins, dim=1).cpu()

    return dists


def margin_distance_lf_class_specific(features, labels, w, b, class_norms):
    """
    Calculate the L2 distance of each sample's margin profile to a Class-Specific Proxy Mean's margin profile.
    """
    num_classes = w.shape[0]
    dists = {}

    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            continue

        sample_features = features[mask]
        sample_margins = calculate_margin_profile(sample_features, w, b, c)

        # Calculate the Proxy Mean's margin profile using the specific class norm
        w_proxy_c = w[c] / torch.norm(w[c]) * class_norms[c]
        proxy_margins = calculate_margin_profile(w_proxy_c.unsqueeze(0), w, b, c)

        dists[c] = torch.norm(sample_margins - proxy_margins, dim=1).cpu()

    return dists


def plot_trajectory_fig(fig, axes, eval_steps, metrics, title):
    # Plot AUC
    axes[0].plot(eval_steps, metrics["mse"]["auc"], marker="s", color="green", label="MSE Loss")
    axes[0].plot(
        eval_steps,
        metrics["proxy"]["auc"],
        marker="o",
        color="purple",
        label="Margin Distance (Global Proxy Mean)",
    )
    axes[0].plot(
        eval_steps,
        metrics["proxy_class"]["auc"],
        marker="x",
        color="blue",
        label="Margin Distance (Class Proxy Mean)",
    )
    axes[0].plot(
        eval_steps,
        metrics["true"]["auc"],
        marker="^",
        color="red",
        label="Margin Distance (True Train Mean)",
    )
    axes[0].set_title(title)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("AUC")
    axes[0].legend()
    axes[0].grid(True)

    # Plot TPR@1%
    axes[1].plot(eval_steps, metrics["mse"]["tpr_1"], marker="s", color="green")
    axes[1].plot(eval_steps, metrics["proxy"]["tpr_1"], marker="o", color="purple")
    axes[1].plot(eval_steps, metrics["proxy_class"]["tpr_1"], marker="x", color="blue")
    axes[1].plot(eval_steps, metrics["true"]["tpr_1"], marker="^", color="red")
    axes[1].set_title("TPR @ 1% FPR")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].grid(True)

    # Plot TPR@5%
    axes[2].plot(eval_steps, metrics["mse"]["tpr_5"], marker="s", color="green")
    axes[2].plot(eval_steps, metrics["proxy"]["tpr_5"], marker="o", color="purple")
    axes[2].plot(eval_steps, metrics["proxy_class"]["tpr_5"], marker="x", color="blue")
    axes[2].plot(eval_steps, metrics["true"]["tpr_5"], marker="^", color="red")
    axes[2].set_title("TPR @ 5% FPR")
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("True Positive Rate")
    axes[2].grid(True)

    fig.tight_layout()


def get_class0_margins(features, labels, w, b, c):
    """Calculate the decision margin for samples in class C and class 0 to their decision boundary."""
    mask_c = labels == c
    mask_0 = labels == 0

    w_diff = w[c] - w[0]
    b_diff = (b[c] - b[0]) if b is not None else 0.0
    norm_w = torch.norm(w_diff, p=2)

    def calc_m(f_sub):
        if norm_w == 0:
            return torch.zeros(f_sub.shape[0])
        return (torch.matmul(f_sub, w_diff) + b_diff) / norm_w

    m_c = calc_m(features[mask_c].float())
    m_0 = -calc_m(
        features[mask_0].float()
    )  # invert so positive means deeper into correct territory

    if len(m_c) == 0 and len(m_0) == 0:
        return torch.tensor([])

    return torch.cat([m_c, m_0])


def main(run_id, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    device = get_device()

    print(f"Loading config for run {run_id}...")
    cfg = load_config(run_id)
    set_all_seeds(cfg.seed)

    train_loader, test_loader, data_container = get_dataloaders(cfg)

    model = cfg.model(
        input_dim=data_container.input_shape,
        num_classes=data_container.num_classes,
    )
    model.to(device)

    steps = _list_checkpoint_steps(run_id)
    if not steps:
        print("No checkpoints found.")
        return

    # Analyze 150k step for margins
    target_step = 150000
    if target_step in steps:
        print(f"\n--- Analyzing Step {target_step} ---")
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_uri = f"runs:/{run_id}/checkpoints/{target_step}/model.pth"
            mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri, dst_path=tmpdir)
            model.load_state_dict(
                torch.load(Path(tmpdir) / "model.pth", map_location=device, weights_only=True)
            )

        train_feats, train_labels, w, b, train_results = extract_features_and_weights(
            model, train_loader, data_container
        )
        test_feats, test_labels, _, _, test_results = extract_features_and_weights(
            model, test_loader, data_container
        )

        # Define plotting pairs: Class C vs Class 0
        pairs_to_plot = [(c, 0) for c in range(1, data_container.num_classes)]

        pool_mean_norm = (
            torch.cat([train_feats.float(), test_feats.float()]).norm(dim=1).mean().item()
        )

        fig, axes = plt.subplots(5, 2, figsize=(16, 20), sharex=False)
        axes = axes.flatten()

        for i, (c, c0) in enumerate(pairs_to_plot):
            ax = axes[i]

            # Use same margin calculation as before
            w_diff = w[c] - w[c0]
            b_diff = (b[c] - b[c0]) if b is not None else 0.0
            norm_w = torch.norm(w_diff, p=2)

            def get_margins(f_subset):
                if norm_w == 0:
                    return np.zeros(f_subset.shape[0])
                return ((torch.matmul(f_subset, w_diff) + b_diff) / norm_w).numpy()

            tr_pos = get_margins(train_feats[train_labels == c].float())
            tr_neg = get_margins(train_feats[train_labels == c0].float())
            te_pos = get_margins(test_feats[test_labels == c].float())
            te_neg = get_margins(test_feats[test_labels == c0].float())

            x_min = (
                min([m.min() for m in [tr_pos, tr_neg, te_pos, te_neg] if len(m) > 0] + [0]) - 0.5
            )
            x_max = (
                max([m.max() for m in [tr_pos, tr_neg, te_pos, te_neg] if len(m) > 0] + [0]) + 0.5
            )
            bins = np.linspace(x_min, x_max, 1500)

            # Plot positive class (Class C) pointing UP
            if len(tr_pos) > 0:
                counts_pos, edges_pos = np.histogram(tr_pos, bins=bins, density=False)
                ax.bar(
                    edges_pos[:-1],
                    counts_pos,
                    width=np.diff(edges_pos),
                    align="edge",
                    color="steelblue",
                    alpha=1.0,
                    edgecolor="none",
                    label=f"Train class {c}",
                )
            if len(te_pos) > 0:
                counts_pos_te, edges_pos_te = np.histogram(te_pos, bins=bins, density=False)
                ax.bar(
                    edges_pos_te[:-1],
                    counts_pos_te,
                    width=np.diff(edges_pos_te),
                    align="edge",
                    color="dodgerblue",
                    alpha=0.6,
                    edgecolor="none",
                    label=f"Test class {c}",
                )

            # Plot negative class (Class 0) pointing DOWN
            if len(tr_neg) > 0:
                counts_neg, edges_neg = np.histogram(tr_neg, bins=bins, density=False)
                ax.bar(
                    edges_neg[:-1],
                    -counts_neg,
                    width=np.diff(edges_neg),
                    align="edge",
                    color="peru",
                    alpha=1.0,
                    edgecolor="none",
                    label=f"Train class {c0}",
                )
            if len(te_neg) > 0:
                counts_neg_te, edges_neg_te = np.histogram(te_neg, bins=bins, density=False)
                ax.bar(
                    edges_neg_te[:-1],
                    -counts_neg_te,
                    width=np.diff(edges_neg_te),
                    align="edge",
                    color="sandybrown",
                    alpha=0.6,
                    edgecolor="none",
                    label=f"Test class {c0}",
                )

            # Compute proxy mean margins
            w_proxy_c = w[c] / torch.norm(w[c]) * pool_mean_norm
            w_proxy_0 = w[c0] / torch.norm(w[c0]) * pool_mean_norm

            proxy_margin_c = ((torch.dot(w_proxy_c, w_diff) + b_diff) / norm_w).item()
            proxy_margin_0 = ((torch.dot(w_proxy_0, w_diff) + b_diff) / norm_w).item()

            ax.axvline(
                proxy_margin_c,
                color="steelblue",
                linestyle="--",
                linewidth=1.5,
                label=f"Est. Mean class {c}",
            )
            ax.axvline(
                proxy_margin_0,
                color="peru",
                linestyle="--",
                linewidth=1.5,
                label=f"Est. Mean class {c0}",
            )

            ax.axhline(0, color="black", linewidth=0.8)
            ax.axvline(0, color="black", linewidth=1.5)

            ax.set_yscale("symlog", linthresh=1.0)
            ax.set_yticks([-1000, -100, -10, 0, 10, 100, 1000])
            ax.set_yticklabels(["1000", "100", "10", "0", "10", "100", "1000"])

            ax.set_title(f"Class {c} vs Class {c0}", fontsize=12)
            ax.set_xlabel("Signed distance to decision boundary")
            ax.set_ylabel("Count (Log Scale)")
            ax.legend(fontsize=8, frameon=True)
            ax.grid(True, axis="both", linestyle=":", alpha=0.5)

        # Hide the last empty subplot (if 9 pairs on a 5x2 grid)
        if len(pairs_to_plot) < 10:
            axes[-1].axis("off")

        fig.suptitle(
            f"Margins of individual examples (Histograms)\nModel {run_id[:8]}... | Step {target_step}",
            fontsize=14,
        )
        fig.tight_layout()

        pdf_path = os.path.join(output_dir, f"margin_plots_{target_step}.pdf")
        fig.savefig(pdf_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved margin plots to {pdf_path}")

        # Plot MSE histograms
        fig_mse, axes_mse = plt.subplots(1, 2, figsize=(18, 6))
        tr_mse = train_results["mse_loss"].detach().cpu().numpy()
        te_mse = test_results["mse_loss"].detach().cpu().numpy()

        # Clip to small epsilon to avoid log(0) issues
        eps = 1e-12
        tr_mse = np.clip(tr_mse, eps, None)
        te_mse = np.clip(te_mse, eps, None)

        min_val = min(tr_mse.min(), te_mse.min())
        max_val = max(tr_mse.max(), te_mse.max())
        bins_mse = np.logspace(np.log10(min_val), np.log10(max_val), 100)

        # Plot 1: Density
        axes_mse[0].hist(
            tr_mse,
            bins=bins_mse,
            alpha=0.5,
            label="Train MSE",
            color="blue",
            density=True,
            log=True,
        )
        axes_mse[0].hist(
            te_mse,
            bins=bins_mse,
            alpha=0.5,
            label="Test MSE",
            color="orange",
            density=True,
            log=True,
        )
        axes_mse[0].set_xscale("log")
        axes_mse[0].set_title(f"MSE Loss Distribution - Density (Step {target_step})")
        axes_mse[0].set_xlabel("MSE Loss (Log Scale)")
        axes_mse[0].set_ylabel("Density (Log Scale)")
        axes_mse[0].legend()
        axes_mse[0].grid(True, linestyle=":", alpha=0.5)

        # Plot 2: Raw Counts
        axes_mse[1].hist(
            tr_mse,
            bins=bins_mse,
            alpha=0.5,
            label="Train MSE",
            color="blue",
            density=False,
            log=True,
        )
        axes_mse[1].hist(
            te_mse,
            bins=bins_mse,
            alpha=0.5,
            label="Test MSE",
            color="orange",
            density=False,
            log=True,
        )
        axes_mse[1].set_xscale("log")
        axes_mse[1].set_title(f"MSE Loss Distribution - Raw Counts (Step {target_step})")
        axes_mse[1].set_xlabel("MSE Loss (Log Scale)")
        axes_mse[1].set_ylabel("Raw Count (Log Scale)")
        axes_mse[1].legend()
        axes_mse[1].grid(True, linestyle=":", alpha=0.5)

        fig_mse.tight_layout()

        mse_pdf_path = os.path.join(output_dir, f"mse_plots_{target_step}.pdf")
        fig_mse.savefig(mse_pdf_path, dpi=150, bbox_inches="tight")
        plt.close(fig_mse)
        print(f"Saved MSE loss plots to {mse_pdf_path}")

    # Analyze MIA AUC over time
    print("\n--- Analyzing MIA AUC Over Time ---")
    eval_steps = [s for s in steps if s % 10000 == 0]

    metrics_history = {
        "mse": {"auc": [], "tpr_1": [], "tpr_5": []},
        "proxy": {"auc": [], "tpr_1": [], "tpr_5": []},
        "proxy_class": {"auc": [], "tpr_1": [], "tpr_5": []},
        "true": {"auc": [], "tpr_1": [], "tpr_5": []},
    }

    metrics_history_balanced = {
        "mse": {"auc": [], "tpr_1": [], "tpr_5": []},
        "proxy": {"auc": [], "tpr_1": [], "tpr_5": []},
        "proxy_class": {"auc": [], "tpr_1": [], "tpr_5": []},
        "true": {"auc": [], "tpr_1": [], "tpr_5": []},
    }

    for step in tqdm(eval_steps, desc="Steps"):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_uri = f"runs:/{run_id}/checkpoints/{step}/model.pth"
            mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri, dst_path=tmpdir)
            model.load_state_dict(
                torch.load(Path(tmpdir) / "model.pth", map_location=device, weights_only=True)
            )

        train_feats, train_labels, w, b, train_results = extract_features_and_weights(
            model, train_loader, data_container
        )
        test_feats, test_labels, _, _, test_results = extract_features_and_weights(
            model, test_loader, data_container
        )

        # Determine balanced test indices
        n_train = len(train_labels)
        n_test = len(test_labels)
        if n_test > n_train:
            bal_idx = torch.randperm(n_test)[:n_train]
        else:
            bal_idx = torch.arange(n_test)

        # 1. MSE Loss MIA
        train_mse = -train_results[
            "mse_loss"
        ].detach()  # Negate because compute_roc_metrics_single_step expects higher score for train
        test_mse = -test_results["mse_loss"].detach()
        m_mse = compute_roc_metrics_single_step(train_mse, test_mse)
        metrics_history["mse"]["auc"].append(m_mse["auc"])
        metrics_history["mse"]["tpr_1"].append(m_mse.get("tpr-at-fpr/1", 0.0))
        metrics_history["mse"]["tpr_5"].append(m_mse.get("tpr-at-fpr/5", 0.0))

        m_mse_bal = compute_roc_metrics_single_step(train_mse, test_mse[bal_idx])
        metrics_history_balanced["mse"]["auc"].append(m_mse_bal["auc"])
        metrics_history_balanced["mse"]["tpr_1"].append(m_mse_bal.get("tpr-at-fpr/1", 0.0))
        metrics_history_balanced["mse"]["tpr_5"].append(m_mse_bal.get("tpr-at-fpr/5", 0.0))

        # 2. Single-threshold Margin MIA (proxy weight circle)
        pool_mean_norm = (
            torch.cat([train_feats.float(), test_feats.float()]).norm(dim=1).mean().item()
        )
        train_dists_lf = margin_distance_lf(train_feats, train_labels, w, b, pool_mean_norm)
        test_dists_lf = margin_distance_lf(test_feats, test_labels, w, b, pool_mean_norm)

        all_train_flat = (
            torch.cat(list(train_dists_lf.values())) if train_dists_lf else torch.tensor([])
        )
        all_test_flat = (
            torch.cat(list(test_dists_lf.values())) if test_dists_lf else torch.tensor([])
        )

        if len(all_train_flat) > 0 and len(all_test_flat) > 0:
            m_single = compute_roc_metrics_single_step(
                -all_train_flat.detach(), -all_test_flat.detach()
            )
            metrics_history["proxy"]["auc"].append(m_single["auc"])
            metrics_history["proxy"]["tpr_1"].append(m_single.get("tpr-at-fpr/1", 0.0))
            metrics_history["proxy"]["tpr_5"].append(m_single.get("tpr-at-fpr/5", 0.0))

            m_single_bal = compute_roc_metrics_single_step(
                -all_train_flat.detach(), -all_test_flat.detach()[bal_idx]
            )
            metrics_history_balanced["proxy"]["auc"].append(m_single_bal["auc"])
            metrics_history_balanced["proxy"]["tpr_1"].append(m_single_bal.get("tpr-at-fpr/1", 0.0))
            metrics_history_balanced["proxy"]["tpr_5"].append(m_single_bal.get("tpr-at-fpr/5", 0.0))
        else:
            for d in [metrics_history, metrics_history_balanced]:
                d["proxy"]["auc"].append(0.5)
                d["proxy"]["tpr_1"].append(0.0)
                d["proxy"]["tpr_5"].append(0.0)

        # 3. Class-Specific Proxy Mean MIA
        all_feats = torch.cat([train_feats.float(), test_feats.float()])
        all_labels_combined = torch.cat([train_labels, test_labels])
        class_norms = {}
        for c in range(data_container.num_classes):
            mask = all_labels_combined == c
            if mask.sum() > 0:
                class_norms[c] = all_feats[mask].norm(dim=1).mean().item()
            else:
                class_norms[c] = 0.0

        train_dists_class = margin_distance_lf_class_specific(
            train_feats, train_labels, w, b, class_norms
        )
        test_dists_class = margin_distance_lf_class_specific(
            test_feats, test_labels, w, b, class_norms
        )

        all_train_class_flat = (
            torch.cat(list(train_dists_class.values())) if train_dists_class else torch.tensor([])
        )
        all_test_class_flat = (
            torch.cat(list(test_dists_class.values())) if test_dists_class else torch.tensor([])
        )

        if len(all_train_class_flat) > 0 and len(all_test_class_flat) > 0:
            m_class = compute_roc_metrics_single_step(
                -all_train_class_flat.detach(), -all_test_class_flat.detach()
            )
            metrics_history["proxy_class"]["auc"].append(m_class["auc"])
            metrics_history["proxy_class"]["tpr_1"].append(m_class.get("tpr-at-fpr/1", 0.0))
            metrics_history["proxy_class"]["tpr_5"].append(m_class.get("tpr-at-fpr/5", 0.0))

            m_class_bal = compute_roc_metrics_single_step(
                -all_train_class_flat.detach(), -all_test_class_flat.detach()[bal_idx]
            )
            metrics_history_balanced["proxy_class"]["auc"].append(m_class_bal["auc"])
            metrics_history_balanced["proxy_class"]["tpr_1"].append(
                m_class_bal.get("tpr-at-fpr/1", 0.0)
            )
            metrics_history_balanced["proxy_class"]["tpr_5"].append(
                m_class_bal.get("tpr-at-fpr/5", 0.0)
            )
        else:
            for d in [metrics_history, metrics_history_balanced]:
                d["proxy_class"]["auc"].append(0.5)
                d["proxy_class"]["tpr_1"].append(0.0)
                d["proxy_class"]["tpr_5"].append(0.0)

        # 4. Margin Distance True Mean MIA
        train_dists_true = margin_distance_true_mean(
            train_feats, train_labels, train_feats, train_labels, w, b
        )
        test_dists_true = margin_distance_true_mean(
            test_feats, test_labels, train_feats, train_labels, w, b
        )

        all_train_true_flat = (
            torch.cat(list(train_dists_true.values())) if train_dists_true else torch.tensor([])
        )
        all_test_true_flat = (
            torch.cat(list(test_dists_true.values())) if test_dists_true else torch.tensor([])
        )

        if len(all_train_true_flat) > 0 and len(all_test_true_flat) > 0:
            m_true = compute_roc_metrics_single_step(
                -all_train_true_flat.detach(), -all_test_true_flat.detach()
            )
            metrics_history["true"]["auc"].append(m_true["auc"])
            metrics_history["true"]["tpr_1"].append(m_true.get("tpr-at-fpr/1", 0.0))
            metrics_history["true"]["tpr_5"].append(m_true.get("tpr-at-fpr/5", 0.0))

            m_true_bal = compute_roc_metrics_single_step(
                -all_train_true_flat.detach(), -all_test_true_flat.detach()[bal_idx]
            )
            metrics_history_balanced["true"]["auc"].append(m_true_bal["auc"])
            metrics_history_balanced["true"]["tpr_1"].append(m_true_bal.get("tpr-at-fpr/1", 0.0))
            metrics_history_balanced["true"]["tpr_5"].append(m_true_bal.get("tpr-at-fpr/5", 0.0))
        else:
            for d in [metrics_history, metrics_history_balanced]:
                d["true"]["auc"].append(0.5)
                d["true"]["tpr_1"].append(0.0)
                d["true"]["tpr_5"].append(0.0)

    if eval_steps:
        # Save imbalanced plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        plot_trajectory_fig(
            fig, axes, eval_steps, metrics_history, "MIA Trajectory (Imbalanced: All Test Samples)"
        )
        pdf_path = os.path.join(output_dir, "mia_auc_trajectory.pdf")
        fig.savefig(pdf_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved MIA AUC trajectory to {pdf_path}")

        # Save balanced plot
        fig_bal, axes_bal = plt.subplots(1, 3, figsize=(18, 5))
        plot_trajectory_fig(
            fig_bal,
            axes_bal,
            eval_steps,
            metrics_history_balanced,
            "MIA Trajectory (Balanced: 1:1 Train/Test Ratio)",
        )
        pdf_bal_path = os.path.join(output_dir, "mia_auc_trajectory_balanced.pdf")
        fig_bal.savefig(pdf_bal_path, dpi=150, bbox_inches="tight")
        plt.close(fig_bal)
        print(f"Saved balanced MIA AUC trajectory to {pdf_bal_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze MIA margins and AUC over time.")
    parser.add_argument("run_id", type=str, help="MLflow Run ID")
    parser.add_argument("--out", type=str, default="mia_analysis_output", help="Output directory")
    args = parser.parse_args()

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5050")
    mlflow.set_tracking_uri(tracking_uri)

    with Logger():
        main(args.run_id, args.out)
