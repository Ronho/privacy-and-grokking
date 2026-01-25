import matplotlib.pyplot as plt
import torch
import json
import polars as pl

from dataclasses import dataclass
from sklearn.metrics import auc, roc_curve
from ..config.model import TrainConfig
from ..logger.registry import get_logger
from ..path_keeper import get_path_keeper


def _flatten_dict(d: dict, parent_key: str = "", sep: str = "_") -> dict:
    items: list[tuple[str, any]] = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def plot_distributions(train_data_last, test_data_last, step, cfg, pk, metric_name, filename_prefix):
    """Plot histogram distributions with mean and median lines."""
    train_mean = train_data_last.mean()
    train_median = train_data_last.median().item()
    test_mean = test_data_last.mean()
    test_median = test_data_last.median().item()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(train_data_last, bins=50, alpha=0.6, label="Members", density=True)
    ax.hist(test_data_last, bins=50, alpha=0.6, label="Non-members", density=True)
    ax.axvline(train_mean, color="blue", linestyle="--", linewidth=2, alpha=0.8, label=f"Members Mean: {train_mean:.3f}")
    ax.axvline(train_median, color="blue", linestyle=":", linewidth=2, alpha=0.8, label=f"Members Median: {train_median:.3f}")
    ax.axvline(test_mean, color="orange", linestyle="--", linewidth=2, alpha=0.8, label=f"Non-members Mean: {test_mean:.3f}")
    ax.axvline(test_median, color="orange", linestyle=":", linewidth=2, alpha=0.8, label=f"Non-members Median: {test_median:.3f}")
    ax.set_xlabel(metric_name)
    ax.set_ylabel("Density")
    ax.set_title(f"Distribution at Step {step}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.text(0.02, 0.01, f"Model: {cfg.name}", fontsize=9, alpha=0.6)
    fig.tight_layout()
    fig.savefig(pk.IMAGE_FOLDER / f"{filename_prefix}_distributions_step_{step}.png")
    plt.close(fig)

def plot_evolution(train_data_over_time, test_data_over_time, steps, cfg, pk, metric_name, filename_prefix):
    """Plot mean +/- std evolution over training steps."""
    train_mean = train_data_over_time.mean(dim=1)
    train_std = train_data_over_time.std(dim=1)
    test_mean = test_data_over_time.mean(dim=1)
    test_std = test_data_over_time.std(dim=1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, train_mean, "b-", label="Train (Members)", linewidth=2)
    ax.fill_between(steps, train_mean - train_std, train_mean + train_std, alpha=0.3, color="blue")
    ax.plot(steps, test_mean, "orange", label="Test (Non-members)", linewidth=2)
    ax.fill_between(steps, test_mean - test_std, test_mean + test_std, alpha=0.3, color="orange")
    ax.set_xlabel("Training Step")
    ax.set_ylabel(metric_name)
    ax.set_title(f"Evolution of {metric_name} During Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.text(0.02, 0.01, f"Model: {cfg.name}", fontsize=9, alpha=0.6)
    fig.tight_layout()
    fig.savefig(pk.IMAGE_FOLDER / f"{filename_prefix}_evolution.png")
    plt.close(fig)

def compute_roc_metrics(train_data_over_time, test_data_over_time, steps, fpr_rates=[0.01, 0.05, 0.10]):
    """Compute ROC metrics (TPR at FPR and AUC) over training steps."""
    tpr_at_fprs = {fpr_rate: [] for fpr_rate in fpr_rates}
    auc_scores = []
    
    for step_idx in range(len(steps)):
        train_data_step = train_data_over_time[step_idx]
        test_data_step = test_data_over_time[step_idx]
        
        y_true = torch.concatenate([torch.ones(len(train_data_step)), torch.zeros(len(test_data_step))])
        y_scores = torch.concatenate([train_data_step, test_data_step])
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        auc_scores.append(roc_auc)

        for fpr_rate in fpr_rates:
            idx = torch.argmin(torch.abs(torch.tensor(fpr) - fpr_rate)).item()
            tpr_at_fprs[fpr_rate].append(tpr[idx])
    
    return tpr_at_fprs, auc_scores

def plot_tpr_at_fpr(tpr_at_fprs, steps, cfg, pk, filename_prefix):
    """Plot TPR at fixed FPR values over training steps."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for fpr_rate in tpr_at_fprs.keys():
        ax.plot(steps, tpr_at_fprs[fpr_rate], label=f"TPR @ FPR={fpr_rate*100:.0f}%")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("True Positive Rate at Fixed False Positive Rates")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    fig.text(0.02, 0.01, f"Model: {cfg.name}", fontsize=9, alpha=0.6)
    fig.tight_layout()
    fig.savefig(pk.IMAGE_FOLDER / f"{filename_prefix}_tpr_at_fpr_evolution.png")
    plt.close(fig)

def plot_auc_evolution(auc_scores, steps, cfg, pk, filename_prefix):
    """Plot AUC score evolution over training steps."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, auc_scores)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("AUC")
    ax.set_title("ROC AUC Score Evolution")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random Classifier")
    ax.legend()
    fig.text(0.02, 0.01, f"Model: {cfg.name}", fontsize=9, alpha=0.6)
    fig.tight_layout()
    fig.savefig(pk.IMAGE_FOLDER / f"{filename_prefix}_auc_evolution.png")
    plt.close(fig)

def plot_roc_curve(train_data_last, test_data_last, step, cfg, pk, filename_prefix):
    """Plot ROC curve for the last training step."""
    y_true_last = torch.concatenate([torch.ones(len(train_data_last)), torch.zeros(len(test_data_last))]).numpy()
    y_scores_last = torch.concatenate([train_data_last, test_data_last]).numpy()
    fpr_last, tpr_last, _ = roc_curve(y_true_last, y_scores_last)
    roc_auc_last = auc(fpr_last, tpr_last)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(fpr_last, tpr_last, label=f"ROC curve (AUC = {roc_auc_last:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Classifier")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title(f"ROC Curve at Step {step}")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.text(0.02, 0.01, f"Model: {cfg.name}", fontsize=9, alpha=0.6)
    fig.tight_layout()
    fig.savefig(pk.IMAGE_FOLDER / f"{filename_prefix}_roc_curve_step_{step}.png")
    plt.close(fig)

def plot_all_attacks_roc_curves(attack_containers, cfg, pk):
    """Plot ROC curves for all MIA attacks on the same plot."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for cont in attack_containers:
        train_data_last = cont.train_data[-1]
        test_data_last = cont.test_data[-1]
        
        y_true = torch.concatenate([torch.ones(len(train_data_last)), torch.zeros(len(test_data_last))]).numpy()
        y_scores = torch.concatenate([train_data_last, test_data_last]).numpy()
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, label=f"{cont.name} (AUC = {roc_auc:.4f})", linewidth=2)
    
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Classifier", linewidth=2)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title(f"ROC Curves for All MIA Attacks at Step {attack_containers[0].steps[-1]}")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.text(0.02, 0.01, f"Model: {cfg.name}", fontsize=9, alpha=0.6)
    fig.tight_layout()
    fig.savefig(pk.IMAGE_FOLDER / f"all_attacks_roc_curves_step_{attack_containers[0].steps[-1]}.png")
    plt.close(fig)

def plot_all_attacks_score_distributions(attack_containers, cfg, pk):
    """Plot score distributions (0-1 normalized) for all MIA attacks."""
    n_attacks = len(attack_containers)
    fig, axes = plt.subplots(n_attacks, 1, figsize=(12, 4 * n_attacks))
    
    # Handle single attack case
    if n_attacks == 1:
        axes = [axes]
    
    for idx, cont in enumerate(attack_containers):
        ax = axes[idx]
        
        # Get final step data
        train_scores = cont.train_data[-1].numpy()
        test_scores = cont.test_data[-1].numpy()
        
        # Normalize scores to [0, 1] range
        all_scores = torch.concatenate([torch.tensor(train_scores), torch.tensor(test_scores)]).numpy()
        min_score = all_scores.min()
        max_score = all_scores.max()
        
        if max_score > min_score:
            train_norm = (train_scores - min_score) / (max_score - min_score)
            test_norm = (test_scores - min_score) / (max_score - min_score)
        else:
            train_norm = train_scores
            test_norm = test_scores
        
        # Create histogram with 100 bins
        bins = torch.linspace(0, 1, 101).numpy()
        ax.hist(train_norm, bins=bins, alpha=0.6, label='Members', color='blue', edgecolor='black', linewidth=0.5)
        ax.hist(test_norm, bins=bins, alpha=0.6, label='Non-members', color='orange', edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Normalized Attack Score (0 = lowest, 1 = highest)', fontsize=11)
        ax.set_ylabel('Number of Samples', fontsize=11)
        ax.set_title(f'{cont.name} - Score Distribution', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xlim(0, 1)
    
    fig.suptitle(f'Attack Score Distributions at Step {attack_containers[0].steps[-1]}', 
                 fontsize=14, fontweight='bold', y=0.995)
    fig.text(0.02, 0.01, f"Model: {cfg.name}", fontsize=9, alpha=0.6)
    fig.tight_layout()
    fig.savefig(pk.IMAGE_FOLDER / f"all_attacks_score_distributions_step_{attack_containers[0].steps[-1]}.png", dpi=150)
    plt.close(fig)

def plot_training_and_attack_evolution(attack_containers, cfg, pk):
    """Plot combined training metrics and attack performance evolution."""
    # Load training metrics
    if not pk.TRAIN_METRICS.exists():
        return
    
    metrics = json.loads(pk.TRAIN_METRICS.read_text())
    flat_metrics = [_flatten_dict(m) for m in metrics]
    df = pl.DataFrame(flat_metrics)
    
    # Create figure with 4 subplots sharing x-axis
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    
    # 1. Accuracy plot
    ax1.plot(df["step"], df["train_accuracy"], label="Train Accuracy", linewidth=2, color='#2563eb')
    ax1.plot(df["step"], df["test_accuracy"], label="Test Accuracy", linewidth=2, color='#dc2626')
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.set_title("Model Performance Metrics", fontsize=12, fontweight='bold')
    ax1.legend(loc="best", fontsize=10)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.set_ylim(0, 1.05)
    ax1.set_xscale('log')
    
    # 2. Weight Norm plot
    ax2.plot(df["step"], df["norm"], label="Total Weight Norm", linewidth=2, color='#7c3aed')
    ax2.plot(df["step"], df["last_layer_norm"], label="Last Layer Norm", linewidth=2, color='#ec4899')
    ax2.set_ylabel("Weight Norm (log scale)", fontsize=11)
    ax2.set_yscale('log')
    ax2.legend(loc="best", fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')
    
    # 3. Loss plot
    ax3.plot(df["step"], df["train_loss"], label="Train Loss", linewidth=2, color='#2563eb')
    ax3.plot(df["step"], df["test_loss"], label="Test Loss", linewidth=2, color='#dc2626')
    ax3.set_ylabel("Loss (log scale)", fontsize=11)
    ax3.set_yscale('log')
    ax3.legend(loc="best", fontsize=10)
    ax3.grid(True, alpha=0.3, which='both')
    
    # 4. ROC AUC evolution for all attacks
    for cont in attack_containers:
        _, auc_scores = compute_roc_metrics(cont.train_data, cont.test_data, cont.steps)
        ax4.plot(cont.steps, auc_scores, label=cont.name, linewidth=2)
    
    ax4.axhline(y=0.5, color='gray', linestyle='--', linewidth=2, alpha=0.5, label='Random Classifier')
    ax4.set_xlabel("Training Step (log scale)", fontsize=11)
    ax4.set_ylabel("ROC AUC", fontsize=11)
    ax4.legend(loc="best", fontsize=9)
    ax4.grid(True, alpha=0.3, which='both')
    ax4.set_ylim(0, 1.05)
    
    fig.text(0.02, 0.01, f"Model: {cfg.name}", fontsize=9, alpha=0.6)
    fig.tight_layout()
    fig.savefig(pk.IMAGE_FOLDER / f"combined_training_and_attack_evolution.png", dpi=150)
    plt.close(fig)

def plot_combined_models_superplot(models_data, architecture, dataset, pk):
    """Plot combined training and attack metrics for all models with same architecture and dataset."""
    n_models = len(models_data)
    fig, axes = plt.subplots(4, n_models, figsize=(6 * n_models, 14), sharex='col', sharey='row')
    
    # Handle single model case
    if n_models == 1:
        axes = axes.reshape(-1, 1)
    
    for col_idx, (model_name, cfg, attack_containers) in enumerate(models_data):
        # Load training metrics
        pk.set_params({"model": model_name})
        if not pk.TRAIN_METRICS.exists():
            continue
        
        metrics = json.loads(pk.TRAIN_METRICS.read_text())
        flat_metrics = [_flatten_dict(m) for m in metrics]
        df = pl.DataFrame(flat_metrics)
        
        # Row 0: Accuracy
        ax = axes[0, col_idx]
        ax.plot(df["step"], df["train_accuracy"], label="Train", linewidth=2, color='#2563eb')
        ax.plot(df["step"], df["test_accuracy"], label="Test", linewidth=2, color='#dc2626')
        ax.set_ylabel("Accuracy" if col_idx == 0 else "", fontsize=11)
        ax.set_title(model_name.replace(f"{dataset}_{architecture}_", ""), fontsize=10, fontweight='bold')
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3, which='both')
        ax.set_ylim(0, 1.05)
        ax.set_xscale('log')
        
        # Row 1: Weight Norm
        ax = axes[1, col_idx]
        ax.plot(df["step"], df["norm"], label="Total", linewidth=2, color='#7c3aed')
        ax.plot(df["step"], df["last_layer_norm"], label="Last Layer", linewidth=2, color='#ec4899')
        ax.set_ylabel("Weight Norm (log scale)" if col_idx == 0 else "", fontsize=11)
        ax.set_yscale('log')
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3, which='both')
        
        # Row 2: Loss
        ax = axes[2, col_idx]
        ax.plot(df["step"], df["train_loss"], label="Train", linewidth=2, color='#2563eb')
        ax.plot(df["step"], df["test_loss"], label="Test", linewidth=2, color='#dc2626')
        ax.set_ylabel("Loss (log scale)" if col_idx == 0 else "", fontsize=11)
        ax.set_yscale('log')
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3, which='both')
        
        # Row 3: ROC AUC
        ax = axes[3, col_idx]
        for cont in attack_containers:
            _, auc_scores = compute_roc_metrics(cont.train_data, cont.test_data, cont.steps)
            ax.plot(cont.steps, auc_scores, label=cont.name, linewidth=1.5)
        
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=2, alpha=0.5, label='Random')
        ax.set_xlabel("Training Step (log scale)", fontsize=11)
        ax.set_ylabel("ROC AUC" if col_idx == 0 else "", fontsize=11)
        ax.legend(loc="best", fontsize=7)
        ax.grid(True, alpha=0.3, which='both')
        ax.set_ylim(0, 1.05)
    
    fig.suptitle(f"Combined Analysis: {architecture} on {dataset}", fontsize=14, fontweight='bold')
    fig.tight_layout()
    
    # Save to GENERAL model path
    pk.set_params({"model": "GENERAL"})
    fig.savefig(pk.IMAGE_FOLDER / f"superplot_{architecture}_{dataset}.png", dpi=150)
    plt.close(fig)

@dataclass
class AttackContainer:
    name: str
    prefix: str
    steps: list[int]
    train_data: torch.Tensor
    test_data: torch.Tensor

def visualize(cfgs: list[TrainConfig]):
    logger = get_logger()
    pk = get_path_keeper()

    config_lookup = {cfg.name: cfg for cfg in cfgs}
    mia_container: dict[str, list[AttackContainer]] = {}

    # Data Preparation
    for cfg in cfgs:
        pk.set_params({"model": cfg.name})
        container: list[AttackContainer] = []

        if (not (pk.ATTACK_FOLDER / "mia_threshold.pt").exists()):
            logger.info("No attack data found for MIA Threshold. Skipping evaluation.", extra={"model": cfg.name})
            return
        data = torch.load(pk.ATTACK_FOLDER / "mia_threshold.pt")

        container.append(AttackContainer(
            name="Probability of Correct Class",
            prefix="mia_probability_threshold",
            steps=data["steps"],
            train_data=data["train_probs"].cpu(),
            test_data=data["test_probs"].cpu()
        ))
        container.append(AttackContainer(
            name="Logit of Correct Class",
            prefix="mia_logit_threshold",
            steps=data["steps"],
            train_data=data["train_logits"].cpu(),
            test_data=data["test_logits"].cpu()
        ))
        container.append(AttackContainer(
            name="CrossEntropy Loss",
            prefix="mia_ce_loss_threshold",
            steps=data["steps"],
            train_data=-1 * data["train_ce_losses"].cpu(),
            test_data=-1 * data["test_ce_losses"].cpu()
        ))
        container.append(AttackContainer(
            name="MSE Loss",
            prefix="mia_mse_loss_threshold",
            steps=data["steps"],
            train_data=-1 * data["train_mse_losses"].cpu(),
            test_data=-1 * data["test_mse_losses"].cpu()
        ))
    
        if (not (pk.ATTACK_FOLDER / "mia_rmia.pt").exists()):
            logger.info("No attack data found for MIA RMIA Scores. Skipping evaluation.", extra={"model": cfg.name})
            return
        data = torch.load(pk.ATTACK_FOLDER / "mia_rmia.pt")
        container.append(AttackContainer(
            name="RMIA Score",
            prefix="mia_rmia_score",
            steps=data["steps"],
            train_data=data["train_scores"].cpu(),
            test_data=data["test_scores"].cpu()
        ))
        mia_container[cfg.name] = container
    
    for model_name, attack_containers in mia_container.items():
        cfg = config_lookup[model_name]
        pk.set_params({"model": cfg.name})
        plot_all_attacks_roc_curves(attack_containers, cfg, pk)
        plot_all_attacks_score_distributions(attack_containers, cfg, pk)
        plot_training_and_attack_evolution(attack_containers, cfg, pk)

        for cont in attack_containers:
            tpr_at_fprs, auc_scores = compute_roc_metrics(cont.train_data, cont.test_data, cont.steps)
            plot_distributions(cont.train_data[-1], cont.test_data[-1], cont.steps[-1], cfg, pk, cont.name, cont.prefix)
            plot_roc_curve(cont.train_data[-1], cont.test_data[-1], cont.steps[-1], cfg, pk, cont.prefix)
            plot_evolution(cont.train_data, cont.test_data, cont.steps, cfg, pk, cont.name, cont.prefix)
            plot_tpr_at_fpr(tpr_at_fprs, cont.steps, cfg, pk, cont.prefix)
            plot_auc_evolution(auc_scores, cont.steps, cfg, pk, cont.prefix)
    
    # Create superplots grouped by architecture and dataset
    grouped_models = {}
    for model_name, attack_containers in mia_container.items():
        cfg = config_lookup[model_name]
        # Extract architecture (model type) and dataset from config
        architecture = cfg.model.upper()
        dataset = cfg.dataset.name.upper()
        key = (architecture, dataset)
        
        if key not in grouped_models:
            grouped_models[key] = []
        grouped_models[key].append((model_name, cfg, attack_containers))
    
    # Generate superplot for each group
    for (architecture, dataset), models_data in grouped_models.items():
        if len(models_data) > 1:  # Only create superplot if there are multiple models
            # Sort models by name
            models_data_sorted = sorted(models_data, key=lambda x: x[0])
            plot_combined_models_superplot(models_data_sorted, architecture, dataset, pk)
