"""RNC1 analysis for model c9a3105bba4a4fe499b1e6ce139d4c85 at step 100k.

Steps:
  1. Load checkpoint 100k of model c9a3105bba4a4fe499b1e6ce139d4c85 (v5.0.0-metric).
  2. Run inference for every MNIST datapoint and capture the penultimate layer
     representation (input to the last linear layer, i.e. fc2 output after ReLU).
  3. Compute RNC1 on the resulting representations.
  4. For each class, compute the class mean on the train set.
  5. Plot train and test distributions (PCA-projected to 2D) with class means.
"""

import json
from pathlib import Path
import sys
import io
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

import mlflow
from privacy_and_grokking.utils.logger import Logger
from privacy_and_grokking.config import TrainConfig
from privacy_and_grokking.metrics.neural_collapse import compute_rnc1
from privacy_and_grokking.utils.mlflow import setup_mlflow

class PrintCapture(io.StringIO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.original_stdout = sys.stdout

    def write(self, s):
        self.original_stdout.write(s)
        super().write(s)

    def flush(self):
        self.original_stdout.flush()
        super().flush()

print_capture = PrintCapture()
sys.stdout = print_capture

Logger().setup()  # required before any project code calls Logger.get()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# RUN_ID = "c9a3105bba4a4fe499b1e6ce139d4c85"
RUN_ID = "86388ca2375c4189b8e80658c770a72f"
CHECKPOINT_STEP = 350
TRAIN_MODELS = False

from privacy_and_grokking.utils.mlflow import TRACKING_URI
mlflow.set_tracking_uri(TRACKING_URI)
CHECKPOINT_PATH = mlflow.artifacts.download_artifacts(
    run_id=RUN_ID, artifact_path=f"checkpoints/{CHECKPOINT_STEP}/model.pth"
)
CONFIG_PATH = mlflow.artifacts.download_artifacts(
    run_id=RUN_ID, artifact_path="training_config.json"
)

temp_dir = tempfile.TemporaryDirectory()
OUT_DIR = Path(temp_dir.name)

# ---------------------------------------------------------------------------
# Load config and build datasets
# ---------------------------------------------------------------------------
with open(CONFIG_PATH) as f:
    config_dict = json.load(f)

config = TrainConfig.model_validate(config_dict)

print("Building MNIST datasets...")
data_container = config.data()  # applies train_size=1000 subsetting and masking

train_dataset = data_container.train   # 1000 class-balanced training samples
test_dataset = data_container.test     # full 10k MNIST test set
num_classes = data_container.num_classes

print(f"  Train samples : {len(train_dataset)}")   # type: ignore[arg-type]
print(f"  Test samples  : {len(test_dataset)}")    # type: ignore[arg-type]

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = config.model(
    input_dim=data_container.input_shape,
    num_classes=num_classes,
)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
model.to(device)
model.eval()

# ---------------------------------------------------------------------------
# Feature extraction — penultimate layer (input to fc3 = output of fc2+ReLU)
# ---------------------------------------------------------------------------

def extract_features(
    dataset,
    model,
    device: torch.device,
    batch_size: int = 512,
    normalization = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (features [N, 200], logits [N, C], labels [N]) for the given dataset."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    features_list: list[torch.Tensor] = []
    logits_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    
    if normalization is not None:
        mean = torch.tensor(normalization.mean, device=device).view(-1, 1, 1)
        std = torch.tensor(normalization.std, device=device).view(-1, 1, 1)

    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            if normalization is not None:
                imgs = (imgs - mean) / std
            # MLP forward up to the last layer input
            y, z = model(imgs, verbose=True)
            features_list.append(z.cpu())
            logits_list.append(y.cpu())
            labels_list.append(lbls.cpu() if isinstance(lbls, torch.Tensor) else torch.tensor(lbls))

    return torch.cat(features_list), torch.cat(logits_list), torch.cat(labels_list)


print("Extracting train features...")
norm_to_apply = data_container.normalization
train_features, train_logits, train_labels = extract_features(train_dataset, model, device, normalization=norm_to_apply)
print("Extracting test features...")
test_features, test_logits, test_labels = extract_features(test_dataset, model, device, normalization=norm_to_apply)

print(f"  train_features: {train_features.shape}")
print(f"  test_features : {test_features.shape}")

# ---------------------------------------------------------------------------
# Model Accuracy
# ---------------------------------------------------------------------------
train_preds = train_logits.argmax(dim=1)
train_acc = (train_preds == train_labels).float().mean().item()
test_preds = test_logits.argmax(dim=1)
test_acc = (test_preds == test_labels).float().mean().item()
print(f"\nOverall Train accuracy : {train_acc:.2%}")
print(f"Overall Test accuracy  : {test_acc:.2%}")

print("\nPer-class Accuracy:")
for c in range(num_classes):
    train_mask = train_labels == c
    test_mask = test_labels == c
    
    c_train_acc = (train_preds[train_mask] == train_labels[train_mask]).float().mean().item() if train_mask.sum() > 0 else float('nan')
    c_test_acc = (test_preds[test_mask] == test_labels[test_mask]).float().mean().item() if test_mask.sum() > 0 else float('nan')
    
    print(f"  Class {c} -> Train: {c_train_acc:.2%} | Test: {c_test_acc:.2%}")
print()

def get_canary_mask(dataset) -> torch.Tensor:
    if hasattr(dataset, "dataset") and hasattr(dataset.dataset, "canary_indices"):
        canary_ds = dataset.dataset
        indices = dataset.indices
    elif hasattr(dataset, "canary_indices"):
        canary_ds = dataset
        indices = range(len(dataset))
    else:
        return torch.zeros(len(dataset), dtype=torch.bool)
        
    if len(canary_ds.canary_indices) == 0:
        return torch.zeros(len(dataset), dtype=torch.bool)
        
    canary_set = set(canary_ds.canary_indices.tolist())
    is_canary = []
    for i in indices:
        raw_idx = canary_ds.subset_indices[i]
        raw_idx_val = raw_idx.item() if torch.is_tensor(raw_idx) else raw_idx
        is_canary.append(int(raw_idx_val) in canary_set)
    return torch.tensor(is_canary, dtype=torch.bool)

canary_mask = get_canary_mask(train_dataset)
train_f_normal = train_features[~canary_mask].float()
train_l_normal = train_labels[~canary_mask]
train_f_canary = train_features[canary_mask].float()
train_l_canary = train_labels[canary_mask]
train_y_canary = train_logits[canary_mask]

# --- Generate Test Canaries ---
from privacy_and_grokking.datasets.canaries.uniform_noise import UniformNoiseCanary
from torch.utils.data import DataLoader
test_f_canary = torch.empty(0)
test_l_canary = torch.empty(0)
test_y_canary = torch.empty(0)
if len(train_f_canary) > 0:
    print("Generating unseen test canaries for evaluation...")
    noise_gen = UniformNoiseCanary(dim=(1, 28, 28))
    tcf_list, tcl_list, tcy_list = [], [], []
    sub_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    with torch.no_grad():
        for imgs, lbls in sub_loader:
            noisy_imgs = torch.stack([noise_gen(img) for img in imgs]).to(device)
            if norm_to_apply is not None:
                mean = torch.tensor(norm_to_apply.mean, device=device).view(-1, 1, 1)
                std = torch.tensor(norm_to_apply.std, device=device).view(-1, 1, 1)
                noisy_imgs = (noisy_imgs - mean) / std
            y, z = model(noisy_imgs, verbose=True)
            tcf_list.append(z.cpu())
            tcl_list.append(lbls.cpu())
            tcy_list.append(y.cpu())
            if sum(len(b) for b in tcf_list) >= len(train_f_canary): break
    test_f_canary = torch.cat(tcf_list)
    test_l_canary = torch.cat(tcl_list)
    test_y_canary = torch.cat(tcy_list)

print(f"  train_features_normal: {train_f_normal.shape}")
print(f"  train_features_canary: {train_f_canary.shape}")
if len(train_f_canary) > 0:
    print(f"  test_features_canary:  {test_f_canary.shape}")

# ---------------------------------------------------------------------------
# RNC1
# ---------------------------------------------------------------------------
rnc1_train = compute_rnc1(train_features, train_labels)
rnc1_test  = compute_rnc1(test_features,  test_labels)
print(f"\nRNC1 (train set) : {rnc1_train:.6f}")
print(f"RNC1 (test set)  : {rnc1_test:.6f}")

# ---------------------------------------------------------------------------
# Representation Scale Distribution
# ---------------------------------------------------------------------------
print("\n--- Representation Scale Analysis ---")
train_norms = train_f_normal.norm(dim=1).numpy()
test_norms = test_features.float().norm(dim=1).numpy()

from scipy.stats import shapiro, skew, kurtosis

def get_dist_text(data):
    if len(data) < 3: return "N/A"
    std_val = np.std(data)
    if std_val < 1e-5:
        return f"μ={np.mean(data):.2f}, σ~0 (Dirac Delta)"
    
    sample_data = data if len(data) <= 5000 else np.random.choice(data, 5000, replace=False)
    s, p = shapiro(sample_data)
    sk = skew(data)
    
    if p > 0.05:
        dist_type = "Normal"
    else:
        dist_type = "Right-Skewed" if sk > 1.0 else ("Left-Skewed" if sk < -1.0 else "Non-Normal")
    return f"μ={np.mean(data):.2f}, σ={std_val:.2f} | {dist_type} (p={p:.4f})"

fig_norm, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()

for c in range(num_classes):
    ax = axes[c]
    
    c_train_norms = train_f_normal[train_l_normal == c].norm(dim=1).numpy()
    c_test_norms = test_features.float()[test_labels == c].norm(dim=1).numpy()
    
    if len(c_train_norms) > 0:
        ax.hist(c_train_norms, bins=30, alpha=0.5, density=True, color='blue', label='Train')
    if len(c_test_norms) > 0:
        ax.hist(c_test_norms, bins=30, alpha=0.5, density=True, color='orange', label='Test')
    
    txt_train = "Train: " + get_dist_text(c_train_norms)
    txt_test = "Test:  " + get_dist_text(c_test_norms)
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    info_str = f"{txt_train}\n{txt_test}"
    ax.text(0.05, 0.95, info_str, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=props)
            
    ax.set_title(f"Class {c}")
    if c >= 5:
        ax.set_xlabel('L2 Norm (Scale)')
    if c == 0 or c == 5:
        ax.set_ylabel('Density')
    if c == 0:
        ax.legend(loc='lower right', fontsize=8)

fig_norm.suptitle(f"Per-Class Representation Scale Distribution\nModel {RUN_ID[:8]}… | step {CHECKPOINT_STEP:,}", fontsize=14)
plt.tight_layout()
out_path_norm = OUT_DIR / "rnc1_scale_dist_per_class.png"
fig_norm.savefig(out_path_norm, dpi=150, bbox_inches="tight")
print(f"Per-Class Scale distribution plot saved to: {out_path_norm}")
# plt.show()

# ---------------------------------------------------------------------------
# Class means on the TRAIN set (in the scaled feature space, as per RNC1 def)
# ---------------------------------------------------------------------------
train_f = train_features.float()
B_g = train_f.norm(dim=1).max()          # global max-norm (same normaliser as RNC1)
train_f_scaled = train_f / B_g

class_means_scaled = torch.zeros(num_classes, train_f_scaled.shape[1])
for c in range(num_classes):
    mask = train_labels == c
    class_means_scaled[c] = train_f_scaled[mask].mean(dim=0)

# ---------------------------------------------------------------------------
# PCA projection to 2D for visualisation
# (fit on all data so train and test share the same embedding space)
# ---------------------------------------------------------------------------
all_features_np = torch.cat([train_f, test_features.float()]).numpy()
pca = PCA(n_components=2, random_state=0)
pca.fit(all_features_np)

train_2d = pca.transform(train_f.numpy())
test_2d  = pca.transform(test_features.float().numpy())
# Project class means back to original space, then to PCA
means_2d = pca.transform((class_means_scaled * B_g).numpy())

explained = pca.explained_variance_ratio_
print(f"\nPCA explained variance: PC1={explained[0]:.1%}, PC2={explained[1]:.1%}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
cmap = plt.cm.get_cmap("tab10", num_classes)
colors = [cmap(c) for c in range(num_classes)]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

splits = [
    ("Train  (n=1 000)", train_2d, train_labels.numpy(), rnc1_train),
    ("Test  (n=10 000)", test_2d,  test_labels.numpy(),  rnc1_test),
]

for ax, (title, feats_2d, labels_np, rnc1_val) in zip(axes, splits):
    for c in range(num_classes):
        mask = labels_np == c
        ax.scatter(
            feats_2d[mask, 0],
            feats_2d[mask, 1],
            color=colors[c],
            alpha=0.45,
            s=12,
            linewidths=0,
            label=f"class {c}",
        )
    # Class means (from train set) overlaid on both panels
    for c in range(num_classes):
        ax.scatter(
            means_2d[c, 0],
            means_2d[c, 1],
            color=colors[c],
            marker="*",
            s=250,
            edgecolors="black",
            linewidths=0.6,
            zorder=10,
        )
    ax.set_title(f"{title}\nRNC1 = {rnc1_val:.5f}", fontsize=11)
    ax.set_xlabel(f"PC 1  ({explained[0]:.1%})")
    ax.set_ylabel(f"PC 2  ({explained[1]:.1%})")

# Shared legend (classes + mean marker)
handles, lbls = axes[0].get_legend_handles_labels()
import matplotlib.lines as mlines
mean_handle = mlines.Line2D(
    [], [], color="grey", marker="*", linestyle="None",
    markersize=10, markeredgecolor="black", label="class mean (train)"
)
fig.legend(
    handles + [mean_handle],
    lbls + ["class mean (train)"],
    loc="lower center",
    ncol=6,
    fontsize=9,
    bbox_to_anchor=(0.5, -0.04),
)

fig.suptitle(
    f"MNIST penultimate-layer representations\n"
    f"Model {RUN_ID[:8]}…  |  step {CHECKPOINT_STEP:,}  |  ★ = train class mean",
    fontsize=12,
    y=1.01,
)
plt.tight_layout()

out_path = OUT_DIR / "rnc1_analysis.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved to: {out_path}")
# plt.show()

# ---------------------------------------------------------------------------
# Density of distance-to-class-mean for each class  (train vs test)
# ---------------------------------------------------------------------------
from scipy.stats import gaussian_kde as _gaussian_kde
import numpy as np

class _DummyKDE:
    def __call__(self, x):
        if hasattr(x, "shape") and len(x.shape) > 1:
            return np.zeros(x.shape[1])
        return np.zeros(np.shape(x)[-1] if np.shape(x) else 1)

def gaussian_kde(*args, **kwargs):
    try:
        # Also prevent some common warnings by pre-checking 1D variance
        data = np.asarray(args[0])
        if data.ndim == 1 and np.std(data) < 1e-8:
            return _DummyKDE()
        return _gaussian_kde(*args, **kwargs)
    except Exception as e:
        print(f"Warning: KDE failed ({e}), returning dummy KDE")
        return _DummyKDE()

# Compute class means in the UNSCALED feature space using the TRAIN set
class_means_unscaled = torch.zeros(num_classes, train_f.shape[1])
for c in range(num_classes):
    mask_c = train_l_normal == c
    if mask_c.sum() > 0:
        class_means_unscaled[c] = train_f_normal[mask_c].mean(dim=0)
    else:
        # fallback to all train if no normal samples for this class
        mask_c_all = train_labels == c
        class_means_unscaled[c] = train_f[mask_c_all].mean(dim=0)

def distances_to_class_mean(
    features: torch.Tensor,
    labels: torch.Tensor,
    class_means: torch.Tensor,
) -> dict[int, np.ndarray]:
    """L2 distance of each sample to its class mean (using train-set means)."""
    dists: dict[int, np.ndarray] = {}
    for c in range(num_classes):
        mask_c = labels == c
        if mask_c.sum() == 0:
            continue
        diff = features[mask_c].float() - class_means[c]
        dists[c] = diff.norm(dim=1).numpy()
    return dists

train_dists = distances_to_class_mean(train_f_normal,       train_l_normal, class_means_unscaled)
test_dists  = distances_to_class_mean(test_features.float(), test_labels,  class_means_unscaled)
canary_dists = distances_to_class_mean(train_f_canary,      train_l_canary, class_means_unscaled)
test_canary_dists = distances_to_class_mean(test_f_canary, test_l_canary, class_means_unscaled)

train_test_dists = {}
for c in range(num_classes):
    train_c = train_dists.get(c, np.array([]))
    test_c = test_dists.get(c, np.array([]))
    train_test_dists[c] = np.concatenate([train_c, test_c]) if len(train_c) > 0 or len(test_c) > 0 else np.array([])

# Combined grid for KDE evaluation
all_dist_values = np.concatenate(
    list(train_dists.values()) + list(test_dists.values()) + list(canary_dists.values()) + list(test_canary_dists.values())
)
x_min, x_max = 0.0, float(np.percentile(all_dist_values, 99.5))
x_grid = np.linspace(x_min, x_max, 400)

# ---------------------------------------------------------------------------
# Combined density of distance-to-class-mean
# ---------------------------------------------------------------------------
fig_comb, ax_comb = plt.subplots(figsize=(8, 5))

all_train_flat = np.concatenate(list(train_dists.values())) if train_dists else np.array([])
all_test_flat = np.concatenate(list(test_dists.values())) if test_dists else np.array([])
all_canary_flat = np.concatenate(list(canary_dists.values())) if canary_dists else np.array([])

all_train_test_flat = np.concatenate([all_train_flat, all_test_flat]) if len(all_train_flat) > 0 or len(all_test_flat) > 0 else np.array([])
all_test_canary_flat = np.concatenate(list(test_canary_dists.values())) if test_canary_dists else np.array([])

for dists_flat, label, ls, color, alpha in [
    (all_train_test_flat, "train+test", "-", "purple", 0.85),
    (all_train_flat, "train", "-",  "blue", 0.85),
    (all_test_flat,  "test",  "--", "orange", 0.65),
    (all_canary_flat, "train_canary", ":", "red", 0.85),
    (all_test_canary_flat, "test_canary", ":", "brown", 0.85),
]:
    if len(dists_flat) < 2:
        continue
    kde = gaussian_kde(dists_flat, bw_method="scott")
    ax_comb.plot(x_grid, kde(x_grid), linestyle=ls, color=color, alpha=alpha, label=label, lw=1.8)
    ax_comb.fill_between(x_grid, kde(x_grid), alpha=alpha * 0.18, color=color)

ax_comb.set_title(f"Combined Distance to Class Mean\nModel {RUN_ID[:8]}… | step {CHECKPOINT_STEP:,}", fontsize=11)
ax_comb.set_xlabel("‖h − μ_c‖₂")
ax_comb.set_ylabel("density")
ax_comb.legend(fontsize=10, frameon=False)
ax_comb.set_xlim(x_min, x_max)
ax_comb.set_ylim(bottom=0)
fig_comb.tight_layout()

out_path_comb = OUT_DIR / "rnc1_dist_density_combined.png"
fig_comb.savefig(out_path_comb, dpi=150, bbox_inches="tight")
print(f"Combined density plot saved to: {out_path_comb}")
# plt.show()

fig2, axes2 = plt.subplots(2, 5, figsize=(18, 7), sharey=False)
axes2 = axes2.flatten()

for c in range(num_classes):
    ax = axes2[c]
    color = colors[c]

    for dists, label, ls, alpha, lw, c_color in [
        (train_test_dists, "train+test", "-", 0.95, 2.5, color),
        (train_dists, "train", "-.",  0.85, 1.8, color),
        (test_dists,  "test",  "--", 0.65, 1.8, color),
        (canary_dists, "train_canary", ":", 0.85, 1.8, color),
        (test_canary_dists, "test_canary", ":", 0.85, 1.8, "brown"),
    ]:
        if c not in dists or len(dists[c]) < 2:
            continue
        kde = gaussian_kde(dists[c], bw_method="scott")
        ax.plot(x_grid, kde(x_grid), linestyle=ls, color=c_color, alpha=alpha, label=label, lw=lw)
        ax.fill_between(x_grid, kde(x_grid), alpha=alpha * 0.18, color=c_color)

    ax.set_title(f"Class {c}", fontsize=10)
    ax.set_xlabel("‖h − μ_c‖₂")
    ax.set_ylabel("density" if c % 5 == 0 else "")
    ax.legend(fontsize=8, frameon=False)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(bottom=0)

fig2.suptitle(
    f"Distance to class mean (train-set means)\n"
    f"Model {RUN_ID[:8]}…  |  step {CHECKPOINT_STEP:,}  |  solid=train+test, dashdot=train, dashed=test, dotted=canary",
    fontsize=12,
)
plt.tight_layout()

out_path2 = OUT_DIR / "rnc1_dist_density.png"
plt.savefig(out_path2, dpi=150, bbox_inches="tight")
print(f"Density plot saved to: {out_path2}")
# plt.show()

# ---------------------------------------------------------------------------
# Membership Inference Attack (MIA)
# ---------------------------------------------------------------------------

def get_gaussian_intersection(mu1, std1, mu2, std2):
    """Finds the intersection points of two Gaussians."""
    a = 1.0 / (2*std1**2) - 1.0 / (2*std2**2)
    b = mu2 / (std2**2) - mu1 / (std1**2)
    c = mu1**2 / (2*std1**2) - mu2**2 / (2*std2**2) - np.log(std2/std1)
    
    if np.abs(a) < 1e-9:
        if np.abs(b) < 1e-9:
            return []
        return [-c / b]
        
    delta = b**2 - 4*a*c
    if delta < 0:
        return []
    
    r1 = (-b - np.sqrt(delta)) / (2*a)
    r2 = (-b + np.sqrt(delta)) / (2*a)
    return sorted([r1, r2])

def evaluate_mia(train_d, test_d):
    """Evaluates MIA on a set of train and test distances."""
    if len(train_d) < 2 or len(test_d) < 2:
        return None
    
    mu_train, std_train = np.mean(train_d), np.std(train_d)
    mu_test, std_test = np.mean(test_d), np.std(test_d)
    
    target_fprs = [0.01, 0.05, 0.10]
    fixed_fpr_results = {}
    for fpr_target in target_fprs:
        if mu_train < mu_test:
            bound_val = np.percentile(test_d, fpr_target * 100)
            tpr_val = np.mean(train_d < bound_val)
        else:
            bound_val = np.percentile(test_d, 100 - fpr_target * 100)
            tpr_val = np.mean(train_d > bound_val)
        fixed_fpr_results[fpr_target] = {"bound": bound_val, "tpr": tpr_val}
    
    # If standard deviation is effectively zero (e.g. overfitted train data),
    # the gaussian intersection logic fails or returns unreliable intervals.
    # We should only classify exactly the mean value as Train.
    if std_train < 1e-5:
        eps = 1e-4
        bounds = (mu_train - eps, mu_train + eps)
        train_preds = (train_d >= bounds[0]) & (train_d <= bounds[1])
        test_preds = (test_d >= bounds[0]) & (test_d <= bounds[1])
        
        tpr = np.mean(train_preds)
        fpr = np.mean(test_preds)
        acc = (np.sum(train_preds) + np.sum(~test_preds)) / (len(train_d) + len(test_d))
        return {
            "mu_train": mu_train, "std_train": std_train,
            "mu_test": mu_test, "std_test": std_test,
            "bounds": bounds,
            "tpr": tpr, "fpr": fpr, "acc": acc,
            "fixed_fprs": fixed_fpr_results
        }
        
    roots = get_gaussian_intersection(mu_train, std_train, mu_test, std_test)
    if not roots:
        return None
        
    if len(roots) == 1:
        bound = roots[0]
        p_train_left = np.exp(-0.5 * ((bound - 1 - mu_train)/std_train)**2) / std_train
        p_test_left = np.exp(-0.5 * ((bound - 1 - mu_test)/std_test)**2) / std_test
        if p_train_left > p_test_left:
            bounds = (None, bound)
            train_preds = train_d < bound
            test_preds = test_d < bound
        else:
            bounds = (bound, None)
            train_preds = train_d > bound
            test_preds = test_d > bound
    else:
        r1, r2 = roots
        midpoint = (r1 + r2) / 2
        p_train_mid = np.exp(-0.5 * ((midpoint - mu_train)/std_train)**2) / std_train
        p_test_mid = np.exp(-0.5 * ((midpoint - mu_test)/std_test)**2) / std_test
        
        if p_train_mid > p_test_mid:
            bounds = (r1, r2)
            train_preds = (train_d >= r1) & (train_d <= r2)
            test_preds = (test_d >= r1) & (test_d <= r2)
        else:
            bounds = (r2, r1) # flipped to indicate outside
            train_preds = (train_d < r1) | (train_d > r2)
            test_preds = (test_d < r1) | (test_d > r2)
        
    tpr = np.mean(train_preds)
    fpr = np.mean(test_preds)
    acc = (np.sum(train_preds) + np.sum(~test_preds)) / (len(train_d) + len(test_d))
    return {
        "mu_train": mu_train, "std_train": std_train,
        "mu_test": mu_test, "std_test": std_test,
        "bounds": bounds,
        "tpr": tpr, "fpr": fpr, "acc": acc,
        "fixed_fprs": fixed_fpr_results
    }

# Global MIA
global_mia = evaluate_mia(all_train_flat, all_test_flat)

# Per-class MIA
class_mias = {}
for c in range(num_classes):
    train_c = train_dists.get(c, np.array([]))
    test_c = test_dists.get(c, np.array([]))
    class_mias[c] = evaluate_mia(train_c, test_c)

print("\n--- Membership Inference Attack ---")
if global_mia:
    b = global_mia['bounds']
    if b[0] is None:
        b_str = f"x < {b[1]:.3f}"
    elif b[1] is None:
        b_str = f"x > {b[0]:.3f}"
    else:
        b_str = f"[{b[0]:.3f}, {b[1]:.3f}]" if b[0] < b[1] else f"x < {b[1]:.3f} OR x > {b[0]:.3f}"
    print(f"Global MIA (Intersection) -> Bounds: {b_str}  |  Acc: {global_mia['acc']:.1%}  |  TPR: {global_mia['tpr']:.1%}  |  FPR: {global_mia['fpr']:.1%}")
    ff = global_mia['fixed_fprs']
    print(f"Global MIA (Fixed FPR)    -> TPR@1%FPR: {ff[0.01]['tpr']:.1%}  |  TPR@5%FPR: {ff[0.05]['tpr']:.1%}  |  TPR@10%FPR: {ff[0.10]['tpr']:.1%}")

# Plot MIA
fig_mia, axes_mia = plt.subplots(2, 6, figsize=(22, 7))
axes_mia = axes_mia.flatten()

# Plot Global MIA
ax_g = axes_mia[0]
if global_mia:
    kde_train = gaussian_kde(all_train_flat, bw_method="scott")
    kde_test = gaussian_kde(all_test_flat, bw_method="scott")
    ax_g.plot(x_grid, kde_train(x_grid), color="blue", label="train")
    ax_g.plot(x_grid, kde_test(x_grid), color="orange", linestyle="--", label="test")
    
    b = global_mia['bounds']
    if b[0] is None:
        ax_g.axvline(b[1], color='k', linestyle=':')
        ax_g.axvspan(x_min, b[1], color='blue', alpha=0.1)
    elif b[1] is None:
        ax_g.axvline(b[0], color='k', linestyle=':')
        ax_g.axvspan(b[0], x_max, color='blue', alpha=0.1)
    elif b[0] <= b[1]:
        ax_g.axvline(b[0], color='k', linestyle=':')
        ax_g.axvline(b[1], color='k', linestyle=':')
        ax_g.axvspan(b[0], b[1], color='blue', alpha=0.1)
    else:
        ax_g.axvline(b[0], color='k', linestyle=':')
        ax_g.axvline(b[1], color='k', linestyle=':')
        ax_g.axvspan(x_min, b[1], color='blue', alpha=0.1)
        ax_g.axvspan(b[0], x_max, color='blue', alpha=0.1)
    
    ff = global_mia['fixed_fprs']
    ax_g.set_title(f"Global\nAcc: {global_mia['acc']:.1%} | TPR: {global_mia['tpr']:.1%} | FPR: {global_mia['fpr']:.1%}\nTPR @ 1%: {ff[0.01]['tpr']:.1%} | 5%: {ff[0.05]['tpr']:.1%} | 10%: {ff[0.10]['tpr']:.1%}", fontsize=9)
    ax_g.legend(fontsize=8, frameon=False)

for c in range(num_classes):
    ax = axes_mia[c + 1]
    mia_c = class_mias.get(c)
    if not mia_c:
        continue
    
    train_c = train_dists.get(c, np.array([]))
    test_c = test_dists.get(c, np.array([]))
    if len(train_c) > 1 and len(test_c) > 1:
        kde_train = gaussian_kde(train_c, bw_method="scott")
        kde_test = gaussian_kde(test_c, bw_method="scott")
        ax.plot(x_grid, kde_train(x_grid), color="blue")
        ax.plot(x_grid, kde_test(x_grid), color="orange", linestyle="--")
        
        b = mia_c['bounds']
        if b[0] is None:
            ax.axvline(b[1], color='k', linestyle=':')
            ax.axvspan(x_min, b[1], color='blue', alpha=0.1)
        elif b[1] is None:
            ax.axvline(b[0], color='k', linestyle=':')
            ax.axvspan(b[0], x_max, color='blue', alpha=0.1)
        elif b[0] <= b[1]:
            ax.axvline(b[0], color='k', linestyle=':')
            ax.axvline(b[1], color='k', linestyle=':')
            ax.axvspan(b[0], b[1], color='blue', alpha=0.1)
        else:
            ax.axvline(b[0], color='k', linestyle=':')
            ax.axvline(b[1], color='k', linestyle=':')
            ax.axvspan(x_min, b[1], color='blue', alpha=0.1)
            ax.axvspan(b[0], x_max, color='blue', alpha=0.1)
            
        ff = mia_c['fixed_fprs']
        ax.set_title(f"Class {c}\nAcc: {mia_c['acc']:.1%} | TPR: {mia_c['tpr']:.1%} | FPR: {mia_c['fpr']:.1%}\nTPR @ 1%: {ff[0.01]['tpr']:.1%} | 5%: {ff[0.05]['tpr']:.1%} | 10%: {ff[0.10]['tpr']:.1%}", fontsize=9)

# Hide the last unused subplot
axes_mia[-1].axis('off')

fig_mia.suptitle(f"MIA Decision Boundaries (Equal Priors Gaussian Intersection)\nModel {RUN_ID[:8]}…  |  step {CHECKPOINT_STEP:,}")
fig_mia.tight_layout()

out_path_mia = OUT_DIR / "rnc1_mia.png"
fig_mia.savefig(out_path_mia, dpi=150, bbox_inches="tight")
print(f"MIA plot saved to: {out_path_mia}")
# plt.show()


# ---------------------------------------------------------------------------
# MIA: Idealized Gaussians Plot
# ---------------------------------------------------------------------------
def plot_idealized_gaussian(ax, mu, std, color, label, linestyle="-"):
    if std > 0:
        x = np.linspace(mu - 4*std, mu + 4*std, 500)
        y = np.exp(-0.5 * ((x - mu) / std)**2) / (std * np.sqrt(2 * np.pi))
        ax.plot(x, y, color=color, label=label, linestyle=linestyle)

fig_gauss, axes_gauss = plt.subplots(2, 6, figsize=(22, 7))
axes_gauss = axes_gauss.flatten()

# Plot Global MIA Gaussians
ax_g = axes_gauss[0]
if global_mia:
    plot_idealized_gaussian(ax_g, global_mia['mu_train'], global_mia['std_train'], "blue", "train")
    plot_idealized_gaussian(ax_g, global_mia['mu_test'], global_mia['std_test'], "orange", "test", "--")
    
    b = global_mia['bounds']
    if b[0] is not None:
        ax_g.axvline(b[0], color='k', linestyle=':')
    if b[1] is not None:
        ax_g.axvline(b[1], color='k', linestyle=':')
        
    ax_g.set_title(f"Global Idealized\n$\\mu_T$={global_mia['mu_train']:.2f}, $\\mu_V$={global_mia['mu_test']:.2f} | FPR: {global_mia['fpr']:.1%}", fontsize=10)
    ax_g.legend(fontsize=8, frameon=False)

for c in range(num_classes):
    ax = axes_gauss[c + 1]
    mia_c = class_mias.get(c)
    if not mia_c:
        continue
    
    plot_idealized_gaussian(ax, mia_c['mu_train'], mia_c['std_train'], "blue", "train")
    plot_idealized_gaussian(ax, mia_c['mu_test'], mia_c['std_test'], "orange", "test", "--")
    
    b = mia_c['bounds']
    if b[0] is not None:
        ax.axvline(b[0], color='k', linestyle=':')
    if b[1] is not None:
        ax.axvline(b[1], color='k', linestyle=':')
        
    ax.set_title(f"Class {c}\n$\\mu_T$={mia_c['mu_train']:.2f}, $\\mu_V$={mia_c['mu_test']:.2f} | FPR: {mia_c['fpr']:.1%}", fontsize=10)

axes_gauss[-1].axis('off')

fig_gauss.suptitle(f"MIA Idealized Gaussian Distributions\nModel {RUN_ID[:8]}…  |  step {CHECKPOINT_STEP:,}")
fig_gauss.tight_layout()

out_path_gauss = OUT_DIR / "rnc1_mia_gaussians.png"
fig_gauss.savefig(out_path_gauss, dpi=150, bbox_inches="tight")
print(f"MIA Gaussians plot saved to: {out_path_gauss}")
# plt.show()


# ---------------------------------------------------------------------------
# Margin Density Plots (Decision Boundaries)
# ---------------------------------------------------------------------------
w = model.classifier().weight.detach().cpu()
_bias = model.classifier().bias
b = _bias.detach().cpu() if _bias is not None else torch.zeros(w.shape[0])

pairs_to_plot = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
fig_margin, axes_margin = plt.subplots(len(pairs_to_plot), 2, figsize=(16, 3.5 * len(pairs_to_plot)), sharex=False)

def plot_margin_violin(ax, f, l, w, b, c_pos, c_neg):
    mask_pos = (l == c_pos)
    mask_neg = (l == c_neg)
    
    w_diff = w[c_pos] - w[c_neg]
    b_diff = b[c_pos] - b[c_neg]
    norm_w_diff = torch.norm(w_diff, p=2)
    
    def get_margins(f_subset):
        return (torch.matmul(f_subset, w_diff) + b_diff) / norm_w_diff

    margins_pos = get_margins(f[mask_pos]).numpy()
    margins_neg = get_margins(f[mask_neg]).numpy()
    
    if len(margins_pos) < 2 or len(margins_neg) < 2:
        return
        
    x_min = min(margins_pos.min(), margins_neg.min()) - 0.5
    x_max = max(margins_pos.max(), margins_neg.max()) + 0.5
    x_grid = np.linspace(x_min, x_max, 500)
    
    kde_pos = gaussian_kde(margins_pos, bw_method="scott")
    kde_neg = gaussian_kde(margins_neg, bw_method="scott")
    
    y_pos = kde_pos(x_grid)
    y_neg = -kde_neg(x_grid)
    
    ax.fill_between(x_grid, 0, y_pos, color="steelblue", alpha=1.0, edgecolor="black", linewidth=0.5, label=f"class {c_pos}")
    ax.fill_between(x_grid, 0, y_neg, color="peru", alpha=1.0, edgecolor="black", linewidth=0.5, label=f"class {c_neg}")
    
    # Draw individual example lines
    step = max(1, len(margins_pos) // 50)
    for m in np.sort(margins_pos)[::step]:
        ax.plot([m, m], [0, kde_pos(m)[0]], color="black", linewidth=0.3, alpha=0.5)
    step = max(1, len(margins_neg) // 50)
    for m in np.sort(margins_neg)[::step]:
        ax.plot([m, m], [0, -kde_neg(m)[0]], color="black", linewidth=0.3, alpha=0.5)
        
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=1.5)
    
    ax.set_yticks([])
    ax.set_xlabel(f"Signed distance to decision boundary (class {c_pos} vs. class {c_neg})")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8, frameon=True)
    ax.grid(True, axis='x', linestyle='-', alpha=0.5)

for i, (c1, c2) in enumerate(pairs_to_plot):
    ax_tr = axes_margin[i, 0]
    ax_te = axes_margin[i, 1]
    
    plot_margin_violin(ax_tr, train_f_normal, train_l_normal, w, b, c1, c2)
    plot_margin_violin(ax_te, test_features.float(), test_labels, w, b, c1, c2)
    
    if i == 0:
        ax_tr.set_title("Train Examples", fontsize=12)
        ax_te.set_title("Unseen Examples", fontsize=12)

fig_margin.suptitle(f"Margins of individual examples (Decision Boundaries)\nModel {RUN_ID[:8]}…  |  step {CHECKPOINT_STEP:,}", fontsize=14)
fig_margin.tight_layout()

out_path_margin = OUT_DIR / "rnc1_margin_plots.png"
fig_margin.savefig(out_path_margin, dpi=150, bbox_inches="tight")
print(f"Margin plots saved to: {out_path_margin}")
# plt.show()


# ---------------------------------------------------------------------------
# Margin Histogram Plots (Decision Boundaries)
# ---------------------------------------------------------------------------
fig_margin_hist, axes_margin_hist = plt.subplots(len(pairs_to_plot), 2, figsize=(16, 3.5 * len(pairs_to_plot)), sharex=False)

def plot_margin_histogram(ax, f, l, w, b, c_pos, c_neg):
    mask_pos = (l == c_pos)
    mask_neg = (l == c_neg)
    
    w_diff = w[c_pos] - w[c_neg]
    b_diff = b[c_pos] - b[c_neg]
    norm_w_diff = torch.norm(w_diff, p=2)
    
    def get_margins(f_subset):
        return (torch.matmul(f_subset, w_diff) + b_diff) / norm_w_diff
        
    margins_pos = get_margins(f[mask_pos]).numpy()
    margins_neg = get_margins(f[mask_neg]).numpy()
    
    if len(margins_pos) < 2 or len(margins_neg) < 2:
        return
        
    x_min = min(margins_pos.min(), margins_neg.min()) - 0.5
    x_max = max(margins_pos.max(), margins_neg.max()) + 0.5
    bins = np.linspace(x_min, x_max, 1500)
    
    counts_pos, edges_pos = np.histogram(margins_pos, bins=bins, density=False)
    ax.bar(edges_pos[:-1], counts_pos, width=np.diff(edges_pos), align="edge", color="steelblue", alpha=1.0, edgecolor="none", label=f"class {c_pos}")
    
    counts_neg, edges_neg = np.histogram(margins_neg, bins=bins, density=False)
    ax.bar(edges_neg[:-1], -counts_neg, width=np.diff(edges_neg), align="edge", color="peru", alpha=1.0, edgecolor="none", label=f"class {c_neg}")
    
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=1.5)
    
    ax.set_yscale('symlog', linthresh=1.0)
    
    # Custom y-ticks to show logarithmic nature
    ax.set_yticks([-1000, -100, -10, 0, 10, 100, 1000])
    ax.set_yticklabels(['1000', '100', '10', '0', '10', '100', '1000'])
    
    ax.set_xlabel(f"Signed distance to decision boundary (class {c_pos} vs. class {c_neg})")
    ax.set_ylabel("Count (Log Scale)")
    ax.legend(fontsize=8, frameon=True)
    ax.grid(True, axis='both', linestyle=':', alpha=0.5)

for i, (c1, c2) in enumerate(pairs_to_plot):
    ax_tr = axes_margin_hist[i, 0]
    ax_te = axes_margin_hist[i, 1]
    
    plot_margin_histogram(ax_tr, train_f_normal, train_l_normal, w, b, c1, c2)
    plot_margin_histogram(ax_te, test_features.float(), test_labels, w, b, c1, c2)
    
    if i == 0:
        ax_tr.set_title("Train Examples (Histograms)", fontsize=12)
        ax_te.set_title("Unseen Examples (Histograms)", fontsize=12)

fig_margin_hist.suptitle(f"Margins of individual examples (Histograms)\nModel {RUN_ID[:8]}…  |  step {CHECKPOINT_STEP:,}", fontsize=14)
fig_margin_hist.tight_layout()

out_path_margin_hist = OUT_DIR / "rnc1_margin_hist_plots.png"
fig_margin_hist.savefig(out_path_margin_hist, dpi=150, bbox_inches="tight")
print(f"Margin histogram plots saved to: {out_path_margin_hist}")
# plt.show()


# ---------------------------------------------------------------------------
# Confidence Histograms
# ---------------------------------------------------------------------------
print("\n--- Confidence Histograms ---")

import torch.nn.functional as F

train_probs = F.softmax(train_logits, dim=1)
test_probs = F.softmax(test_logits, dim=1)

# Get confidence of the true class
train_conf = train_probs[torch.arange(len(train_labels)), train_labels].detach().cpu().numpy()
test_conf = test_probs[torch.arange(len(test_labels)), test_labels].detach().cpu().numpy()

fig_conf, axes_conf = plt.subplots(1, 2, figsize=(15, 6))

bins_conf = np.linspace(0, 1.0, 100)

axes_conf[0].hist(train_conf, bins=bins_conf, color="steelblue", alpha=0.7, edgecolor="none")
axes_conf[0].set_yscale('symlog', linthresh=1.0)
axes_conf[0].set_yticks([0, 10, 100, 1000, 10000])
axes_conf[0].set_yticklabels(['0', '10', '100', '1000', '10000'])
axes_conf[0].set_title("Train Confidence (True Class)")
axes_conf[0].set_xlabel("Confidence")
axes_conf[0].set_ylabel("Count (Log Scale)")
axes_conf[0].grid(True, linestyle=':', alpha=0.5)

axes_conf[1].hist(test_conf, bins=bins_conf, color="peru", alpha=0.7, edgecolor="none")
axes_conf[1].set_yscale('symlog', linthresh=1.0)
axes_conf[1].set_yticks([0, 10, 100, 1000, 10000])
axes_conf[1].set_yticklabels(['0', '10', '100', '1000', '10000'])
axes_conf[1].set_title("Test Confidence (True Class)")
axes_conf[1].set_xlabel("Confidence")
axes_conf[1].set_ylabel("Count (Log Scale)")
axes_conf[1].grid(True, linestyle=':', alpha=0.5)

fig_conf.suptitle(f"Model Confidence for True Class\nModel {RUN_ID[:8]}…  |  step {CHECKPOINT_STEP:,}", fontsize=14)
fig_conf.tight_layout()

out_path_conf = OUT_DIR / "rnc1_confidence_hist.png"
fig_conf.savefig(out_path_conf, dpi=150, bbox_inches="tight")
print(f"Confidence histogram saved to: {out_path_conf}")

# ---------------------------------------------------------------------------
# Margin-based MIA
# ---------------------------------------------------------------------------
print("\n--- Margin-based Membership Inference Attack ---")

def get_correct_side_margins(f, l, w, b, c_main, c_other):
    mask_main = (l == c_main)
    mask_other = (l == c_other)
    
    w_diff = w[c_main] - w[c_other]
    b_diff = b[c_main] - b[c_other]
    norm_w_diff = torch.norm(w_diff, p=2)
    
    def calc_m(f_sub):
        return (torch.matmul(f_sub, w_diff) + b_diff) / norm_w_diff
        
    m_main = calc_m(f[mask_main]).numpy()
    m_other = -calc_m(f[mask_other]).numpy() # invert so positive means deeper into correct territory
    
    return np.concatenate([m_main, m_other]) if len(m_main) > 0 or len(m_other) > 0 else np.array([])

pairs_margin_mia = [(c, 0) for c in range(1, 10)]

all_train_margin = []
all_test_margin = []

margin_mias = {}

for c, c0 in pairs_margin_mia:
    tr_m = get_correct_side_margins(train_f_normal, train_l_normal, w, b, c, c0)
    te_m = get_correct_side_margins(test_features.float(), test_labels, w, b, c, c0)
    
    if len(tr_m) > 1 and len(te_m) > 1:
        all_train_margin.append(tr_m)
        all_test_margin.append(te_m)
        margin_mias[c] = evaluate_mia(tr_m, te_m)
    else:
        margin_mias[c] = None

all_train_margin_flat = np.concatenate(all_train_margin) if all_train_margin else np.array([])
all_test_margin_flat = np.concatenate(all_test_margin) if all_test_margin else np.array([])

global_margin_mia = evaluate_mia(all_train_margin_flat, all_test_margin_flat)

if global_margin_mia:
    b_bounds = global_margin_mia['bounds']
    if b_bounds[0] is None:
        b_str = f"x < {b_bounds[1]:.3f}"
    elif b_bounds[1] is None:
        b_str = f"x > {b_bounds[0]:.3f}"
    else:
        b_str = f"[{b_bounds[0]:.3f}, {b_bounds[1]:.3f}]" if b_bounds[0] < b_bounds[1] else f"x < {b_bounds[1]:.3f} OR x > {b_bounds[0]:.3f}"
    print(f"Global Margin MIA -> Bounds: {b_str}  |  Acc: {global_margin_mia['acc']:.1%}  |  TPR: {global_margin_mia['tpr']:.1%}  |  FPR: {global_margin_mia['fpr']:.1%}")
    ff = global_margin_mia['fixed_fprs']
    print(f"Global Margin MIA (Fixed FPR) -> TPR@1%: {ff[0.01]['tpr']:.1%} | TPR@5%: {ff[0.05]['tpr']:.1%} | TPR@10%: {ff[0.10]['tpr']:.1%}")

# Plot Margin MIA
fig_margin_mia, axes_margin_mia = plt.subplots(2, 5, figsize=(18, 7))
axes_margin_mia = axes_margin_mia.flatten()

ax_g = axes_margin_mia[0]
if global_margin_mia and len(all_train_margin_flat) > 1 and len(all_test_margin_flat) > 1:
    x_min = min(all_train_margin_flat.min(), all_test_margin_flat.min()) - 0.5
    x_max = max(all_train_margin_flat.max(), all_test_margin_flat.max()) + 0.5
    x_grid = np.linspace(x_min, x_max, 500)
    
    kde_train = gaussian_kde(all_train_margin_flat, bw_method="scott")
    kde_test = gaussian_kde(all_test_margin_flat, bw_method="scott")
    ax_g.plot(x_grid, kde_train(x_grid), color="blue", label="train")
    ax_g.plot(x_grid, kde_test(x_grid), color="orange", linestyle="--", label="test")
    
    b_bounds = global_margin_mia['bounds']
    if b_bounds[0] is None:
        ax_g.axvline(b_bounds[1], color='k', linestyle=':')
        ax_g.axvspan(x_min, b_bounds[1], color='blue', alpha=0.1)
    elif b_bounds[1] is None:
        ax_g.axvline(b_bounds[0], color='k', linestyle=':')
        ax_g.axvspan(b_bounds[0], x_max, color='blue', alpha=0.1)
    elif b_bounds[0] <= b_bounds[1]:
        ax_g.axvline(b_bounds[0], color='k', linestyle=':')
        ax_g.axvline(b_bounds[1], color='k', linestyle=':')
        ax_g.axvspan(b_bounds[0], b_bounds[1], color='blue', alpha=0.1)
    else:
        ax_g.axvline(b_bounds[0], color='k', linestyle=':')
        ax_g.axvline(b_bounds[1], color='k', linestyle=':')
        ax_g.axvspan(x_min, b_bounds[1], color='blue', alpha=0.1)
        ax_g.axvspan(b_bounds[0], x_max, color='blue', alpha=0.1)
    
    ff = global_margin_mia['fixed_fprs']
    ax_g.set_title(f"Global\n$\\mu_T$={global_margin_mia['mu_train']:.2f} ($\\sigma_T$={global_margin_mia['std_train']:.2f}) | $\\mu_V$={global_margin_mia['mu_test']:.2f} ($\\sigma_V$={global_margin_mia['std_test']:.2f})\nAcc: {global_margin_mia['acc']:.1%} | TPR: {global_margin_mia['tpr']:.1%} | FPR: {global_margin_mia['fpr']:.1%}\nTPR @ 1%: {ff[0.01]['tpr']:.1%} | 5%: {ff[0.05]['tpr']:.1%} | 10%: {ff[0.10]['tpr']:.1%}", fontsize=8)
    ax_g.legend(fontsize=8, frameon=False)

for i, (c, c0) in enumerate(pairs_margin_mia):
    ax = axes_margin_mia[i + 1]
    mia_c = margin_mias.get(c)
    if not mia_c:
        continue
        
    tr_m = get_correct_side_margins(train_f_normal, train_l_normal, w, b, c, c0)
    te_m = get_correct_side_margins(test_features.float(), test_labels, w, b, c, c0)
    
    if len(tr_m) > 1 and len(te_m) > 1:
        x_min = min(tr_m.min(), te_m.min()) - 0.5
        x_max = max(tr_m.max(), te_m.max()) + 0.5
        x_grid = np.linspace(x_min, x_max, 500)
        
        kde_train = gaussian_kde(tr_m, bw_method="scott")
        kde_test = gaussian_kde(te_m, bw_method="scott")
        ax.plot(x_grid, kde_train(x_grid), color="blue")
        ax.plot(x_grid, kde_test(x_grid), color="orange", linestyle="--")
        
        b_bounds = mia_c['bounds']
        if b_bounds[0] is None:
            ax.axvline(b_bounds[1], color='k', linestyle=':')
            ax.axvspan(x_min, b_bounds[1], color='blue', alpha=0.1)
        elif b_bounds[1] is None:
            ax.axvline(b_bounds[0], color='k', linestyle=':')
            ax.axvspan(b_bounds[0], x_max, color='blue', alpha=0.1)
        elif b_bounds[0] <= b_bounds[1]:
            ax.axvline(b_bounds[0], color='k', linestyle=':')
            ax.axvline(b_bounds[1], color='k', linestyle=':')
            ax.axvspan(b_bounds[0], b_bounds[1], color='blue', alpha=0.1)
        else:
            ax.axvline(b_bounds[0], color='k', linestyle=':')
            ax.axvline(b_bounds[1], color='k', linestyle=':')
            ax.axvspan(x_min, b_bounds[1], color='blue', alpha=0.1)
            ax.axvspan(b_bounds[0], x_max, color='blue', alpha=0.1)
            
        ff = mia_c['fixed_fprs']
        ax.set_title(f"Class {c} vs 0\n$\\mu_T$={mia_c['mu_train']:.2f} ($\\sigma_T$={mia_c['std_train']:.2f}) | $\\mu_V$={mia_c['mu_test']:.2f} ($\\sigma_V$={mia_c['std_test']:.2f})\nAcc: {mia_c['acc']:.1%} | TPR: {mia_c['tpr']:.1%} | FPR: {mia_c['fpr']:.1%}\nTPR @ 1%: {ff[0.01]['tpr']:.1%} | 5%: {ff[0.05]['tpr']:.1%} | 10%: {ff[0.10]['tpr']:.1%}", fontsize=8)

fig_margin_mia.suptitle(f"Margin MIA Decision Boundaries (Class C vs Class 0)\nModel {RUN_ID[:8]}…  |  step {CHECKPOINT_STEP:,}")
fig_margin_mia.tight_layout()

out_path_margin_mia = OUT_DIR / "rnc1_margin_mia.png"
fig_margin_mia.savefig(out_path_margin_mia, dpi=150, bbox_inches="tight")
print(f"Margin MIA plot saved to: {out_path_margin_mia}")
# plt.show()

# ---------------------------------------------------------------------------
# Approach 3: Removing Orthogonal Noise (PCA on Class Means)
# ---------------------------------------------------------------------------
print("\n--- Removing Orthogonal Noise (PCA on Class Means) ---")

# 1. Fit PCA ONLY on the class means to extract the C-1 Neural Collapse subspace
pca_collapse = PCA(n_components=num_classes - 1, random_state=0)
pca_collapse.fit(class_means_unscaled.numpy())

def safe_pca_transform(pca_obj, tensor_data):
    if len(tensor_data) == 0:
        return torch.empty((0, pca_obj.n_components_), dtype=torch.float32)
    return torch.tensor(pca_obj.transform(tensor_data.numpy()), dtype=torch.float32)

# 2. Transform all features into this noise-free subspace
train_f_clean = safe_pca_transform(pca_collapse, train_f_normal)
test_f_clean  = safe_pca_transform(pca_collapse, test_features.float())
canary_f_clean = safe_pca_transform(pca_collapse, train_f_canary)
train_f_all_clean = safe_pca_transform(pca_collapse, train_features.float())

# Compute RNC1 on the cleaned features
rnc1_train_clean = compute_rnc1(train_f_all_clean, train_labels)
rnc1_test_clean  = compute_rnc1(test_f_clean,  test_labels)
print(f"Clean RNC1 (train set) : {rnc1_train_clean:.6f}")
print(f"Clean RNC1 (test set)  : {rnc1_test_clean:.6f}")

# 3. Re-evaluate Distances to Class Mean in the clean subspace
class_means_clean = safe_pca_transform(pca_collapse, class_means_unscaled)

train_dists_clean = distances_to_class_mean(train_f_clean, train_l_normal, class_means_clean)
test_dists_clean  = distances_to_class_mean(test_f_clean, test_labels, class_means_clean)
canary_dists_clean = distances_to_class_mean(canary_f_clean, train_l_canary, class_means_clean)
test_f_canary_clean = safe_pca_transform(pca_collapse, test_f_canary)
test_canary_dists_clean = distances_to_class_mean(test_f_canary_clean, test_l_canary, class_means_clean)

train_test_dists_clean = {}
for c in range(num_classes):
    tr_c = train_dists_clean.get(c, np.array([]))
    te_c = test_dists_clean.get(c, np.array([]))
    train_test_dists_clean[c] = np.concatenate([tr_c, te_c]) if len(tr_c) > 0 or len(te_c) > 0 else np.array([])

all_train_flat_clean = np.concatenate(list(train_dists_clean.values())) if train_dists_clean else np.array([])
all_test_flat_clean = np.concatenate(list(test_dists_clean.values())) if test_dists_clean else np.array([])
all_dist_values_clean = np.concatenate([all_train_flat_clean, all_test_flat_clean])
if len(all_dist_values_clean) > 0:
    x_min_c, x_max_c = 0.0, float(np.percentile(all_dist_values_clean, 99.5))
else:
    x_min_c, x_max_c = 0.0, 1.0
x_grid_c = np.linspace(x_min_c, x_max_c, 400)

fig_clean, axes_clean = plt.subplots(2, 5, figsize=(18, 7), sharey=False)
axes_clean = axes_clean.flatten()

for c in range(num_classes):
    ax = axes_clean[c]
    color = colors[c]
    
    for dists, label, ls, alpha, lw, c_color in [
        (train_test_dists_clean, "train+test", "-", 0.95, 2.5, color),
        (train_dists_clean, "train", "-.",  0.85, 1.8, color),
        (test_dists_clean,  "test",  "--", 0.65, 1.8, color),
        (canary_dists_clean, "train_canary", ":", 0.85, 1.8, color),
        (test_canary_dists_clean, "test_canary", ":", 0.85, 1.8, "brown"),
    ]:
        if c not in dists or len(dists[c]) < 2:
            continue
        try:
            kde = gaussian_kde(dists[c], bw_method="scott")
            ax.plot(x_grid_c, kde(x_grid_c), linestyle=ls, color=c_color, alpha=alpha, label=label, lw=lw)
            ax.fill_between(x_grid_c, kde(x_grid_c), alpha=alpha * 0.18, color=c_color)
        except (np.linalg.LinAlgError, ValueError):
            # If variance is effectively zero, KDE might fail. Plot a sharp spike.
            mean_val = np.mean(dists[c])
            ax.axvline(mean_val, color=c_color, linestyle=ls, alpha=alpha, label=label, lw=lw)

    ax.set_title(f"Class {c}", fontsize=10)
    ax.set_xlabel("‖h − μ_c‖₂ (Clean Subspace)")
    ax.set_ylabel("density" if c % 5 == 0 else "")
    ax.legend(fontsize=8, frameon=False)
    ax.set_xlim(x_min_c, x_max_c)
    ax.set_ylim(bottom=0)

fig_clean.suptitle(
    f"Distance to class mean (NOISE-FREE {num_classes-1}D SUBSPACE)\n"
    f"Model {RUN_ID[:8]}…  |  step {CHECKPOINT_STEP:,}  |  solid=train+test, dashdot=train, dashed=test",
    fontsize=12,
)
plt.tight_layout()

out_path_clean = OUT_DIR / "rnc1_dist_density_clean.png"
fig_clean.savefig(out_path_clean, dpi=150, bbox_inches="tight")
print(f"Clean density plot saved to: {out_path_clean}")
# plt.show()

# ---------------------------------------------------------------------------
# Advanced MIA: 9D Clean Subspace Distance & Orthogonal Noise
# ---------------------------------------------------------------------------
print("\n--- Advanced MIA Results ---")

# 1. Evaluate MIA using the 9D Subspace Distances (Tiny Sphere)
mia_9d_global = evaluate_mia(all_train_flat_clean, all_test_flat_clean)
if mia_9d_global:
    ff = mia_9d_global['fixed_fprs']
    print(f"9D Subspace Distance MIA -> Acc: {mia_9d_global['acc']:.1%} | TPR: {mia_9d_global['tpr']:.1%} | FPR: {mia_9d_global['fpr']:.1%}")
    print(f"                            TPR@1%FPR: {ff[0.01]['tpr']:.1%} | TPR@5%FPR: {ff[0.05]['tpr']:.1%} | TPR@10%FPR: {ff[0.10]['tpr']:.1%}")

# 2. Evaluate MIA using Orthogonal Noise
# Reconstruct features to calculate orthogonal distance
train_recon = pca_collapse.inverse_transform(train_f_all_clean.numpy())
test_recon  = pca_collapse.inverse_transform(test_f_clean.numpy())

# The orthogonal noise is the Euclidean distance between original and reconstructed features
train_ortho_noise = np.linalg.norm(train_features.float().numpy() - train_recon, axis=1)
test_ortho_noise  = np.linalg.norm(test_features.float().numpy() - test_recon, axis=1)

mia_ortho_global = evaluate_mia(train_ortho_noise, test_ortho_noise)
if mia_ortho_global:
    ff = mia_ortho_global['fixed_fprs']
    print(f"Orthogonal Noise MIA     -> Acc: {mia_ortho_global['acc']:.1%} | TPR: {mia_ortho_global['tpr']:.1%} | FPR: {mia_ortho_global['fpr']:.1%}")
    print(f"                            TPR@1%FPR: {ff[0.01]['tpr']:.1%} | TPR@5%FPR: {ff[0.05]['tpr']:.1%} | TPR@10%FPR: {ff[0.10]['tpr']:.1%}")

# Plot the orthogonal noise distributions
fig_ortho, ax_ortho = plt.subplots(figsize=(8, 5))

# Filter out extreme outliers for cleaner plotting
max_val = max(np.percentile(train_ortho_noise, 99), np.percentile(test_ortho_noise, 99))
x_grid_o = np.linspace(0, max_val * 1.1, 400)

kde_tr = gaussian_kde(train_ortho_noise, bw_method="scott")
kde_te = gaussian_kde(test_ortho_noise, bw_method="scott")

ax_ortho.plot(x_grid_o, kde_tr(x_grid_o), color="blue", label=f"Train (mean={np.mean(train_ortho_noise):.2f})", lw=2)
ax_ortho.fill_between(x_grid_o, kde_tr(x_grid_o), alpha=0.2, color="blue")

ax_ortho.plot(x_grid_o, kde_te(x_grid_o), color="orange", linestyle="--", label=f"Test (mean={np.mean(test_ortho_noise):.2f})", lw=2)
ax_ortho.fill_between(x_grid_o, kde_te(x_grid_o), alpha=0.2, color="orange")

ax_ortho.set_title(f"Orthogonal Noise Distribution (191 Dimensions)\nModel {RUN_ID[:8]}… | step {CHECKPOINT_STEP:,}", fontsize=11)
ax_ortho.set_xlabel("‖h - h_proj‖₂ (Orthogonal Distance)")
ax_ortho.set_ylabel("Density")
ax_ortho.legend(fontsize=10, frameon=False)
ax_ortho.set_xlim(0, max_val * 1.1)
ax_ortho.set_ylim(bottom=0)

plt.tight_layout()
out_path_ortho = OUT_DIR / "rnc1_ortho_noise.png"
fig_ortho.savefig(out_path_ortho, dpi=150, bbox_inches="tight")
print(f"Orthogonal Noise plot saved to: {out_path_ortho}")
# plt.show()

# ---------------------------------------------------------------------------
# Multi-Boundary Margin MIA (All C-1 boundaries)
# ---------------------------------------------------------------------------
print("\n--- Multi-Boundary Margin MIA (Per-Class) ---")

all_tr_multi = []
all_te_multi = []

for c in range(num_classes):
    mask_tr = train_l_normal == c
    mask_te = test_labels == c
    
    f_tr = train_f_normal[mask_tr]
    f_te = test_features.float()[mask_te]
    
    if len(f_tr) < 2 or len(f_te) < 2:
        continue
        
    # Calculate margins to all other classes k != c
    def get_all_margins(f_sub, class_c):
        margins = []
        for k in range(num_classes):
            if k == class_c:
                continue
            w_diff = w[class_c] - w[k]
            b_diff = b[class_c] - b[k]
            norm_w = torch.norm(w_diff, p=2)
            m = (torch.matmul(f_sub, w_diff) + b_diff) / norm_w
            margins.append(m.unsqueeze(1))
        return torch.cat(margins, dim=1).numpy() # Shape: (N, 9)
        
    tr_margins = get_all_margins(f_tr, c)
    te_margins = get_all_margins(f_te, c)
    
    # Exact class mean of the margins on Train data
    tr_margin_mean = np.mean(tr_margins, axis=0)
    
    # Distance to the exact Train mean margins
    # If a sample is exactly at the class mean for ALL boundaries, this distance is ~0.
    tr_margin_dist = np.linalg.norm(tr_margins - tr_margin_mean, axis=1)
    te_margin_dist = np.linalg.norm(te_margins - tr_margin_mean, axis=1)
    
    all_tr_multi.append(tr_margin_dist)
    all_te_multi.append(te_margin_dist)
    
    mia_result = evaluate_mia(tr_margin_dist, te_margin_dist)
    if mia_result:
        print(f"Class {c} Multi-Boundary MIA -> Acc: {mia_result['acc']:.1%} | TPR: {mia_result['tpr']:.1%} | FPR: {mia_result['fpr']:.1%}")

if all_tr_multi and all_te_multi:
    global_multi_mia = evaluate_mia(np.concatenate(all_tr_multi), np.concatenate(all_te_multi))
    if global_multi_mia:
        print(f"Global Multi-Boundary MIA  -> Acc: {global_multi_mia['acc']:.1%} | TPR: {global_multi_mia['tpr']:.1%} | FPR: {global_multi_mia['fpr']:.1%}")

# ---------------------------------------------------------------------------
# Leakage-Free Multi-Boundary MIA (Weight-based Proxy)
# ---------------------------------------------------------------------------
print("\n--- Leakage-Free Multi-Boundary MIA (Weight Proxy) ---")

def apply_coin_flip_bound(d, bounds):
    if bounds[0] is None:
        in_region = d < bounds[1]
    elif bounds[1] is None:
        in_region = d > bounds[0]
    elif bounds[0] <= bounds[1]:
        in_region = (d >= bounds[0]) & (d <= bounds[1])
    else:
        in_region = (d < bounds[1]) | (d > bounds[0])
    
    preds = np.zeros_like(d, dtype=bool)
    coin_flips = np.random.rand(np.sum(in_region)) < 0.5
    preds[in_region] = coin_flips
    return preds

# Estimate the global feature norm scale from a pool of all candidate data (Train + Test)
# to properly scale the weight proxy without knowing which is which.
pool_features = torch.cat([train_f_normal, test_features.float()])
pool_mean_norm = pool_features.norm(dim=1).mean().item()

all_tr_leak_free = []
all_te_leak_free = []

for c in range(num_classes):
    mask_tr = train_l_normal == c
    mask_te = test_labels == c
    
    f_tr = train_f_normal[mask_tr]
    f_te = test_features.float()[mask_te]
    
    if len(f_tr) < 2 or len(f_te) < 2:
        continue
        
    # The proxy for the class mean is the weight vector, normalized and scaled to typical feature magnitude
    w_proxy = w[c] / torch.norm(w[c]) * pool_mean_norm
    w_proxy = w_proxy.unsqueeze(0) # Shape: (1, 200)
    
    # Calculate the ideal margins using ONLY the weight proxy (leakage free!)
    proxy_margins = get_all_margins(w_proxy, c)
    proxy_margin_mean = proxy_margins[0] 
    
    tr_margins = get_all_margins(f_tr, c)
    te_margins = get_all_margins(f_te, c)
    
    # Distance to the proxy margins instead of the empirical train margins
    tr_margin_dist_lf = np.linalg.norm(tr_margins - proxy_margin_mean, axis=1)
    te_margin_dist_lf = np.linalg.norm(te_margins - proxy_margin_mean, axis=1)
    
    all_tr_leak_free.append(tr_margin_dist_lf)
    all_te_leak_free.append(te_margin_dist_lf)
    
    mia_result = evaluate_mia(tr_margin_dist_lf, te_margin_dist_lf)
    if mia_result:
        print(f"Class {c} Leakage-Free MIA -> Acc: {mia_result['acc']:.1%} | TPR: {mia_result['tpr']:.1%} | FPR: {mia_result['fpr']:.1%}")
        bounds = mia_result['bounds']
        tr_preds_rand = apply_coin_flip_bound(tr_margin_dist_lf, bounds)
        te_preds_rand = apply_coin_flip_bound(te_margin_dist_lf, bounds)
        acc_rand = (np.sum(tr_preds_rand) + np.sum(~te_preds_rand)) / (len(tr_margin_dist_lf) + len(te_margin_dist_lf))
        tpr_rand = np.mean(tr_preds_rand)
        fpr_rand = np.mean(te_preds_rand)
        print(f"Class {c} LF Random MIA -> Acc: {acc_rand:.1%} | TPR: {tpr_rand:.1%} | FPR: {fpr_rand:.1%}")

tr_preds_rand_global = None
te_preds_rand_global = None

if all_tr_leak_free and all_te_leak_free:
    all_tr_lf = np.concatenate(all_tr_leak_free)
    all_te_lf = np.concatenate(all_te_leak_free)
    global_lf_mia = evaluate_mia(all_tr_lf, all_te_lf)
    if global_lf_mia:
        print(f"Global Leakage-Free MIA  -> Acc: {global_lf_mia['acc']:.1%} | TPR: {global_lf_mia['tpr']:.1%} | FPR: {global_lf_mia['fpr']:.1%}")
        bounds = global_lf_mia['bounds']
        tr_preds_rand_global = apply_coin_flip_bound(all_tr_lf, bounds)
        te_preds_rand_global = apply_coin_flip_bound(all_te_lf, bounds)
        acc_rand = (np.sum(tr_preds_rand_global) + np.sum(~te_preds_rand_global)) / (len(all_tr_lf) + len(all_te_lf))
        tpr_rand = np.mean(tr_preds_rand_global)
        fpr_rand = np.mean(te_preds_rand_global)
        print(f"Global LF Random MIA  -> Acc: {acc_rand:.1%} | TPR: {tpr_rand:.1%} | FPR: {fpr_rand:.1%}")

# ---------------------------------------------------------------------------
# Exact Match Verification (Proving the coordinates are identical)
# ---------------------------------------------------------------------------
print("\n--- Exact Match Verification ---")
print("What percentage of samples sit on the EXACT coordinate as the Train Class Mean?")

all_train_dists_200d = np.concatenate(list(train_dists.values())) if train_dists else np.array([])
all_train_dists_9d = np.concatenate(list(train_dists_clean.values())) if train_dists_clean else np.array([])
all_test_dists_200d = np.concatenate(list(test_dists.values())) if test_dists else np.array([])
all_test_dists_9d = np.concatenate(list(test_dists_clean.values())) if test_dists_clean else np.array([])

for eps in [0.5, 0.1, 1e-2, 1e-3, 1e-4, 1e-5]:
    if len(all_test_dists_200d) > 0 and len(all_test_dists_9d) > 0 and len(all_train_dists_200d) > 0 and len(all_train_dists_9d) > 0:
        pct_tr_200d = np.mean(all_train_dists_200d < eps) * 100
        pct_tr_9d = np.mean(all_train_dists_9d < eps) * 100
        pct_te_200d = np.mean(all_test_dists_200d < eps) * 100
        pct_te_9d = np.mean(all_test_dists_9d < eps) * 100
        print(f"Distance < {eps:.0e} | TRAIN: {pct_tr_200d:6.2f}% in 200D, {pct_tr_9d:6.2f}% in 9D | TEST: {pct_te_200d:6.2f}% in 200D, {pct_te_9d:6.2f}% in 9D")

# ---------------------------------------------------------------------------
# ROC Curve (Visualizing TPR vs FPR)
# ---------------------------------------------------------------------------
from sklearn.metrics import roc_curve, auc

print("\n--- Plotting ROC Curves ---")

def plot_roc(ax, train_scores, test_scores, name, color, invert=False):
    if len(train_scores) == 0 or len(test_scores) == 0:
        return
    
    y_true = np.concatenate([np.ones(len(train_scores)), np.zeros(len(test_scores))])
    y_scores = np.concatenate([np.array(train_scores), np.array(test_scores)])
    
    # If the metric is a distance (lower = more likely Train), we invert it so higher = Train
    if invert:
        y_scores = -y_scores
        
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')

fig_roc, ax_roc = plt.subplots(figsize=(8, 6))

# The MIAs we have built:
plot_roc(ax_roc, all_train_margin_flat, all_test_margin_flat, "1D Margin", "purple", invert=False)
plot_roc(ax_roc, all_train_flat_clean, all_test_flat_clean, "9D Subspace Distance", "blue", invert=True)
plot_roc(ax_roc, train_ortho_noise, test_ortho_noise, "Orthogonal Noise", "orange", invert=True)

if all_tr_multi and all_te_multi:
    plot_roc(ax_roc, np.concatenate(all_tr_multi), np.concatenate(all_te_multi), "Multi-Boundary Margin", "green", invert=True)
if all_tr_leak_free and all_te_leak_free:
    plot_roc(ax_roc, np.concatenate(all_tr_leak_free), np.concatenate(all_te_leak_free), "Leakage-Free Multi-Boundary", "cyan", invert=True)
if tr_preds_rand_global is not None and te_preds_rand_global is not None:
    tpr_val = np.mean(tr_preds_rand_global)
    fpr_val = np.mean(te_preds_rand_global)
    ax_roc.scatter([fpr_val], [tpr_val], color='magenta', marker='*', s=150, zorder=5, label=f'LF Random Coin-Flip Point')

ax_roc.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Guess')

# Highlight the low FPR region
ax_roc.axvspan(0, 0.05, color='red', alpha=0.1, label="Low FPR Region (<5%)")

ax_roc.set_xlim([0.0, 1.0])
ax_roc.set_ylim([0.0, 1.05])
ax_roc.set_xlabel('False Positive Rate (FPR)')
ax_roc.set_ylabel('True Positive Rate (TPR)')
ax_roc.set_title(f"ROC Curves for Membership Inference Attacks\nModel {RUN_ID[:8]}… | step {CHECKPOINT_STEP:,}")
ax_roc.legend(loc="lower right")
ax_roc.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
out_path_roc = OUT_DIR / "rnc1_roc_curves.png"
fig_roc.savefig(out_path_roc, dpi=150, bbox_inches="tight")
print(f"ROC Curves plotted to: {out_path_roc}")
# plt.show()

# ---------------------------------------------------------------------------
# Meta-Classifier (Empirically finding the Optimal MIA)
# ---------------------------------------------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

print("\n--- Meta-Classifier (The Ultimate Optimal MIA) ---")
print("Training ML models to distinguish Train vs Test using ALL 200 dimensions...")

# Prepare dataset: Features = 200D vectors, Labels = 1 (Train), 0 (Test)
# We sample an equal number of train and test points
min_samples = min(len(train_f_normal), len(test_features))

if min_samples < 2:
    print("Insufficient normal train samples available for Meta-Classifier MIA. Ending analysis early.")
    import sys
    sys.exit(0)

X_train_mia = np.concatenate([
    train_f_normal[:min_samples].numpy(),
    test_features.float()[:min_samples].numpy()
])
y_train_mia = np.concatenate([
    np.ones(min_samples),
    np.zeros(min_samples)
])

# Train/Test split specifically for evaluating the Meta-Classifier
X_mia_tr, X_mia_te, y_mia_tr, y_mia_te = train_test_split(X_train_mia, y_train_mia, test_size=0.3, random_state=42)

if TRAIN_MODELS:
    # 1. Logistic Regression (Finds optimal linear combination of all 200 dimensions)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_mia_tr, y_mia_tr)
    lr_probs = lr.predict_proba(X_mia_te)[:, 1]

    # 2. Random Forest (Finds optimal non-linear boundaries across all dimensions/noise)
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, n_jobs=-1, random_state=42)
    rf.fit(X_mia_tr, y_mia_tr)
    rf_probs = rf.predict_proba(X_mia_te)[:, 1]

    # 3. Deep Neural Network (Universal Function Approximator for ultra-complex manifolds)
    from sklearn.neural_network import MLPClassifier
    # An ultra-sophisticated DNN: Wider, deeper, adaptive learning, no early stopping
    dnn = MLPClassifier(
        hidden_layer_sizes=(512, 512, 256, 128), 
        activation='relu',
        solver='adam',
        learning_rate='adaptive',
        max_iter=2000, 
        early_stopping=False, # Let it train to full convergence to find the most complex manifold possible
        random_state=42
    )
    dnn.fit(X_mia_tr, y_mia_tr)
    dnn_probs = dnn.predict_proba(X_mia_te)[:, 1]

    fig_opt, ax_opt = plt.subplots(figsize=(8, 6))

    fpr_lr, tpr_lr, _ = roc_curve(y_mia_te, lr_probs)
    auc_lr = auc(fpr_lr, tpr_lr)
    ax_opt.plot(fpr_lr, tpr_lr, color='red', lw=2, label=f'Optimal Linear MIA (LR) (AUC = {auc_lr:.2f})')

    fpr_rf, tpr_rf, _ = roc_curve(y_mia_te, rf_probs)
    auc_rf = auc(fpr_rf, tpr_rf)
    ax_opt.plot(fpr_rf, tpr_rf, color='darkred', lw=2, label=f'Optimal Non-Linear MIA (RF) (AUC = {auc_rf:.2f})')

    fpr_dnn, tpr_dnn, _ = roc_curve(y_mia_te, dnn_probs)
    auc_dnn = auc(fpr_dnn, tpr_dnn)
    ax_opt.plot(fpr_dnn, tpr_dnn, color='purple', lw=2, label=f'Optimal Deep MIA (DNN) (AUC = {auc_dnn:.2f})')

    ax_opt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Guess')
    ax_opt.axvspan(0, 0.05, color='red', alpha=0.1, label="Low FPR Region (<5%)")

    ax_opt.set_xlim([0.0, 1.0])
    ax_opt.set_ylim([0.0, 1.05])
    ax_opt.set_xlabel('False Positive Rate (FPR)')
    ax_opt.set_ylabel('True Positive Rate (TPR)')
    ax_opt.set_title(f"Optimal Meta-Classifier MIA (All 200 Dimensions)\nModel {RUN_ID[:8]}… | step {CHECKPOINT_STEP:,}")
    ax_opt.legend(loc="lower right")
    ax_opt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    out_path_opt = OUT_DIR / "rnc1_optimal_mia.png"
    fig_opt.savefig(out_path_opt, dpi=150, bbox_inches="tight")
    print(f"Optimal MIA ROC Curves plotted to: {out_path_opt}")
    # plt.show()
else:
    print("Skipping Meta-Classifier training (TRAIN_MODELS=False).")

# ---------------------------------------------------------------------------
# Nearest Neighbor Overlap Analysis
# ---------------------------------------------------------------------------
from sklearn.neighbors import NearestNeighbors

print("\n--- Nearest Neighbor Overlap Analysis ---")
print("Calculating microscopic distances between Train and Test points...")

nn_dist_train = []
nn_dist_test = []

# We calculate this per-class to only compare points of the same class
for c in range(num_classes):
    f_tr_c = train_f_normal[train_l_normal == c].numpy()
    f_te_c = test_features.float()[test_labels == c].numpy()
    
    if len(f_tr_c) < 2 or len(f_te_c) == 0:
        continue
        
    # Fit NearestNeighbors on the Train data of class c
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(f_tr_c)
    
    # 1. Train-to-Train distance (distance to nearest OTHER train point)
    # We use n_neighbors=2 because the 1st neighbor is always the point itself (distance=0)
    distances_tr, _ = nbrs.kneighbors(f_tr_c)
    nn_dist_train.extend(distances_tr[:, 1])
    
    # 2. Test-to-Train distance (distance to the nearest Train point)
    # n_neighbors=1 is fine here because the Test point is not in the Train set
    distances_te, _ = nbrs.kneighbors(f_te_c, n_neighbors=1)
    nn_dist_test.extend(distances_te[:, 0])

nn_dist_train = np.array(nn_dist_train)
nn_dist_test = np.array(nn_dist_test)

# Plot the histograms
fig_nn, ax_nn = plt.subplots(figsize=(8, 6))

ax_nn.hist(nn_dist_train, bins=50, alpha=0.5, density=True, label='Train-to-Train Nearest Neighbor', color='blue')
ax_nn.hist(nn_dist_test, bins=50, alpha=0.5, density=True, label='Test-to-Train Nearest Neighbor', color='orange')

ax_nn.set_xlabel('L2 Distance (in 200D Space)')
ax_nn.set_ylabel('Density')
ax_nn.set_title(f"Microscopic Overlap Analysis (Nearest Neighbor Distances)\nModel {RUN_ID[:8]}… | step {CHECKPOINT_STEP:,}")
ax_nn.legend()
ax_nn.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
out_path_nn = OUT_DIR / "rnc1_nn_overlap.png"
fig_nn.savefig(out_path_nn, dpi=150, bbox_inches="tight")
print(f"Nearest Neighbor plot saved to: {out_path_nn}")
# plt.show()

# Calculate the overlap metric
median_tr = np.median(nn_dist_train)
median_te = np.median(nn_dist_test)
print(f"Median Train-to-Train Distance: {median_tr:.4f}")
print(f"Median Test-to-Train Distance:  {median_te:.4f}")

# ---------------------------------------------------------------------------
# Hypersphere Expansion Analysis (TPR & FPR vs. Radius)
# ---------------------------------------------------------------------------
print("\n--- Hypersphere Expansion Analysis ---")
print("Plotting TPR and FPR vs. Hypersphere Radius around Train Class Means (9D Subspace)...")

fig_hyper, axes_hyper = plt.subplots(2, 5, figsize=(20, 8), sharey=True)
axes_hyper = axes_hyper.flatten()

for c in range(num_classes):
    mask_tr = train_l_normal == c
    mask_te = test_labels == c
    
    # We use the 9D clean subspace as it perfectly represents the classifier's view
    f_tr = train_f_clean[mask_tr].numpy()
    f_te = test_f_clean[mask_te].numpy()
    
    if len(f_tr) == 0 or len(f_te) == 0:
        continue
        
    mean_tr = np.mean(f_tr, axis=0)
    
    dist_tr = np.linalg.norm(f_tr - mean_tr, axis=1)
    dist_te = np.linalg.norm(f_te - mean_tr, axis=1)
    
    # Sort all unique distances to act as our expanding hypersphere radii
    all_radii = np.sort(np.unique(np.concatenate([dist_tr, dist_te])))
    
    # Calculate TPR and FPR at each expanding radius
    tprs = np.searchsorted(np.sort(dist_tr), all_radii, side='right') / len(dist_tr)
    fprs = np.searchsorted(np.sort(dist_te), all_radii, side='right') / len(dist_te)
    
    ax = axes_hyper[c]
    ax.plot(all_radii, tprs, color='blue', lw=2, label='TPR (Train Enclosed)')
    ax.plot(all_radii, fprs, color='orange', lw=2, label='FPR (Test Enclosed)')
    
    # Fill the area where the Test data perfectly overlaps the Train data
    ax.fill_between(all_radii, 0, fprs, color='orange', alpha=0.3, label='Inseparable Overlap')
    
    # Zoom the x-axis to the microscopic region where the Train data sits
    max_radius = np.percentile(dist_tr, 95)
    if max_radius > 0:
        ax.set_xlim([0, max_radius * 1.5])
        
    ax.set_title(f"Class {c}")
    ax.set_xlabel("Radius (L2 Dist to Mean)")
    if c == 0 or c == 5:
        ax.set_ylabel("Percentage Enclosed")
    ax.grid(True, linestyle=':', alpha=0.6)
    
    if c == 0:
        ax.legend(loc='lower right')

fig_hyper.suptitle(f"Hypersphere Expansion: TPR & FPR vs Radius (9D Subspace)\nModel {RUN_ID[:8]}… | step {CHECKPOINT_STEP:,}", fontsize=16)
plt.tight_layout()
out_path_hyper = OUT_DIR / "rnc1_hypersphere.png"
fig_hyper.savefig(out_path_hyper, dpi=150, bbox_inches="tight")
print(f"Hypersphere Expansion plot saved to: {out_path_hyper}")
# plt.show()
# ---------------------------------------------------------------------------
# Hypersphere Expansion Analysis (200D Raw Space)
# ---------------------------------------------------------------------------
print("\n--- Hypersphere Expansion Analysis (200D) ---")
print("Plotting TPR and FPR vs. Hypersphere Radius around Train Class Means (200D Space)...")

fig_hyper200, axes_hyper200 = plt.subplots(2, 5, figsize=(20, 8), sharey=True)
axes_hyper200 = axes_hyper200.flatten()

for c in range(num_classes):
    mask_tr = train_l_normal == c
    mask_te = test_labels == c
    
    f_tr = train_f_normal[mask_tr].numpy()
    f_te = test_features.float()[mask_te].numpy()
    
    if len(f_tr) == 0 or len(f_te) == 0:
        continue
        
    mean_tr = np.mean(f_tr, axis=0)
    
    dist_tr = np.linalg.norm(f_tr - mean_tr, axis=1)
    dist_te = np.linalg.norm(f_te - mean_tr, axis=1)
    
    all_radii = np.sort(np.unique(np.concatenate([dist_tr, dist_te])))
    tprs = np.searchsorted(np.sort(dist_tr), all_radii, side='right') / len(dist_tr)
    fprs = np.searchsorted(np.sort(dist_te), all_radii, side='right') / len(dist_te)
    
    ax = axes_hyper200[c]
    ax.plot(all_radii, tprs, color='blue', lw=2, label='TPR (Train Enclosed)')
    ax.plot(all_radii, fprs, color='orange', lw=2, label='FPR (Test Enclosed)')
    ax.fill_between(all_radii, 0, fprs, color='orange', alpha=0.3, label='Inseparable Overlap')
    
    max_radius = np.percentile(dist_tr, 95)
    if max_radius > 0:
        ax.set_xlim([0, max_radius * 1.5])
        
    ax.set_title(f"Class {c}")
    ax.set_xlabel("Radius (L2 Dist to Mean)")
    if c == 0 or c == 5:
        ax.set_ylabel("Percentage Enclosed")
    ax.grid(True, linestyle=':', alpha=0.6)
    
    if c == 0:
        ax.legend(loc='lower right')

fig_hyper200.suptitle(f"Hypersphere Expansion: TPR & FPR vs Radius (200D Space)\nModel {RUN_ID[:8]}… | step {CHECKPOINT_STEP:,}", fontsize=16)
plt.tight_layout()
out_path_hyper200 = OUT_DIR / "rnc1_hypersphere_200d.png"
fig_hyper200.savefig(out_path_hyper200, dpi=150, bbox_inches="tight")
print(f"Hypersphere Expansion (200D) plot saved to: {out_path_hyper200}")
# plt.show()

# ---------------------------------------------------------------------------
# Test Data Distribution: Dist to Mean vs. Dist to Nearest Train (200D)
# ---------------------------------------------------------------------------
print("\n--- Test Data Distribution Analysis ---")
print("Plotting Distance to Train Mean vs. Distance to Nearest Train Point...")

fig_dist, axes_dist = plt.subplots(2, 5, figsize=(20, 8), sharex=True, sharey=True)
axes_dist = axes_dist.flatten()

from sklearn.neighbors import NearestNeighbors

for c in range(num_classes):
    mask_tr = train_l_normal == c
    mask_te = test_labels == c
    
    f_tr = train_f_normal[mask_tr].numpy()
    f_te = test_features.float()[mask_te].numpy()
    
    if len(f_tr) < 2 or len(f_te) == 0:
        continue
        
    mean_tr = np.mean(f_tr, axis=0)
    
    # X-axis: Distance to Mean
    dist_mean_te = np.linalg.norm(f_te - mean_tr, axis=1)
    
    # Y-axis: Distance to Nearest Train point
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(f_tr)
    distances_te, _ = nbrs.kneighbors(f_te)
    dist_nn_te = distances_te[:, 0]
    
    ax = axes_dist[c]
    
    # Plot Train points for comparison! 
    # X: Dist to mean. Y: Dist to nearest OTHER train point.
    dist_mean_tr = np.linalg.norm(f_tr - mean_tr, axis=1)
    distances_tr, _ = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(f_tr).kneighbors(f_tr)
    dist_nn_tr = distances_tr[:, 1]
    
    # We plot Train first so Test is overlaid on top
    ax.scatter(dist_mean_tr, dist_nn_tr, color='blue', alpha=0.3, s=8, label='Train Points')
    ax.scatter(dist_mean_te, dist_nn_te, color='orange', alpha=0.3, s=8, label='Test Points')
    
    # Reference line y = x (Distance to NN == Distance to Mean)
    max_val = max(np.max(dist_mean_te), np.max(dist_nn_te))
    ax.plot([0, max_val], [0, max_val], color='gray', linestyle='--', label='y = x')
    
    ax.set_title(f"Class {c}")
    if c > 4:
        ax.set_xlabel("Dist to Class Mean")
    if c == 0 or c == 5:
        ax.set_ylabel("Dist to Nearest Train Pt")
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Zoom in to the cluster where Train points exist
    max_train_dist = np.percentile(dist_mean_tr, 99)
    if max_train_dist > 0:
        ax.set_xlim([0, max_train_dist * 2])
        ax.set_ylim([0, np.max(dist_nn_tr) * 2])
    
    if c == 0:
        ax.legend(loc='upper left')

fig_dist.suptitle(f"Microscopic Distribution: Dist to Mean vs Dist to Nearest Train Point (200D Space)\nModel {RUN_ID[:8]}… | step {CHECKPOINT_STEP:,}", fontsize=16)
plt.tight_layout()
out_path_dist = OUT_DIR / "rnc1_dist_vs_nn.png"
fig_dist.savefig(out_path_dist, dpi=150, bbox_inches="tight")
print(f"Distance distribution scatter plot saved to: {out_path_dist}")
# plt.show()

# ---------------------------------------------------------------------------
# Theoretical Discriminator Limit (Ultimate Overfitter)
# ---------------------------------------------------------------------------
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

print("\n--- Theoretical Discriminator Limit (Overfitted 1-NN) ---")
print("Training an ultimate overfitted discriminator on ALL balanced data and evaluating on the SAME data...")

# We use the balanced dataset (X_train_mia, y_train_mia) created earlier (200D space)
# 1. Evaluate in 200D space
nn_200d = KNeighborsClassifier(n_neighbors=1, algorithm='auto')
nn_200d.fit(X_train_mia, y_train_mia)
preds_200d = nn_200d.predict(X_train_mia)
acc_200d = accuracy_score(y_train_mia, preds_200d)

# 2. Evaluate in 9D Subspace
X_train_9d = np.concatenate([
    train_f_clean[:min_samples].numpy(),
    test_f_clean[:min_samples].numpy()
])
nn_9d = KNeighborsClassifier(n_neighbors=1, algorithm='auto')
nn_9d.fit(X_train_9d, y_train_mia)
preds_9d = nn_9d.predict(X_train_9d)
acc_9d = accuracy_score(y_train_mia, preds_9d)

print(f"Absolute Overfitted Accuracy (200D Space): {acc_200d:.2%}")
print(f"Absolute Overfitted Accuracy (9D Subspace):  {acc_9d:.2%}")

# ---------------------------------------------------------------------------
# KNN Accuracy, TPR, FPR vs. n_neighbors (The Bayes Error Collapse)
# ---------------------------------------------------------------------------
print("\n--- KNN Collapse with TPR and FPR ---")
print("Plotting how metrics collapse to the Bayes Error limit as we increase n_neighbors...")

n_range = np.arange(1, 51, 2) # 1, 3, 5, ..., 49

fig_knn, axes_knn = plt.subplots(1, 2, figsize=(18, 7), sharey=True)

# Helper function to compute metrics for different n and e
def compute_knn_metrics(features, labels, n_range, e_limit=None):
    nbrs = NearestNeighbors(n_neighbors=max(n_range), algorithm='auto').fit(features)
    distances, indices = nbrs.kneighbors(features)
    
    accs, tprs, fprs = [], [], []
    for n in n_range:
        n_dist = distances[:, :n]
        n_idx = indices[:, :n]
        
        preds = []
        for i in range(len(features)):
            valid_idx = n_idx[i]
            if e_limit is not None:
                mask = n_dist[i] <= e_limit
                valid_idx = valid_idx[mask]
                
            if len(valid_idx) == 0:
                pred = labels[i]
            else:
                neighbor_labels = labels[valid_idx]
                pred = np.bincount(neighbor_labels.astype(int)).argmax()
            preds.append(pred)
            
        preds = np.array(preds)
        
        acc = accuracy_score(labels, preds)
        tpr = np.sum((preds == 1) & (labels == 1)) / np.sum(labels == 1)
        fpr = np.sum((preds == 1) & (labels == 0)) / np.sum(labels == 0)
        
        accs.append(acc)
        tprs.append(tpr)
        fprs.append(fpr)
        
    return accs, tprs, fprs

# Use a small 'e' based on the Train-to-Train median calculated earlier
e_small = median_tr * 2.0 

# 1. Evaluate 200D
accs_200d_std, tprs_200d_std, fprs_200d_std = compute_knn_metrics(X_train_mia, y_train_mia, n_range, e_limit=None)
accs_200d_lim, tprs_200d_lim, fprs_200d_lim = compute_knn_metrics(X_train_mia, y_train_mia, n_range, e_limit=e_small)

ax = axes_knn[0]
ax.plot(n_range, accs_200d_std, marker='o', label='Acc (Std)', color='blue')
ax.plot(n_range, tprs_200d_std, marker='^', linestyle='--', label='TPR (Std)', color='green')
ax.plot(n_range, fprs_200d_std, marker='v', linestyle='--', label='FPR (Std)', color='red')

ax.plot(n_range, accs_200d_lim, marker='s', label=f'Acc (Lim e={e_small:.4f})', color='orange')
ax.plot(n_range, tprs_200d_lim, marker='^', linestyle=':', label='TPR (Lim)', color='lime')
ax.plot(n_range, fprs_200d_lim, marker='v', linestyle=':', label='FPR (Lim)', color='darkred')

ax.set_title("200D Space")
ax.set_xlabel("n_neighbors")
ax.set_ylabel("Metric Value (Evaluating on Training Data)")
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(fontsize=9, loc='center left', bbox_to_anchor=(1, 0.5))

# 2. Evaluate 9D
nbrs_9d_tr = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(train_f_clean.numpy())
dist_9d_tr, _ = nbrs_9d_tr.kneighbors(train_f_clean.numpy())
e_small_9d = np.median(dist_9d_tr[:, 1]) * 2.0

accs_9d_std, tprs_9d_std, fprs_9d_std = compute_knn_metrics(X_train_9d, y_train_mia, n_range, e_limit=None)
accs_9d_lim, tprs_9d_lim, fprs_9d_lim = compute_knn_metrics(X_train_9d, y_train_mia, n_range, e_limit=e_small_9d)

ax = axes_knn[1]
ax.plot(n_range, accs_9d_std, marker='o', label='Acc (Std)', color='blue')
ax.plot(n_range, tprs_9d_std, marker='^', linestyle='--', label='TPR (Std)', color='green')
ax.plot(n_range, fprs_9d_std, marker='v', linestyle='--', label='FPR (Std)', color='red')

ax.plot(n_range, accs_9d_lim, marker='s', label=f'Acc (Lim e={e_small_9d:.4f})', color='orange')
ax.plot(n_range, tprs_9d_lim, marker='^', linestyle=':', label='TPR (Lim)', color='lime')
ax.plot(n_range, fprs_9d_lim, marker='v', linestyle=':', label='FPR (Lim)', color='darkred')

ax.set_title("9D Subspace")
ax.set_xlabel("n_neighbors")
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(fontsize=9, loc='center left', bbox_to_anchor=(1, 0.5))

fig_knn.suptitle(f"The Bayes Error Collapse: Accuracy, TPR, FPR vs. n_neighbors\nModel {RUN_ID[:8]}… | step {CHECKPOINT_STEP:,}", fontsize=16)
plt.tight_layout()
out_path_knn = OUT_DIR / "rnc1_knn_collapse.png"
fig_knn.savefig(out_path_knn, dpi=150, bbox_inches="tight")
print(f"KNN Collapse plot saved to: {out_path_knn}")
# plt.show()

# ---------------------------------------------------------------------------
# Estimating True Train Mean via PDF Maximum (Mode Extraction)
# ---------------------------------------------------------------------------
print("\n--- Extracting True Train Mean via PDF Maximum ---")


diffs_test_only = []
diffs_combined = []

for c in range(num_classes):
    mask_tr = train_l_normal == c
    mask_te = test_labels == c
    
    y_tr = train_f_normal[mask_tr]
    y_te = test_features.float()[mask_te]
    
    if len(y_tr) < 2 or len(y_te) < 2:
        continue
        
    y_comb = torch.cat([y_tr, y_te], dim=0)
    
    # 9D features (for stable KDE)
    x_tr = train_f_clean[mask_tr].numpy()
    x_te = test_f_clean[mask_te].numpy()
    x_comb = np.concatenate([x_tr, x_te], axis=0)
    
    true_mean = class_means_unscaled[c]
    
    # 1. Test data only
    try:
        kde_te = gaussian_kde(x_te.T, bw_method="scott")
        densities_te = kde_te(x_te.T)
        idx_max_te = np.argmax(densities_te)
        est_mean_te = y_te[idx_max_te]
        dist_te = torch.norm(est_mean_te - true_mean).item()
        diffs_test_only.append(dist_te)
    except Exception as e:
        dist_te = float('nan')
        
    # 2. Combined data
    try:
        kde_comb = gaussian_kde(x_comb.T, bw_method="scott")
        densities_comb = kde_comb(x_comb.T)
        idx_max_comb = np.argmax(densities_comb)
        est_mean_comb = y_comb[idx_max_comb]
        dist_comb = torch.norm(est_mean_comb - true_mean).item()
        diffs_combined.append(dist_comb)
    except Exception as e:
        dist_comb = float('nan')
        
    print(f"Class {c}: Distance to true mean | Test only: {dist_te:.4f} | Combined: {dist_comb:.4f}")

    print(f"Average Distance (Test only) : {np.nanmean(diffs_test_only):.4f}")
if diffs_combined:
    print(f"Average Distance (Combined)  : {np.nanmean(diffs_combined):.4f}")

# ---------------------------------------------------------------------------
# Outlier MIA (Train Canaries vs Test Canaries)
# ---------------------------------------------------------------------------
print("\n--- Outlier MIA (Train Canaries vs Test Canaries) ---")

from privacy_and_grokking.datasets.canaries.uniform_noise import UniformNoiseCanary

if len(train_f_canary) > 0:
    print("\n--- Canary Classification Correctness ---")
    
    def print_classification_matrix(logits, labels, name):
        preds = logits.argmax(dim=-1)
        correct = (preds == labels)
        total_correct = correct.sum().item()
        total_count = len(labels)
        print(f"{name} Canaries: {total_correct}/{total_count} ({total_correct/total_count:.1%}) correctly classified.")
        print(f"{'Class':<6} | {'Total':<6} | {'Correct':<8} | {'False':<8}")
        print("-" * 35)
        for c in range(10):
            c_mask = labels == c
            if c_mask.sum() == 0: continue
            c_correct = correct[c_mask].sum().item()
            c_total = c_mask.sum().item()
            print(f"{c:<6} | {c_total:<6} | {c_correct:<8} | {c_total - c_correct:<8}")
        print()
        
    print_classification_matrix(train_y_canary, train_l_canary, "Train")
    print_classification_matrix(test_y_canary, test_l_canary, "Test")

    # We already have train canary distances from earlier: canary_dists_clean
    all_train_canary_flat = np.concatenate(list(canary_dists_clean.values())) if canary_dists_clean else np.array([])
    all_test_canary_flat = np.concatenate(list(test_canary_dists_clean.values())) if test_canary_dists_clean else np.array([])
    
    if len(all_train_canary_flat) > 0 and len(all_test_canary_flat) > 0:
        outlier_mia_global = evaluate_mia(all_train_canary_flat, all_test_canary_flat)
        if outlier_mia_global:
            ff = outlier_mia_global['fixed_fprs']
            print(f"Global Outlier MIA (9D Distance) -> Acc: {outlier_mia_global['acc']:.1%} | TPR: {outlier_mia_global['tpr']:.1%} | FPR: {outlier_mia_global['fpr']:.1%}")
            print(f"                                    TPR@1%FPR: {ff[0.01]['tpr']:.1%} | TPR@5%FPR: {ff[0.05]['tpr']:.1%} | TPR@10%FPR: {ff[0.10]['tpr']:.1%}")
    
        fig_outlier_roc, ax_outlier_roc = plt.subplots(figsize=(8, 6))
        
        # invert=True because lower distance = more likely Train
        plot_roc(ax_outlier_roc, all_train_canary_flat, all_test_canary_flat, "Outlier MIA (Train Canary vs Test Canary)", "red", invert=True)
        
        ax_outlier_roc.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Guess')
        ax_outlier_roc.axvspan(0, 0.05, color='red', alpha=0.1, label="Low FPR Region (<5%)")
        
        ax_outlier_roc.set_xlim([0.0, 1.0])
        ax_outlier_roc.set_ylim([0.0, 1.05])
        ax_outlier_roc.set_xlabel('False Positive Rate (FPR)')
        ax_outlier_roc.set_ylabel('True Positive Rate (TPR)')
        ax_outlier_roc.set_title(f"Outlier MIA: ROC Curve (Distinguishing Memorized vs Unseen Outliers)\nModel {RUN_ID[:8]}… | step {CHECKPOINT_STEP:,}")
        ax_outlier_roc.legend(loc="lower right")
        ax_outlier_roc.grid(True, linestyle=':', alpha=0.6)
        
        plt.tight_layout()
        out_path_outlier_roc = OUT_DIR / "rnc1_outlier_mia_roc.png"
        fig_outlier_roc.savefig(out_path_outlier_roc, dpi=150, bbox_inches="tight")
        print(f"Outlier MIA ROC plotted to: {out_path_outlier_roc}")
        # plt.show()
else:
    print("Skipping Outlier MIA: No canaries were injected during training for this model.")

# ---------------------------------------------------------------------------
# Neural Collapse Metrics (NC1 - NC4)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 1. Directional Analysis: Overshooting vs Undershooting (Cosine Sim to Mean)
# ---------------------------------------------------------------------------
print("\n--- Directional Analysis 1: Overshooting vs Undershooting ---")
fig_dir1, axes_dir1 = plt.subplots(2, 5, figsize=(20, 8))
axes_dir1 = axes_dir1.flatten()

for c in range(num_classes):
    ax = axes_dir1[c]
    mask_tr = train_l_normal == c
    mask_te = test_labels == c
    
    f_tr = train_f_normal[mask_tr]
    f_te = test_features.float()[mask_te]
    
    if len(f_tr) < 2 or len(f_te) == 0:
        continue
        
    mu_c = class_means_unscaled[c]
    norm_mu_c = torch.norm(mu_c)
    
    if norm_mu_c.item() == 0:
        continue
        
    r_tr = f_tr - mu_c
    r_te = f_te - mu_c
    
    dist_tr = torch.norm(r_tr, dim=1)
    dist_te = torch.norm(r_te, dim=1)
    
    # Cosine similarity to mu_c: <r, mu_c> / (||r|| * ||mu_c||)
    cos_tr = torch.matmul(r_tr, mu_c) / (dist_tr * norm_mu_c + 1e-9)
    cos_te = torch.matmul(r_te, mu_c) / (dist_te * norm_mu_c + 1e-9)
    
    ax.scatter(dist_tr.numpy(), cos_tr.numpy(), color='blue', alpha=0.3, s=10, label='Train')
    ax.scatter(dist_te.numpy(), cos_te.numpy(), color='orange', alpha=0.3, s=10, label='Test')
    
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    
    ax.set_title(f"Class {c}")
    if c >= 5:
        ax.set_xlabel("Distance to Mean ||r_i||")
    if c % 5 == 0:
        ax.set_ylabel("Cos Sim to Mean")
    if c == 0:
        ax.legend()
        
fig_dir1.suptitle(f"Overshooting vs Undershooting (Cosine Sim to Class Mean)\nModel {RUN_ID[:8]}… | step {CHECKPOINT_STEP:,}", fontsize=14)
fig_dir1.tight_layout()
out_path_dir1 = OUT_DIR / "rnc1_dir1_overshooting.png"
fig_dir1.savefig(out_path_dir1, dpi=150, bbox_inches="tight")
print(f"Overshooting analysis plot saved to: {out_path_dir1}")

# ---------------------------------------------------------------------------
# 3. Directional Analysis: Intra-Class Variance (PCA of Residuals)
# ---------------------------------------------------------------------------
print("\n--- Directional Analysis 3: Intra-Class Variance (PCA of Residuals) ---")
fig_dir3, axes_dir3 = plt.subplots(2, 5, figsize=(20, 8))
axes_dir3 = axes_dir3.flatten()

for c in range(num_classes):
    ax = axes_dir3[c]
    mask_tr = train_l_normal == c
    mask_te = test_labels == c
    
    f_tr = train_f_normal[mask_tr]
    f_te = test_features.float()[mask_te]
    
    if len(f_tr) < 3 or len(f_te) == 0:
        continue
        
    mu_c = class_means_unscaled[c]
    r_tr = (f_tr - mu_c).numpy()
    r_te = (f_te - mu_c).numpy()
    
    pca_res = PCA(n_components=2)
    pca_res.fit(r_tr)
    
    r_tr_pca = pca_res.transform(r_tr)
    r_te_pca = pca_res.transform(r_te)
    
    ax.scatter(r_tr_pca[:, 0], r_tr_pca[:, 1], color='blue', alpha=0.3, s=10, label='Train')
    ax.scatter(r_te_pca[:, 0], r_te_pca[:, 1], color='orange', alpha=0.3, s=10, label='Test')
    
    ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
    ax.axvline(0, color='black', linestyle='--', linewidth=0.5)
    
    ax.set_title(f"Class {c} (EV: {pca_res.explained_variance_ratio_[0]:.1%}, {pca_res.explained_variance_ratio_[1]:.1%})")
    if c >= 5:
        ax.set_xlabel("PC 1 of Train Residuals")
    if c % 5 == 0:
        ax.set_ylabel("PC 2 of Train Residuals")
    if c == 0:
        ax.legend()

fig_dir3.suptitle(f"Intra-Class Variance (PCA of Train Residuals)\nModel {RUN_ID[:8]}… | step {CHECKPOINT_STEP:,}", fontsize=14)
fig_dir3.tight_layout()
out_path_dir3 = OUT_DIR / "rnc1_dir3_intra_class_pca.png"
fig_dir3.savefig(out_path_dir3, dpi=150, bbox_inches="tight")
print(f"Intra-Class PCA plot saved to: {out_path_dir3}")

# ---------------------------------------------------------------------------
# 5. Directional Analysis: Spherical Distribution (Pairwise Cosine Similarities)
# ---------------------------------------------------------------------------
print("\n--- Directional Analysis 5: Spherical Distribution (Pairwise Cos Sim) ---")
fig_dir5, axes_dir5 = plt.subplots(2, 5, figsize=(20, 8))
axes_dir5 = axes_dir5.flatten()

for c in range(num_classes):
    ax = axes_dir5[c]
    mask_tr = train_l_normal == c
    mask_te = test_labels == c
    
    f_tr = train_f_normal[mask_tr]
    f_te = test_features.float()[mask_te]
    
    if len(f_tr) < 2 or len(f_te) < 2:
        continue
        
    mu_c = class_means_unscaled[c]
    r_tr = f_tr - mu_c
    r_te = f_te - mu_c
    
    # Normalize residuals
    v_tr = F.normalize(r_tr, p=2, dim=1)
    v_te = F.normalize(r_te, p=2, dim=1)
    
    # Subsample to max 500 to avoid huge matrices
    if len(v_tr) > 500: v_tr = v_tr[torch.randperm(len(v_tr))[:500]]
    if len(v_te) > 500: v_te = v_te[torch.randperm(len(v_te))[:500]]
    
    cos_tr_tr = torch.matmul(v_tr, v_tr.T)
    cos_te_te = torch.matmul(v_te, v_te.T)
    cos_tr_te = torch.matmul(v_tr, v_te.T)
    
    # Extract upper triangle without diagonal for tr_tr and te_te
    idx_tr = torch.triu_indices(len(v_tr), len(v_tr), offset=1)
    vals_tr_tr = cos_tr_tr[idx_tr[0], idx_tr[1]].numpy()
    
    idx_te = torch.triu_indices(len(v_te), len(v_te), offset=1)
    vals_te_te = cos_te_te[idx_te[0], idx_te[1]].numpy()
    
    vals_tr_te = cos_tr_te.flatten().numpy()
    
    x_grid = np.linspace(-1, 1, 200)
    
    try:
        kde_tr_tr = gaussian_kde(vals_tr_tr, bw_method="scott")
        ax.plot(x_grid, kde_tr_tr(x_grid), color='blue', label='Train-Train')
    except: pass
    
    try:
        kde_te_te = gaussian_kde(vals_te_te, bw_method="scott")
        ax.plot(x_grid, kde_te_te(x_grid), color='orange', linestyle='--', label='Test-Test')
    except: pass
    
    try:
        kde_tr_te = gaussian_kde(vals_tr_te, bw_method="scott")
        ax.plot(x_grid, kde_tr_te(x_grid), color='green', linestyle=':', label='Train-Test')
    except: pass
    
    ax.set_title(f"Class {c}")
    ax.set_xlim([-1, 1])
    if c >= 5:
        ax.set_xlabel("Pairwise Cosine Similarity")
    if c % 5 == 0:
        ax.set_ylabel("Density")
    if c == 0:
        ax.legend(fontsize=8)
        
fig_dir5.suptitle(f"Spherical Isotropy: Pairwise Cosine Similarities of Residuals\nModel {RUN_ID[:8]}… | step {CHECKPOINT_STEP:,}", fontsize=14)
fig_dir5.tight_layout()
out_path_dir5 = OUT_DIR / "rnc1_dir5_spherical_isotropy.png"
fig_dir5.savefig(out_path_dir5, dpi=150, bbox_inches="tight")
print(f"Spherical isotropy plot saved to: {out_path_dir5}")


print("\n--- Neural Collapse Metrics (Train vs Test) ---")
from privacy_and_grokking.metrics.neural_collapse import compute_all_nc_metrics

last_layer = model.classifier()
W = last_layer.weight
b = last_layer.bias if hasattr(last_layer, 'bias') else None

train_nc = compute_all_nc_metrics(train_features, train_labels, W, b)
test_nc = compute_all_nc_metrics(test_features, test_labels, W, b)

print(f"NC1 (Variance Collapse)                                | Train: {train_nc.nc1:.4f} | Test: {test_nc.nc1:.4f}")
print(f"NC2 Equinorm (Variance of Norms, -> 0)                 | Train: {train_nc.nc2_equinorm:.4f} | Test: {test_nc.nc2_equinorm:.4f}")
print(f"NC2 Equiangular (Deviation from ETF angles, -> 0)      | Train: {train_nc.nc2_equiangular:.4f} | Test: {test_nc.nc2_equiangular:.4f}")
print(f"NC3 (Self-Duality Frobenius Diff, -> 0)                | Train: {train_nc.nc3_papyan:.4f} | Test: {test_nc.nc3_papyan:.4f}")
print(f"NC4 (Agreement of Linear Classifier and NCC, -> 1)     | Train: {train_nc.nc4:.4f} | Test: {test_nc.nc4:.4f}")

# ---------------------------------------------------------------------------
# MLFlow Logging & Output Capture
# ---------------------------------------------------------------------------
sys.stdout = print_capture.original_stdout

with open(OUT_DIR / "analysis_output.txt", "w") as f:
    f.write(print_capture.getvalue())

try:
    with mlflow.start_run(run_id=RUN_ID) as run:
        mlflow.log_artifacts(str(OUT_DIR), artifact_path=f"rnc1_analysis_{CHECKPOINT_STEP}")
    print(f"Logged artifacts to MLflow run {RUN_ID} under 'rnc1_analysis_{CHECKPOINT_STEP}'")
except (mlflow.exceptions.RestException, mlflow.exceptions.MlflowException) as e:
    print(f"Failed to attach to run {RUN_ID} ({e}). Creating a new run for this analysis.")
    with mlflow.start_run() as run:
        mlflow.set_tag("original_run_id", RUN_ID)
        mlflow.log_artifacts(str(OUT_DIR), artifact_path=f"rnc1_analysis_{CHECKPOINT_STEP}")
    print(f"Logged artifacts to new MLflow run {run.info.run_id} under 'rnc1_analysis_{CHECKPOINT_STEP}'")
    
temp_dir.cleanup()

