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

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from privacy_and_grokking.utils.logger import Logger
from privacy_and_grokking.config import TrainConfig
from privacy_and_grokking.metrics.neural_collapse import compute_rnc1

Logger().setup()  # required before any project code calls Logger.get()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# RUN_ID = "c9a3105bba4a4fe499b1e6ce139d4c85"
RUN_ID = "9c95201d8ada4c0db02da91615c5c984"
CHECKPOINT_STEP = 100_000
ARTIFACT_BASE = Path(__file__).parent.parent / "mlartifacts" / "7" #"6"
CHECKPOINT_PATH = (
    ARTIFACT_BASE / RUN_ID / "artifacts" / "checkpoints" / str(CHECKPOINT_STEP) / "model.pth"
)
CONFIG_PATH = ARTIFACT_BASE / RUN_ID / "artifacts" / "training_config.json"

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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (features [N, 200], labels [N]) for the given dataset."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    features_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []

    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            # MLP forward up to the last layer input
            y = torch.flatten(imgs, 1)
            y = F.relu(model.fc1(y))
            y = F.relu(model.fc2(y))
            y = F.relu(model.fc3(y))
            features_list.append(y.cpu())
            labels_list.append(lbls.cpu() if isinstance(lbls, torch.Tensor) else torch.tensor(lbls))

    return torch.cat(features_list), torch.cat(labels_list)


print("Extracting train features...")
train_features, train_labels = extract_features(train_dataset, model, device)
print("Extracting test features...")
test_features, test_labels = extract_features(test_dataset, model, device)

print(f"  train_features: {train_features.shape}")
print(f"  test_features : {test_features.shape}")

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

print(f"  train_features_normal: {train_f_normal.shape}")
print(f"  train_features_canary: {train_f_canary.shape}")

# ---------------------------------------------------------------------------
# RNC1
# ---------------------------------------------------------------------------
rnc1_train = compute_rnc1(train_features, train_labels)
rnc1_test  = compute_rnc1(test_features,  test_labels)
print(f"\nRNC1 (train set) : {rnc1_train:.6f}")
print(f"RNC1 (test set)  : {rnc1_test:.6f}")

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

out_path = Path(__file__).parent / f"rnc1_analysis_{RUN_ID[:8]}_step{CHECKPOINT_STEP}.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved to: {out_path}")
plt.show()

# ---------------------------------------------------------------------------
# Density of distance-to-class-mean for each class  (train vs test)
# ---------------------------------------------------------------------------
from scipy.stats import gaussian_kde

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

train_test_dists = {}
for c in range(num_classes):
    train_c = train_dists.get(c, np.array([]))
    test_c = test_dists.get(c, np.array([]))
    train_test_dists[c] = np.concatenate([train_c, test_c]) if len(train_c) > 0 or len(test_c) > 0 else np.array([])

# Combined grid for KDE evaluation
all_dist_values = np.concatenate(
    list(train_dists.values()) + list(test_dists.values()) + list(canary_dists.values())
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

for dists_flat, label, ls, color, alpha in [
    (all_train_test_flat, "train+test", "-", "purple", 0.85),
    (all_train_flat, "train", "-",  "blue", 0.85),
    (all_test_flat,  "test",  "--", "orange", 0.65),
    (all_canary_flat, "canary", ":", "red", 0.85),
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

out_path_comb = Path(__file__).parent / f"rnc1_dist_density_combined_{RUN_ID[:8]}_step{CHECKPOINT_STEP}.png"
fig_comb.savefig(out_path_comb, dpi=150, bbox_inches="tight")
print(f"Combined density plot saved to: {out_path_comb}")
plt.show()

fig2, axes2 = plt.subplots(2, 5, figsize=(18, 7), sharey=False)
axes2 = axes2.flatten()

for c in range(num_classes):
    ax = axes2[c]
    color = colors[c]

    for dists, label, ls, alpha in [
        (train_test_dists, "train+test", "-", 0.95),
        (train_dists, "train", "-",  0.85),
        (test_dists,  "test",  "--", 0.65),
        (canary_dists, "canary", ":", 0.85),
    ]:
        if c not in dists or len(dists[c]) < 2:
            continue
        kde = gaussian_kde(dists[c], bw_method="scott")
        ax.plot(x_grid, kde(x_grid), linestyle=ls, color=color, alpha=alpha, label=label, lw=1.8)
        ax.fill_between(x_grid, kde(x_grid), alpha=alpha * 0.18, color=color)

    ax.set_title(f"Class {c}", fontsize=10)
    ax.set_xlabel("‖h − μ_c‖₂")
    ax.set_ylabel("density" if c % 5 == 0 else "")
    ax.legend(fontsize=8, frameon=False)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(bottom=0)

fig2.suptitle(
    f"Distance to class mean (train-set means)\n"
    f"Model {RUN_ID[:8]}…  |  step {CHECKPOINT_STEP:,}  |  solid=train(+test), dashed=test, dotted=canary",
    fontsize=12,
)
plt.tight_layout()

out_path2 = Path(__file__).parent / f"rnc1_dist_density_{RUN_ID[:8]}_step{CHECKPOINT_STEP}.png"
plt.savefig(out_path2, dpi=150, bbox_inches="tight")
print(f"Density plot saved to: {out_path2}")
plt.show()
