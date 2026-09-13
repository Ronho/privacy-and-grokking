#!/usr/bin/env python3
"""compute_hard_samples.py.

Extracts and caches the per-sample loss concentration dynamics across training checkpoints:
1. Top-5 and Top-1 loss share percentage over time (73 checkpoints).
2. Identification of persistent 'hard' samples across all checkpoints.
3. Raw images, true labels, and persistence frequencies for top outlier samples.

Saves results to cache/{run_id}_hard_samples.npz.
"""

from collections import Counter
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from privacy_and_grokking.config import TrainConfig
from privacy_and_grokking.utils.logger import Logger
from scripts.compute_loss_landscape import GpuDataset, find_training_config

DEFAULT_CACHE_DIR = PROJECT_DIR / "cache"
DEFAULT_TRACKING_URI = "http://localhost:5051"

RUNS = [
    {"run_id": "39ca6f59220d438d9d14de7fdb0be8c9", "label": "MLP (MSE)", "loss_type": "mse"},
    {"run_id": "82c4f9ab35bb4b7180fdacc194ce7e07", "label": "MLP (CE)", "loss_type": "ce"},
    {"run_id": "b75317f9447346249f0b811acec427ce", "label": "ViT (CE)", "loss_type": "ce"},
]


def compute_hard_samples_for_run(
    run_id: str,
    loss_type: str,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    force: bool = False,
) -> Path:
    target_path = cache_dir / f"{run_id}_hard_samples.npz"
    if target_path.is_file() and not force:
        print(f"Hard samples data already exists at {target_path}. Use force=True to recompute.")
        return target_path

    Logger().setup()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg_dict = find_training_config(run_id, cache_dir, tracking_uri)
    config = TrainConfig.model_validate(cfg_dict)
    data_container = config.data()

    train_raw = data_container.train
    if data_container.train_canary is not None:
        train_ds = torch.utils.data.ConcatDataset([train_raw, data_container.train_canary])
    else:
        train_ds = train_raw
    test_ds = data_container.test

    gpu_train = GpuDataset(train_ds, device)
    gpu_test = GpuDataset(test_ds, device)

    if data_container.normalization is not None:
        norm = data_container.normalization
        norm_mean = torch.tensor(norm.mean, device=device, dtype=torch.float32).view(1, -1, 1, 1)
        norm_std = torch.tensor(norm.std, device=device, dtype=torch.float32).view(1, -1, 1, 1)
        images_norm = (gpu_train.images - norm_mean) / norm_std
        test_images_norm = (gpu_test.images - norm_mean) / norm_std
    else:
        images_norm = gpu_train.images
        test_images_norm = gpu_test.images

    model = config.model(
        input_dim=data_container.input_shape,
        num_classes=data_container.num_classes,
    ).to(device)

    ckpt_dir = cache_dir / "checkpoints" / run_id
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoints directory not found at {ckpt_dir}")

    steps = sorted([int(p.name) for p in ckpt_dir.iterdir() if p.is_dir() and p.name.isdigit()])
    print(f"Processing {len(steps)} checkpoints for {run_id} ({loss_type}) on {device}...")

    steps_evaluated = []
    top1_share_history = []
    top5_share_history = []
    top5_indices_history = []
    train_acc_history = []
    test_acc_history = []
    mean_logit_mag_history = []

    for step in tqdm(steps, desc="Analyzing checkpoints"):
        ckpt_file = ckpt_dir / str(step) / "model.pth"
        if not ckpt_file.is_file():
            continue

        ckpt = torch.load(ckpt_file, map_location=device, weights_only=True)
        model.load_state_dict(ckpt)
        model.eval()

        with torch.no_grad():
            out = model(images_norm)
            train_acc = float((out.argmax(dim=-1) == gpu_train.labels).float().mean().item())
            mean_logit_mag = float(out.abs().mean().item())

            test_out = model(test_images_norm)
            test_acc = float((test_out.argmax(dim=-1) == gpu_test.labels).float().mean().item())

            if loss_type == "mse":
                y_oh = torch.zeros_like(out).scatter_(1, gpu_train.labels.unsqueeze(1), 1.0)
                losses = torch.sum((out - y_oh) ** 2, dim=1)
            else:
                losses = F.cross_entropy(out, gpu_train.labels, reduction="none")

            sorted_losses, sorted_indices = torch.sort(losses, descending=True)
            total_loss = float(losses.sum().item())
            if total_loss > 0:
                top1_share = float(sorted_losses[0].item() / total_loss * 100.0)
                top5_share = float(sorted_losses[:5].sum().item() / total_loss * 100.0)
            else:
                top1_share = 0.0
                top5_share = 0.0

            steps_evaluated.append(step)
            top1_share_history.append(top1_share)
            top5_share_history.append(top5_share)
            top5_indices_history.append(sorted_indices[:5].cpu().numpy())
            train_acc_history.append(train_acc)
            test_acc_history.append(test_acc)
            mean_logit_mag_history.append(mean_logit_mag)

    # Find the top 5 most persistent hard sample indices across late checkpoints (step > 10,000)
    late_top5_indices = [
        idx
        for s, arr in zip(steps_evaluated, top5_indices_history)
        if s > 10000
        for idx in arr
    ]
    n_late = sum(1 for s in steps_evaluated if s > 10000)
    counts = Counter(late_top5_indices)
    top_persistent = counts.most_common(5)

    hard_sample_indices = np.array([item[0] for item in top_persistent], dtype=np.int64)
    hard_sample_counts = np.array([item[1] for item in top_persistent], dtype=np.int64)
    hard_sample_freqs = hard_sample_counts / max(n_late, 1) * 100.0

    # Extract raw images (unnormalized in [0, 1]) and true labels for these 5 samples
    hard_sample_images = []
    hard_sample_labels = []
    for idx in hard_sample_indices:
        raw_img, raw_lbl = train_ds[int(idx)]
        if isinstance(raw_img, torch.Tensor):
            img_np = raw_img.squeeze().cpu().numpy()
        else:
            img_np = np.array(raw_img, dtype=np.float32)
        hard_sample_images.append(img_np)
        hard_sample_labels.append(int(raw_lbl))

    hard_sample_images = np.stack(hard_sample_images, axis=0)
    hard_sample_labels = np.array(hard_sample_labels, dtype=np.int64)

    np.savez(
        target_path,
        steps=np.array(steps_evaluated, dtype=np.int64),
        top1_share=np.array(top1_share_history, dtype=np.float32),
        top5_share=np.array(top5_share_history, dtype=np.float32),
        train_acc=np.array(train_acc_history, dtype=np.float32),
        test_acc=np.array(test_acc_history, dtype=np.float32),
        mean_logit_magnitude=np.array(mean_logit_mag_history, dtype=np.float32),
        hard_sample_indices=hard_sample_indices,
        hard_sample_frequencies=hard_sample_freqs,
        hard_sample_labels=hard_sample_labels,
        hard_sample_images=hard_sample_images,
    )
    print(f"Saved hard sample dynamics and accuracy to {target_path}")
    return target_path


def main():
    for r in RUNS:
        compute_hard_samples_for_run(r["run_id"], r["loss_type"], force=True)


if __name__ == "__main__":
    main()
