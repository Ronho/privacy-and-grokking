import argparse
import json
import os
import tempfile

import matplotlib.pyplot as plt
import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from privacy_and_grokking.config import TrainConfig
from privacy_and_grokking.utils.logger import Logger
from privacy_and_grokking.utils.mlflow import TRACKING_URI

# ---------------------------------------------------------------------------
# DeepInversion-style helpers
# ---------------------------------------------------------------------------


def register_hooks(model):
    """Register forward hooks on all layers that produce useful activations.
    Returns (hook_handles, activations_dict) where activations_dict is populated
    during each forward pass, keyed by layer name."""
    activations = {}
    handles = []

    def _make_hook(name):
        def hook_fn(_module, _input, output):
            activations[name] = output

        return hook_fn

    for name, module in model.named_modules():
        # Skip the model itself and the final classifier head
        if module is model:
            continue
        # Hook into layers that produce useful spatial/feature activations
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.BatchNorm1d)):
            h = module.register_forward_hook(_make_hook(name))
            handles.append(h)

    return handles, activations


def cosine_similarity_loss(a, b):
    """1 - cosine similarity between flattened tensors. Range [0, 2]."""
    a_flat = a.flatten()
    b_flat = b.flatten()
    return 1.0 - F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0))


def multi_layer_feature_loss(current_acts, target_acts, mode="cosine"):
    """Compute feature matching loss across all hooked layers.

    Args:
        current_acts: dict of current activations from hooked layers
        target_acts: dict of target activations from hooked layers
        mode: 'cosine' (scale-invariant) or 'mse'
    """
    loss = torch.tensor(0.0, device=next(iter(current_acts.values())).device)
    n_layers = 0
    for name in current_acts:
        if name not in target_acts:
            continue
        curr = current_acts[name]
        targ = target_acts[name]
        if mode == "cosine":
            loss = loss + cosine_similarity_loss(curr, targ)
        else:
            loss = loss + F.mse_loss(curr, targ)
        n_layers += 1
    if n_layers > 0:
        loss = loss / n_layers  # average across layers
    return loss


def total_variation_loss(img):
    """Multi-scale total variation loss."""
    _, c, h, w = img.shape
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    tv = (tv_h + tv_w) / (c * h * w)
    # Scale 2: stride-2 differences for larger-scale smoothness
    if h > 2 and w > 2:
        tv_h2 = torch.pow(img[:, :, 2:, :] - img[:, :, :-2, :], 2).sum()
        tv_w2 = torch.pow(img[:, :, :, 2:] - img[:, :, :, :-2], 2).sum()
        tv = tv + 0.5 * (tv_h2 + tv_w2) / (c * h * w)
    return tv


def run_patch_inversion(
    run_id,
    step,
    img_index=0,
    patch_size=5,
    lr=0.1,
    num_iters=1000,
    dataset_split="train",
    use_context_matching=False,
    use_dip=False,
    jitter=2,
    noise_std=0.0,
    scale_jitter=False,
    losses_dict=None,
    start_y=-1,
    start_x=-1,
):
    if losses_dict is None:
        losses_dict = {"feature": 1.0, "tv": 1e-3}
    Logger().setup()

    mlflow.set_tracking_uri(TRACKING_URI)

    print(f"Downloading artifacts for run {run_id} at step {step}...")
    try:
        model_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=f"checkpoints/{step}/model.pth"
        )
        config_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="training_config.json"
        )
    except Exception as e:
        print(f"Failed to download artifacts: {e}")
        return

    with open(config_path) as f:
        config_dict = json.load(f)

    config = TrainConfig.model_validate(config_dict)
    data_container = config.data()
    num_classes = data_container.num_classes
    input_shape = data_container.input_shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    norm_mean = None
    norm_std = None
    if data_container.normalization is not None:
        norm_mean = torch.tensor(data_container.normalization.mean, device=device).view(-1, 1, 1)
        norm_std = torch.tensor(data_container.normalization.std, device=device).view(-1, 1, 1)

    model = config.model(
        input_dim=input_shape,
        num_classes=num_classes,
    )
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    first_weight = None
    for param in model.parameters():
        if param.requires_grad and param.dim() > 1:
            first_weight = param
            break

    # Get image
    dataset = data_container.train if dataset_split == "train" else data_container.test
    true_img, true_label = dataset[img_index]

    # ensure it has a batch dimension
    true_img = true_img.unsqueeze(0).to(device)
    target_tensor = torch.tensor([true_label], device=device)

    # Create mask (1 for unmasked, 0 for masked)
    mask = torch.ones_like(true_img, device=device)
    _, c, h, w = true_img.shape

    if start_y < 0:
        start_y = (h - patch_size) // 2
    if start_x < 0:
        start_x = (w - patch_size) // 2

    mask[:, :, start_y : start_y + patch_size, start_x : start_x + patch_size] = 0

    # Masked image: context pixels preserved, patch region zeroed
    context_img = true_img * mask

    # -----------------------------------------------------------------------
    # Register hooks for multi-layer feature matching
    # -----------------------------------------------------------------------
    hooks, activations = register_hooks(model)
    print(
        f"Registered hooks on {len(hooks)} layers: {list(activations.keys()) if activations else '(will populate on first forward pass)'}"
    )

    # Compute target activations from the TRUE image
    with torch.no_grad():
        true_input = (true_img - norm_mean) / norm_std if norm_mean is not None else true_img
        _ = model(true_input)
        target_activations = {k: v.clone() for k, v in activations.items()}

    print(
        f"Target activations captured from {len(target_activations)} layers: {list(target_activations.keys())}"
    )

    # -----------------------------------------------------------------------
    # Compute natural feature statistics (Idee A)
    # -----------------------------------------------------------------------
    natural_feature_stats = {}
    if losses_dict.get("feature_stats", 0) > 0:
        print("Computing natural feature statistics from training data...")
        # Get a batch of data to compute robust statistics
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        imgs, _ = next(iter(train_loader))
        imgs = imgs.to(device)
        if norm_mean is not None:
            imgs = (imgs - norm_mean) / norm_std

        with torch.no_grad():
            model(imgs)
            for name, act in activations.items():
                # Compute mean and variance per channel for spatial layers
                if act.dim() == 4:
                    ch_mean = act.mean(dim=[0, 2, 3])
                    ch_var = act.var(dim=[0, 2, 3])
                    natural_feature_stats[name] = (ch_mean, ch_var)
        print(f"Computed natural statistics for {len(natural_feature_stats)} spatial layers.")

    # -----------------------------------------------------------------------
    # Initialize the missing patch
    # -----------------------------------------------------------------------
    noise_patch = torch.randn((1, c, patch_size, patch_size), device=device) * 0.5
    initial_img = context_img.clone()
    initial_img[:, :, start_y : start_y + patch_size, start_x : start_x + patch_size] = noise_patch

    if use_dip:

        class TinyDIP(nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(channels, 32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.Conv2d(32, 32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.Conv2d(32, channels, kernel_size=3, padding=1),
                )

            def forward(self, x):
                return self.net(x)

        dip_model = TinyDIP(c).to(device)
        optimizer = optim.Adam(dip_model.parameters(), lr=lr)
        dip_z = noise_patch.clone()
        patch = dip_model(dip_z)
    else:
        patch = noise_patch.clone().requires_grad_(True)
        optimizer = optim.Adam([patch], lr=lr)

    # -----------------------------------------------------------------------
    # Also compute class-mean features for repr_mse (if requested)
    # -----------------------------------------------------------------------
    class_mean_features = None
    if losses_dict.get("repr_mse", 0.0) > 0.0:
        print("Computing class mean for true label...")
        train_loader = torch.utils.data.DataLoader(
            data_container.train, batch_size=256, shuffle=False
        )
        features_list = []
        with torch.no_grad():
            for imgs, lbls in train_loader:
                mask_lbl = lbls == true_label
                if not mask_lbl.any():
                    continue
                imgs = imgs[mask_lbl].to(device)
                imgs_input = (imgs - norm_mean) / norm_std if norm_mean is not None else imgs
                try:
                    output = model(imgs_input, verbose=True)
                    feats = output[1] if isinstance(output, tuple) else output
                except TypeError:
                    output = model(imgs_input)
                    feats = output[1] if isinstance(output, tuple) else output
                features_list.append(feats)

        if len(features_list) > 0:
            class_mean_features = torch.cat(features_list).mean(dim=0, keepdim=True)

    # Helper to compute confidence and entropy
    def get_confidence_and_entropy(img_tensor):
        with torch.no_grad():
            if norm_mean is not None:
                img_tensor = (img_tensor - norm_mean) / norm_std
            output = model(img_tensor)
            logits = output[0] if isinstance(output, tuple) else output
            probs = F.softmax(logits, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean().item()
            return probs[0, true_label].item(), entropy

    def get_dist_to_mean(img_tensor):
        if class_mean_features is None:
            return 0.0
        with torch.no_grad():
            img_input = (img_tensor - norm_mean) / norm_std if norm_mean is not None else img_tensor
            try:
                output = model(img_input, verbose=True)
                feats = output[1] if isinstance(output, tuple) else output
            except TypeError:
                output = model(img_input)
                feats = output[1] if isinstance(output, tuple) else output
            return torch.norm(feats - class_mean_features, p=2).item()

    conf_before, ent_before = get_confidence_and_entropy(true_img)
    conf_with_masked, ent_with_masked = get_confidence_and_entropy(initial_img)

    criterion = config.loss(num_classes=num_classes)

    print(f"Starting Patch Inversion for True Label: {true_label}")
    print(f"Active losses: {', '.join(f'{k}={v}' for k, v in losses_dict.items())}")
    print(f"Jitter: {jitter}, Context matching: {use_context_matching}, DIP: {use_dip}")

    with mlflow.start_run(run_id=run_id):
        intermediate_patches = []
        intermediate_dists = []
        for i in range(num_iters):
            optimizer.zero_grad()

            if use_dip:
                raw_patch = dip_model(dip_z)
                # Ensure DIP output is bounded to the valid image range using Sigmoid.
                # This prevents values from exploding when maximizing logits.
                true_min = true_img.min().item()
                true_max = true_img.max().item()
                patch = torch.sigmoid(raw_patch) * (true_max - true_min) + true_min

            # Reconstruct the full image
            full_img = context_img.clone()
            full_img[:, :, start_y : start_y + patch_size, start_x : start_x + patch_size] = patch

            aug_img = full_img

            # Spatial augmentations: Scale Jitter
            if scale_jitter:
                scale = torch.empty(1).uniform_(0.95, 1.05).item()
                new_h = int(h * scale)
                new_w = int(w * scale)
                scaled_img = F.interpolate(
                    aug_img, size=(new_h, new_w), mode="bilinear", align_corners=False
                )

                if new_h < h:
                    pad_h = (h - new_h) // 2
                    pad_w = (w - new_w) // 2
                    aug_img = F.pad(
                        scaled_img, (pad_w, w - new_w - pad_w, pad_h, h - new_h - pad_h)
                    )
                elif new_h > h:
                    crop_h = (new_h - h) // 2
                    crop_w = (new_w - w) // 2
                    aug_img = scaled_img[:, :, crop_h : crop_h + h, crop_w : crop_w + w]
                else:
                    aug_img = scaled_img

            # Spatial augmentations: Translation Jitter
            if jitter > 0:
                shift_x = torch.randint(-jitter, jitter + 1, (1,)).item()
                shift_y = torch.randint(-jitter, jitter + 1, (1,)).item()
                aug_img = torch.roll(aug_img, shifts=(shift_y, shift_x), dims=(-2, -1))

            # Noise augmentation
            if noise_std > 0.0:
                aug_img = aug_img + torch.randn_like(aug_img) * noise_std

            jittered_img = aug_img

            # Forward pass
            if norm_mean is not None:
                model_input = (jittered_img - norm_mean) / norm_std
            else:
                model_input = jittered_img

            try:
                output = model(model_input, verbose=True)
                logits, feats = output if isinstance(output, tuple) else (output, output)
            except TypeError:
                output = model(model_input)
                logits = output[0] if isinstance(output, tuple) else output
                feats = output[1] if isinstance(output, tuple) else output

            loss = torch.tensor(0.0, device=device)
            metrics_vals = {}
            probs = F.softmax(logits, dim=1)

            target_tensor = torch.tensor([true_label], device=device)

            # --- Multi-layer feature matching (DeepInversion-style) ---
            if losses_dict.get("feature", 0) > 0:
                feat_loss = multi_layer_feature_loss(activations, target_activations, mode="cosine")
                loss = loss + losses_dict["feature"] * feat_loss
                metrics_vals["feature"] = feat_loss.item()

            # --- Legacy: penultimate-layer only MSE matching ---
            if losses_dict.get("repr_mse", 0) > 0 and class_mean_features is not None:
                repr_mse_val = F.mse_loss(feats, class_mean_features, reduction="sum")
                loss = loss + losses_dict["repr_mse"] * repr_mse_val
                metrics_vals["repr_mse"] = repr_mse_val.item()

            # --- Classification losses ---
            if losses_dict.get("ce", 0) > 0:
                ce_val = criterion(logits, target_tensor)
                loss = loss + losses_dict["ce"] * ce_val
                metrics_vals["ce"] = ce_val.item()

            if losses_dict.get("neg_ent", 0) > 0:
                entropy_loss = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
                loss = loss + losses_dict["neg_ent"] * entropy_loss
                metrics_vals["neg_ent"] = entropy_loss.item()

            if losses_dict.get("conf", 0) > 0:
                conf_val = probs[:, true_label].mean()
                loss = loss - losses_dict["conf"] * conf_val
                metrics_vals["conf"] = conf_val.item()

            if losses_dict.get("logit", 0) > 0:
                logit_val = logits[0, true_label]
                # We want to MAXIMIZE the logit, so we subtract it from the loss
                loss = loss - losses_dict["logit"] * logit_val
                metrics_vals["logit"] = logit_val.item()

            if losses_dict.get("mse", 0) > 0:
                target_one_hot = F.one_hot(target_tensor, num_classes=num_classes).float()
                output_mse_val = F.mse_loss(probs, target_one_hot)
                loss = loss + losses_dict["mse"] * output_mse_val
                metrics_vals["mse"] = output_mse_val.item()

            # --- Input gradient regularization ---
            # Exploits overfitting: ∇ₓ z_y ≈ 0 at the memorized sample (logit maximum).
            # Uses raw logit instead of CE because CE saturates (softmax → 1.0).
            if losses_dict.get("grad_reg", 0) > 0:
                target_logit = logits[0, true_label]
                input_grad = torch.autograd.grad(
                    target_logit, patch, create_graph=True, retain_graph=True
                )[0]
                grad_reg_val = input_grad.norm(p=2)
                loss = loss + losses_dict["grad_reg"] * grad_reg_val
                metrics_vals["grad_reg"] = grad_reg_val.item()

            # --- Early layer weight gradient regularization ---
            # Targets the memorization minimum directly in the early layers
            if losses_dict.get("weight_grad", 0) > 0 and first_weight is not None:
                target_logit = logits[0, true_label]
                weight_grad = torch.autograd.grad(
                    target_logit, first_weight, create_graph=True, retain_graph=True
                )[0]
                weight_grad_val = weight_grad.norm(p=2)
                loss = loss + losses_dict["weight_grad"] * weight_grad_val
                metrics_vals["weight_grad"] = weight_grad_val.item()

            # --- Natural Feature Statistics Matching (Idee A) ---
            if losses_dict.get("feature_stats", 0) > 0 and natural_feature_stats:
                stats_loss = torch.tensor(0.0, device=device)
                for name, (true_mean, true_var) in natural_feature_stats.items():
                    if name in activations:
                        act = activations[name]
                        if act.dim() == 4:
                            gen_mean = act.mean(dim=[0, 2, 3])
                            gen_var = act.var(dim=[0, 2, 3])
                            stats_loss += F.mse_loss(gen_mean, true_mean) + F.mse_loss(
                                gen_var, true_var
                            )
                loss = loss + losses_dict["feature_stats"] * stats_loss
                metrics_vals["feat_stats"] = stats_loss.item()

            # --- Regularization ---
            if losses_dict.get("tv", 0) > 0:
                tv_loss = total_variation_loss(patch)
                loss = loss + losses_dict["tv"] * tv_loss
                metrics_vals["tv"] = tv_loss.item()

            if losses_dict.get("l2", 0) > 0:
                l2_loss = torch.norm(patch, p=2)
                loss = loss + losses_dict["l2"] * l2_loss
                metrics_vals["l2"] = l2_loss.item()

            if use_context_matching:
                y_min = max(0, start_y - 2)
                y_max = min(h, start_y + patch_size + 2)
                x_min = max(0, start_x - 2)
                x_max = min(w, start_x + patch_size + 2)
                context_box = true_img[:, :, y_min:y_max, x_min:x_max]
                surround_mean = context_box.mean(dim=(0, 2, 3))
                surround_var = context_box.var(dim=(0, 2, 3))
                patch_mean = patch.mean(dim=(0, 2, 3))
                patch_var = patch.var(dim=(0, 2, 3))
                context_loss = F.mse_loss(patch_mean, surround_mean) + F.mse_loss(
                    patch_var, surround_var
                )
                loss += 0.1 * context_loss

            dist_to_mean = (
                torch.norm(feats - class_mean_features, p=2).item()
                if class_mean_features is not None
                else 0.0
            )

            loss.backward()
            optimizer.step()

            # Pixel clamping: project patch back to valid image range
            if not use_dip:
                with torch.no_grad():
                    true_min = true_img.min().item()
                    true_max = true_img.max().item()
                    patch.clamp_(true_min, true_max)

            if i % 100 == 0:
                intermediate_patches.append((i, patch.clone().detach()))
                intermediate_dists.append(dist_to_mean)

                parts = [f"Iter {i:04d}", f"Total: {loss.item():.4f}"]
                if "feature" in metrics_vals:
                    parts.append(f"Feat: {metrics_vals['feature']:.4f}")
                if "repr_mse" in metrics_vals:
                    parts.append(f"ReprMSE: {metrics_vals['repr_mse']:.4f}")
                if "ce" in metrics_vals:
                    parts.append(f"CE: {metrics_vals['ce']:.4f}")
                if "conf" in metrics_vals:
                    parts.append(f"Conf: {metrics_vals['conf']:.4f}")
                if "logit" in metrics_vals:
                    parts.append(f"Logit: {metrics_vals['logit']:.4f}")
                if "neg_ent" in metrics_vals:
                    parts.append(f"NegEnt: {metrics_vals['neg_ent']:.4f}")
                if "grad_reg" in metrics_vals:
                    parts.append(f"GradReg: {metrics_vals['grad_reg']:.4f}")
                if "weight_grad" in metrics_vals:
                    parts.append(f"WGrad: {metrics_vals['weight_grad']:.4f}")
                if "feat_stats" in metrics_vals:
                    parts.append(f"FStats: {metrics_vals['feat_stats']:.4f}")
                if "tv" in metrics_vals:
                    parts.append(f"TV: {metrics_vals['tv']:.4f}")
                if "l2" in metrics_vals:
                    parts.append(f"L2: {metrics_vals['l2']:.4f}")

                print(" | ".join(parts))

        # Clean up hooks
        for h in hooks:
            h.remove()

        # Plotting
        with torch.no_grad():
            final_img = context_img.clone()
            final_img[:, :, start_y : start_y + patch_size, start_x : start_x + patch_size] = patch

            # Get true image min/max for scaling
            true_min = true_img.min().item()
            true_max = true_img.max().item()

            def to_numpy(t):
                t = t.clone().detach().cpu()
                t = torch.clamp(t, true_min, true_max)
                t -= true_min
                t /= true_max - true_min + 1e-8
                if c == 1:
                    return t[0, 0].numpy()
                else:
                    return t[0].permute(1, 2, 0).numpy()

            num_intermediates = len(intermediate_patches)
            cols = 5
            rows_for_intermediates = (num_intermediates + cols - 1) // cols
            total_rows = 1 + rows_for_intermediates

            fig = plt.figure(figsize=(15, 3 * total_rows))
            gs = fig.add_gridspec(total_rows, cols)

            cmap = "gray" if c == 1 else None

            # Top row: True, Masked, Reconstructed
            ax1 = fig.add_subplot(gs[0, 0])
            conf_after, ent_after = get_confidence_and_entropy(final_img)
            dist_before = get_dist_to_mean(true_img)
            dist_with_masked = get_dist_to_mean(initial_img)
            dist_after = get_dist_to_mean(final_img)

            mse_masked = F.mse_loss(initial_img, true_img).item()
            mse_final = F.mse_loss(final_img, true_img).item()

            ax1.imshow(to_numpy(true_img), cmap=cmap)
            ax1.set_title(
                f"True Image\nConf: {conf_before * 100:.1f}% | Ent: {ent_before:.4f} | Dist: {dist_before:.2f}"
            )
            ax1.axis("off")

            ax2 = fig.add_subplot(gs[0, 2])
            ax2.imshow(to_numpy(initial_img), cmap=cmap)
            ax2.set_title(
                f"Initial Patch\nConf: {conf_with_masked * 100:.1f}% | Ent: {ent_with_masked:.4f} | Dist: {dist_with_masked:.2f}\nMSE: {mse_masked:.4f}"
            )
            ax2.axis("off")

            ax3 = fig.add_subplot(gs[0, 4])
            ax3.imshow(to_numpy(final_img), cmap=cmap)
            ax3.set_title(
                f"Reconstructed\nConf: {conf_after * 100:.1f}% | Ent: {ent_after:.4f} | Dist: {dist_after:.2f}\nMSE: {mse_final:.4f}"
            )
            ax3.axis("off")

            # Intermediate steps
            for idx, (step_i, p_val) in enumerate(intermediate_patches):
                dist = intermediate_dists[idx]
                r = 1 + idx // cols
                c_idx = idx % cols
                ax = fig.add_subplot(gs[r, c_idx])
                temp_img = context_img.clone()
                temp_img[:, :, start_y : start_y + patch_size, start_x : start_x + patch_size] = (
                    p_val
                )
                conf, ent = get_confidence_and_entropy(temp_img)
                mse_val = F.mse_loss(temp_img, true_img).item()
                ax.imshow(to_numpy(temp_img), cmap=cmap)
                ax.set_title(
                    f"Step {step_i}\nConf: {conf * 100:.1f}% | Ent: {ent:.4f} | Dist: {dist:.2f}\nMSE: {mse_val:.4f}"
                )
                ax.axis("off")

            plt.tight_layout()

            temp_dir = tempfile.mkdtemp()
            out_path = os.path.join(temp_dir, f"patch_inversion_{dataset_split}_{img_index}.png")
            plt.savefig(out_path, bbox_inches="tight")

            mlflow.log_artifact(out_path)
            print("Patch inversion result saved to MLflow artifacts.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True, help="MLflow run ID")
    parser.add_argument("--step", type=int, required=True, help="Checkpoint step (e.g., 150000)")
    parser.add_argument(
        "--img_index", type=int, default=0, help="Index of the image in the dataset"
    )
    parser.add_argument("--patch_size", type=int, default=5, help="Size of the missing patch")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument(
        "--num_iters", type=int, default=1500, help="Number of optimization iterations"
    )
    parser.add_argument(
        "--split", type=str, default="train", choices=["train", "test"], help="Dataset split"
    )
    parser.add_argument(
        "--jitter", type=int, default=2, help="Max random pixel shift per iteration (0 to disable)"
    )
    parser.add_argument(
        "--losses",
        type=str,
        default="feature=1.0,tv=1e-3",
        help="Comma separated list of loss_name=weight. "
        "Keys: feature (multi-layer cosine), repr_mse, ce, conf, neg_ent, mse, tv, l2",
    )
    parser.add_argument(
        "--use_context_matching",
        action="store_true",
        help="Match color/variance of surrounding pixels",
    )
    parser.add_argument(
        "--use_dip", action="store_true", help="Use Deep Image Prior (CNN) instead of raw pixels"
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        default=0.0,
        help="Std of Gaussian noise added during optimization",
    )
    parser.add_argument(
        "--scale_jitter", action="store_true", help="Randomly scale patch during optimization"
    )
    parser.add_argument("--start_y", type=int, default=-1, help="Start Y of patch (-1 for center)")
    parser.add_argument("--start_x", type=int, default=-1, help="Start X of patch (-1 for center)")

    args = parser.parse_args()

    losses_dict = {}
    valid_keys = {
        "feature",
        "ce",
        "neg_ent",
        "conf",
        "logit",
        "mse",
        "repr_mse",
        "grad_reg",
        "tv",
        "l2",
        "weight_grad",
        "feature_stats",
    }
    for pair in args.losses.split(","):
        if "=" in pair:
            k, v = pair.split("=")
            k = k.strip()
            if k not in valid_keys:
                raise ValueError(
                    f"Invalid loss key: '{k}'. Valid keys are: {', '.join(sorted(valid_keys))}"
                )
            losses_dict[k] = float(v.strip())

    run_patch_inversion(
        args.run_id,
        args.step,
        args.img_index,
        args.patch_size,
        args.lr,
        args.num_iters,
        args.split,
        args.use_context_matching,
        args.use_dip,
        args.jitter,
        args.noise_std,
        args.scale_jitter,
        losses_dict,
        args.start_y,
        args.start_x,
    )


"""
### How this script works (DeepInversion-style patch inversion):
This script performs 'Patch Inversion' to reconstruct a masked patch of a training image.

Key improvements over basic feature matching:
1. **Multi-layer feature matching**: Hooks capture activations at ALL intermediate layers
   (conv1, conv2, linear1, etc.) and matches them via cosine similarity to the true 
   image's activations. This provides far more spatial constraints than penultimate-layer-only.
2. **Cosine similarity**: Scale-invariant matching (better than MSE for feature matching).
3. **Pixel clamping**: Projects patch values back to valid range every iteration.
4. **Input jitter**: Random small translations prevent adversarial pixel patterns.
5. **Multi-scale TV**: Encourages smoothness at multiple spatial scales.
6. **L2 regularization**: Optional sparsity prior (useful for MNIST).

Use the new `feature` loss key to enable multi-layer matching:
    --losses "feature=1.0,tv=1e-3"
"""
