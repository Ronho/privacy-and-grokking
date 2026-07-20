import argparse
import json
import os
import tempfile
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import mlflow
from pathlib import Path

from privacy_and_grokking.config import TrainConfig
from privacy_and_grokking.utils.mlflow import TRACKING_URI
from privacy_and_grokking.utils.logger import Logger

def run_patch_inversion(run_id, step, img_index=0, patch_size=5, lr=0.1, num_iters=1000, dataset_split="train", use_context_matching=False, use_dip=False, losses_dict=None):
    if losses_dict is None:
        losses_dict = {'ce': 1.0, 'tv': 1e-4}
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

    # Get image
    dataset = data_container.train if dataset_split == "train" else data_container.test
    true_img, true_label = dataset[img_index]
    
    # ensure it has a batch dimension
    true_img = true_img.unsqueeze(0).to(device)
    target_tensor = torch.tensor([true_label], device=device)

    # Create mask (1 for unmasked, 0 for masked)
    mask = torch.ones_like(true_img, device=device)
    _, c, h, w = true_img.shape
    
    start_y = (h - patch_size) // 2
    start_x = (w - patch_size) // 2
    
    mask[:, :, start_y:start_y+patch_size, start_x:start_x+patch_size] = 0
    
    masked_img = true_img * mask
    
    # Initialize the missing patch with random noise
    noise_patch = torch.randn((1, c, patch_size, patch_size), device=device) * 0.5
    masked_img[:, :, start_y:start_y+patch_size, start_x:start_x+patch_size] = noise_patch

    # The patch parameters we optimize
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
                    nn.Conv2d(32, channels, kernel_size=3, padding=1)
                )
            def forward(self, x):
                return self.net(x)
        dip_model = TinyDIP(c).to(device)
        optimizer = optim.Adam(dip_model.parameters(), lr=lr)
        dip_z = noise_patch.clone()
        patch = dip_model(dip_z) # initial patch
    else:
        patch = noise_patch.clone().requires_grad_(True)
        optimizer = optim.Adam([patch], lr=lr)
    
    import torch.nn.functional as F
    def get_confidence_and_entropy(img_tensor):
        with torch.no_grad():
            if norm_mean is not None:
                img_tensor = (img_tensor - norm_mean) / norm_std
            output = model(img_tensor)
            logits = output[0] if isinstance(output, tuple) else output
            probs = F.softmax(logits, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean().item()
            return probs[0, true_label].item(), entropy

    conf_before, ent_before = get_confidence_and_entropy(true_img)
    conf_with_masked, ent_with_masked = get_confidence_and_entropy(masked_img)
    
    # optimizer is defined above based on use_dip
    criterion = config.loss(num_classes=num_classes)
    
    print(f"Starting Patch Inversion for True Label: {true_label}")
    
    class_mean_features = None
    if losses_dict.get("repr_mse", 0.0) > 0.0:
        print("Computing class mean for true label...")
        model.eval()
        train_loader = torch.utils.data.DataLoader(data_container.train, batch_size=256, shuffle=False)
        features_list = []
        with torch.no_grad():
            for imgs, lbls in train_loader:
                mask_lbl = lbls == true_label
                if not mask_lbl.any():
                    continue
                imgs = imgs[mask_lbl].to(device)
                if norm_mean is not None:
                    imgs_input = (imgs - norm_mean) / norm_std
                else:
                    imgs_input = imgs
                try:
                    output = model(imgs_input, verbose=True)
                    feats = output[1] if isinstance(output, tuple) else output
                except TypeError:
                    output = model(imgs_input)
                    feats = output[1] if isinstance(output, tuple) else output
                features_list.append(feats)
                
        if len(features_list) > 0:
            class_mean_features = torch.cat(features_list).mean(dim=0, keepdim=True)
        
    def get_dist_to_mean(img_tensor):
        if class_mean_features is None:
            return 0.0
        with torch.no_grad():
            if norm_mean is not None:
                img_input = (img_tensor - norm_mean) / norm_std
            else:
                img_input = img_tensor
            try:
                output = model(img_input, verbose=True)
                feats = output[1] if isinstance(output, tuple) else output
            except TypeError:
                output = model(img_input)
                feats = output[1] if isinstance(output, tuple) else output
            return torch.norm(feats - class_mean_features, p=2).item()
    
    with mlflow.start_run(run_id=run_id):
        
        intermediate_patches = []
        intermediate_dists = []
        for i in range(num_iters):
            optimizer.zero_grad()
            
            if use_dip:
                patch = dip_model(dip_z)
            
            # Reconstruct the full image
            full_img = masked_img.clone()
            full_img[:, :, start_y:start_y+patch_size, start_x:start_x+patch_size] = patch
            
            # Forward pass
            if norm_mean is not None:
                model_input = (full_img - norm_mean) / norm_std
            else:
                model_input = full_img
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
            
            # Using original target label for patch inversion (which is to reconstruct the true patch)
            target_tensor = torch.tensor([true_label], device=device)
            
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
                
            if losses_dict.get("mse", 0) > 0:
                target_one_hot = F.one_hot(target_tensor, num_classes=num_classes).float()
                output_mse_val = F.mse_loss(probs, target_one_hot)
                loss = loss + losses_dict["mse"] * output_mse_val
                metrics_vals["mse"] = output_mse_val.item()
                
            if losses_dict.get("repr_mse", 0) > 0 and class_mean_features is not None:
                repr_mse_val = torch.nn.functional.mse_loss(feats, class_mean_features, reduction='sum')
                loss = loss + losses_dict["repr_mse"] * repr_mse_val
                metrics_vals["repr_mse"] = repr_mse_val.item()
                
            if losses_dict.get("tv", 0) > 0:
                tv_h = torch.pow(patch[:, :, 1:, :] - patch[:, :, :-1, :], 2).sum()
                tv_w = torch.pow(patch[:, :, :, 1:] - patch[:, :, :, :-1], 2).sum()
                tv_loss = (tv_h + tv_w) / (c * patch_size * patch_size)
                loss = loss + losses_dict["tv"] * tv_loss
                metrics_vals["tv"] = tv_loss.item()

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
                context_loss = torch.nn.functional.mse_loss(patch_mean, surround_mean) + torch.nn.functional.mse_loss(patch_var, surround_var)
                loss += 0.1 * context_loss
            
            dist_to_mean = torch.norm(feats - class_mean_features, p=2).item() if class_mean_features is not None else 0.0
            
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0:
                intermediate_patches.append((i, patch.clone().detach()))
                intermediate_dists.append(dist_to_mean)
                
                parts = [f"Iter {i:04d}", f"Total: {loss.item():.4f}"]
                if "repr_mse" in metrics_vals: parts.append(f"ReprMSE: {metrics_vals['repr_mse']:.4f}")
                if "mse" in metrics_vals: parts.append(f"OutMSE: {metrics_vals['mse']:.4f}")
                if "ce" in metrics_vals: parts.append(f"CELoss: {metrics_vals['ce']:.4f}")
                if "conf" in metrics_vals: parts.append(f"Conf: {metrics_vals['conf']:.4f}")
                if "neg_ent" in metrics_vals: parts.append(f"NegEnt: {metrics_vals['neg_ent']:.4f}")
                if "tv" in metrics_vals: parts.append(f"TV: {metrics_vals['tv']:.4f}")
                
                print(" | ".join(parts))
        
        # Plotting
        with torch.no_grad():
            final_img = masked_img.clone()
            final_img[:, :, start_y:start_y+patch_size, start_x:start_x+patch_size] = patch
            
            # Format for plotting
            # Get true image min/max for scaling
            true_min = true_img.min().item()
            true_max = true_img.max().item()
            
            def to_numpy(t):
                t = t.clone().detach().cpu()
                t = torch.clamp(t, true_min, true_max)
                t -= true_min
                t /= (true_max - true_min + 1e-8)
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
            
            cmap = 'gray' if c == 1 else None
            
            # Top row: True, Masked, Reconstructed
            ax1 = fig.add_subplot(gs[0, 0])
            conf_after, ent_after = get_confidence_and_entropy(final_img)
            dist_before = get_dist_to_mean(true_img)
            dist_with_masked = get_dist_to_mean(masked_img)
            dist_after = get_dist_to_mean(final_img)
            
            mse_masked = torch.nn.functional.mse_loss(masked_img, true_img).item()
            mse_final = torch.nn.functional.mse_loss(final_img, true_img).item()
            
            ax1.imshow(to_numpy(true_img), cmap=cmap)
            ax1.set_title(f"True Image\nConf: {conf_before*100:.1f}% | Ent: {ent_before:.4f} | Dist: {dist_before:.2f}")
            ax1.axis('off')
            
            ax2 = fig.add_subplot(gs[0, 2])
            ax2.imshow(to_numpy(masked_img), cmap=cmap)
            ax2.set_title(f"Initial Patch\nConf: {conf_with_masked*100:.1f}% | Ent: {ent_with_masked:.4f} | Dist: {dist_with_masked:.2f}\nMSE: {mse_masked:.4f}")
            ax2.axis('off')
            
            ax3 = fig.add_subplot(gs[0, 4])
            ax3.imshow(to_numpy(final_img), cmap=cmap)
            ax3.set_title(f"Reconstructed\nConf: {conf_after*100:.1f}% | Ent: {ent_after:.4f} | Dist: {dist_after:.2f}\nMSE: {mse_final:.4f}")
            ax3.axis('off')
            
            # Intermediate steps
            for idx, (step_i, p_val) in enumerate(intermediate_patches):
                dist = intermediate_dists[idx]
                r = 1 + idx // cols
                c_idx = idx % cols
                ax = fig.add_subplot(gs[r, c_idx])
                temp_img = masked_img.clone()
                temp_img[:, :, start_y:start_y+patch_size, start_x:start_x+patch_size] = p_val
                conf, ent = get_confidence_and_entropy(temp_img)
                mse_val = torch.nn.functional.mse_loss(temp_img, true_img).item()
                ax.imshow(to_numpy(temp_img), cmap=cmap)
                ax.set_title(f"Step {step_i}\nConf: {conf*100:.1f}% | Ent: {ent:.4f} | Dist: {dist:.2f}\nMSE: {mse_val:.4f}")
                ax.axis('off')
            
            plt.tight_layout()
            
            temp_dir = tempfile.mkdtemp()
            out_path = os.path.join(temp_dir, f"patch_inversion_{dataset_split}_{img_index}.png")
            plt.savefig(out_path, bbox_inches='tight')
            
            mlflow.log_artifact(out_path)
            print(f"Patch inversion result saved to MLflow artifacts.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True, help="MLflow run ID")
    parser.add_argument("--step", type=int, required=True, help="Checkpoint step (e.g., 150000)")
    parser.add_argument("--img_index", type=int, default=0, help="Index of the image in the dataset")
    parser.add_argument("--patch_size", type=int, default=5, help="Size of the missing patch")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--num_iters", type=int, default=1500, help="Number of optimization iterations")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"], help="Dataset split")
    parser.add_argument("--losses", type=str, default='ce=1.0,tv=1e-4', help="Comma separated list of loss_name=weight")
    parser.add_argument("--use_context_matching", action="store_true", help="Match color/variance of surrounding pixels")
    parser.add_argument("--use_dip", action="store_true", help="Use Deep Image Prior (CNN) instead of raw pixels")
    
    args = parser.parse_args()
    
    losses_dict = {}
    valid_keys = {"ce", "neg_ent", "conf", "mse", "repr_mse", "tv"}
    for pair in args.losses.split(','):
        if '=' in pair:
            k, v = pair.split('=')
            k = k.strip()
            if k not in valid_keys:
                raise ValueError(f"Invalid loss key: '{k}'. Valid keys are: {', '.join(sorted(valid_keys))}")
            losses_dict[k] = float(v.strip())
            
    run_patch_inversion(args.run_id, args.step, args.img_index, args.patch_size, args.lr, args.num_iters, args.split, args.use_context_matching, args.use_dip, losses_dict)

"""
### How this script works:
This script performs 'Patch Inversion' using feature matching.
1. It takes a true image and masks out a small patch (e.g., 5x5 pixels).
2. It passes the masked image through the model to get the intermediate features.
3. It tries to reconstruct the missing patch by optimizing the pixel values of the patch 
   so that the model's intermediate features for the reconstructed image match the 
   mean features of the target class.
4. It only optimizes the missing patch, keeping the rest of the image fixed (which acts as a strong prior).
"""
