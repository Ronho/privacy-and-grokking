import argparse
import json
import os
import tempfile
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import mlflow
from pathlib import Path

from privacy_and_grokking.config import TrainConfig
from privacy_and_grokking.utils.mlflow import TRACKING_URI
from privacy_and_grokking.utils.logger import Logger

def total_variation_loss(img):
    """Calculate the total variation loss of an image to encourage smoothness."""
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    return (tv_h + tv_w) / (c_img * h_img * w_img)

def run_model_inversion(run_id, step, target_class=0, lr=0.1, num_iters=500, jitter=2, losses_dict=None):
    if losses_dict is None:
        losses_dict = {'ce': 1.0, 'tv': 1e-3, 'l2': 0.01}
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
    # Check if dataset is MNIST to conditionally apply L2
    is_mnist = config_dict.get("data", {}).get("data", {}).get("name", "").lower() == "mnist"

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

    # Initialize with uniform noise in [0, 1]
    dummy_img = torch.rand((1, *input_shape), device=device, requires_grad=True)
    
    optimizer = optim.Adam([dummy_img], lr=lr)
    criterion = config.loss(num_classes=num_classes)
    
    target_tensor = torch.tensor([target_class], device=device)
    
    # If the criterion involves internal representation, we need the feature means of the target class
    class_mean_features = None
    if losses_dict.get("repr_mse", 0.0) > 0.0:
        print("Computing class mean for target label...")
        train_loader = torch.utils.data.DataLoader(data_container.train, batch_size=256, shuffle=False)
        features_list = []
        with torch.no_grad():
            for imgs, lbls in train_loader:
                mask_lbl = lbls == target_class
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
        else:
            print(f"Warning: No samples found for class {target_class} in training set.")
            class_mean_features = None

    print(f"Starting Model Inversion for target class: {target_class}")
    
    with mlflow.start_run(run_id=run_id):
        log_dict = {
            f"mi_c{target_class}_checkpoint_step": step,
            f"mi_c{target_class}_lr": lr,
            f"mi_c{target_class}_num_iters": num_iters,
            f"mi_c{target_class}_jitter": jitter
        }
        for k, v in losses_dict.items():
            log_dict[f"mi_c{target_class}_loss_{k}"] = v
        mlflow.log_params(log_dict)
        
        intermediate_imgs = []
        
        for i in range(num_iters):
            optimizer.zero_grad()
            
            # Random Jitter (Translation)
            if jitter > 0:
                shift_x = torch.randint(-jitter, jitter + 1, (1,)).item()
                shift_y = torch.randint(-jitter, jitter + 1, (1,)).item()
                jittered_img = torch.roll(dummy_img, shifts=(shift_y, shift_x), dims=(-2, -1))
            else:
                jittered_img = dummy_img
            
            # Forward pass
            if norm_mean is not None:
                model_input = (jittered_img - norm_mean) / norm_std
            else:
                model_input = jittered_img
                
            if losses_dict.get("repr_mse", 0.0) > 0.0:
                try:
                    output = model(model_input, verbose=True)
                    logits, feats = output if isinstance(output, tuple) else (output, output)
                except TypeError:
                    output = model(model_input)
                    logits = output[0] if isinstance(output, tuple) else output
                    feats = output[1] if isinstance(output, tuple) else output
            else:
                output = model(model_input)
                logits = output[0] if isinstance(output, tuple) else output
                
            loss = torch.tensor(0.0, device=device)
            metrics_vals = {}
            
            probs = F.softmax(logits, dim=1)
            
            if losses_dict.get("ce", 0) > 0:
                ce_val = criterion(logits, target_tensor)
                loss = loss + losses_dict["ce"] * ce_val
                metrics_vals["ce"] = ce_val.item()
                
            if losses_dict.get("neg_ent", 0) > 0:
                entropy_loss = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
                loss = loss + losses_dict["neg_ent"] * entropy_loss
                metrics_vals["neg_ent"] = entropy_loss.item()
                
            if losses_dict.get("conf", 0) > 0:
                conf_val = probs[:, target_class].mean()
                loss = loss - losses_dict["conf"] * conf_val  # minimize negative confidence
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
                tv_loss = total_variation_loss(dummy_img)
                loss = loss + losses_dict["tv"] * tv_loss
                metrics_vals["tv"] = tv_loss.item()
                
            if losses_dict.get("l2", 0) > 0 and is_mnist:
                l2_loss = torch.norm(dummy_img, p=2)
                loss = loss + losses_dict["l2"] * l2_loss
                metrics_vals["l2"] = l2_loss.item()
            
            loss.backward()
            optimizer.step()
            
            # Projected Gradient Descent: clamp values to valid image range
            with torch.no_grad():
                dummy_img.clamp_(0, 1)
            
            if i % 100 == 0 or i == num_iters - 1:
                parts = [f"Iter {i:04d}", f"Total: {loss.item():.4f}"]
                if "repr_mse" in metrics_vals: parts.append(f"ReprMSE: {metrics_vals['repr_mse']:.4f}")
                if "mse" in metrics_vals: parts.append(f"OutMSE: {metrics_vals['mse']:.4f}")
                if "ce" in metrics_vals: parts.append(f"CELoss: {metrics_vals['ce']:.4f}")
                if "conf" in metrics_vals: parts.append(f"Conf: {metrics_vals['conf']:.4f}")
                if "neg_ent" in metrics_vals: parts.append(f"NegEnt: {metrics_vals['neg_ent']:.4f}")
                if "tv" in metrics_vals: parts.append(f"TV: {metrics_vals['tv']:.4f}")
                if "l2" in metrics_vals: parts.append(f"L2: {metrics_vals['l2']:.4f}")
                
                print(" | ".join(parts))
                
                intermediate_imgs.append((i, dummy_img.clone().detach(), metrics_vals.copy()))
        
        # Plotting
        with torch.no_grad():
            def to_numpy(t):
                t = t.clone().detach().cpu()
                if input_shape[0] == 1:
                    return t[0, 0].numpy()
                else:
                    return t[0].permute(1, 2, 0).numpy()

            def get_title(step_i, metrics_dict):
                lines = [f"Step {step_i}"]
                
                # Primary Criteria
                if "repr_mse" in metrics_dict: lines.append(f"Repr MSE: {metrics_dict['repr_mse']:.4f}")
                if "mse" in metrics_dict: lines.append(f"Out MSE: {metrics_dict['mse']:.4f}")
                if "ce" in metrics_dict: lines.append(f"CE Loss: {metrics_dict['ce']:.4f}")
                if "conf" in metrics_dict: lines.append(f"Conf: {metrics_dict['conf']:.4f}")
                if "neg_ent" in metrics_dict: lines.append(f"Neg Ent: {metrics_dict['neg_ent']:.4f}")
                
                # Regularization components
                reg_parts = []
                if "tv" in metrics_dict: reg_parts.append(f"TV: {metrics_dict['tv']:.4f}")
                if "l2" in metrics_dict: reg_parts.append(f"L2: {metrics_dict['l2']:.4f}")
                if reg_parts:
                    lines.append(" | ".join(reg_parts))
                
                return "\n".join(lines)
            
            num_intermediates = len(intermediate_imgs)
            cols = min(5, num_intermediates)
            rows = (num_intermediates + cols - 1) // cols
            
            fig = plt.figure(figsize=(3 * cols, 3 * rows))
            gs = fig.add_gridspec(rows, cols)
            cmap = 'gray' if input_shape[0] == 1 else None
            
            for idx, (step_i, img_val, metrics_dict) in enumerate(intermediate_imgs):
                r = idx // cols
                c_idx = idx % cols
                ax = fig.add_subplot(gs[r, c_idx])
                ax.imshow(to_numpy(img_val), cmap=cmap)
                ax.set_title(get_title(step_i, metrics_dict), fontsize=9)
                ax.axis('off')
                
            plt.tight_layout()
            
            temp_dir = tempfile.mkdtemp()
            out_path = os.path.join(temp_dir, f"model_inversion_class_{target_class}.png")
            plt.savefig(out_path, bbox_inches='tight')
            
            mlflow.log_artifact(out_path)
            print(f"Reconstructed images grid saved to MLflow artifacts.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True, help="MLflow run ID")
    parser.add_argument("--step", type=int, required=True, help="Checkpoint step (e.g., 150000)")
    parser.add_argument("--target_class", type=int, default=0, help="Class to invert")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument('--num_iters', type=int, default=500, help="Number of optimization iterations.")
    parser.add_argument('--jitter', type=int, default=2, help="Max random pixel shift for robust optimization.")
    parser.add_argument('--losses', type=str, default='ce=1.0,tv=0.001,l2=0.01', help="Comma separated list of loss_name=weight. E.g. 'ce=1.0,neg_ent=0.1,tv=0.001'")
    
    args = parser.parse_args()
    
    losses_dict = {}
    valid_keys = {"ce", "neg_ent", "conf", "mse", "repr_mse", "tv", "l2"}
    for pair in args.losses.split(','):
        if '=' in pair:
            k, v = pair.split('=')
            k = k.strip()
            if k not in valid_keys:
                raise ValueError(f"Invalid loss key: '{k}'. Valid keys are: {', '.join(sorted(valid_keys))}")
            losses_dict[k] = float(v.strip())
            
    run_model_inversion(args.run_id, args.step, args.target_class, args.lr, args.num_iters, args.jitter, losses_dict)
