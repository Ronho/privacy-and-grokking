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

def run_model_inversion(run_id: str, step: int, target_class: int, lr: float, num_iters: int, tv_weight: float, l2_weight: float, jitter: int, criterion_choice: str = "ce_loss", entropy_weight: float = 0.1):
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
    
    class_mean_features = None
    if criterion_choice in ["internal_representation_mse_loss", "internal_representation_mse_loss_and_negative_entropy_loss"]:
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
        mlflow.log_params({
            f"mi_c{target_class}_checkpoint_step": step,
            f"mi_c{target_class}_lr": lr,
            f"mi_c{target_class}_num_iters": num_iters,
            f"mi_c{target_class}_tv_weight": tv_weight,
            f"mi_c{target_class}_l2_weight": l2_weight,
            f"mi_c{target_class}_jitter": jitter
        })
        
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
                
            if criterion_choice in ["internal_representation_mse_loss", "internal_representation_mse_loss_and_negative_entropy_loss"]:
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
                
            probs = F.softmax(logits, dim=1)
            entropy_loss = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
            conf_val = probs[:, target_class].mean()
            ce_val = criterion(logits, target_tensor)
            
            target_one_hot = F.one_hot(target_tensor, num_classes=num_classes).float()
            output_mse_val = F.mse_loss(probs, target_one_hot)
            
            repr_mse_val = None
            if criterion_choice in ["internal_representation_mse_loss", "internal_representation_mse_loss_and_negative_entropy_loss"] and class_mean_features is not None:
                repr_mse_val = torch.nn.functional.mse_loss(feats, class_mean_features)
                
            if criterion_choice == "ce_loss":
                task_loss = ce_val
            elif criterion_choice == "negative_entropy_loss":
                task_loss = entropy_loss
            elif criterion_choice == "internal_representation_mse_loss":
                task_loss = repr_mse_val if repr_mse_val is not None else torch.tensor(0.0, device=device)
            elif criterion_choice == "internal_representation_mse_loss_and_negative_entropy_loss":
                task_loss = (repr_mse_val if repr_mse_val is not None else torch.tensor(0.0, device=device)) + entropy_weight * entropy_loss
            elif criterion_choice == "ce_loss_and_negative_entropy_loss":
                task_loss = ce_val + entropy_weight * entropy_loss
            elif criterion_choice == "conf_loss":
                task_loss = -conf_val
            elif criterion_choice == "conf_loss_and_negative_entropy_loss":
                task_loss = -conf_val + entropy_weight * entropy_loss
            elif criterion_choice == "mse_loss":
                task_loss = output_mse_val
            elif criterion_choice == "mse_loss_and_negative_entropy_loss":
                task_loss = output_mse_val + entropy_weight * entropy_loss
            else:
                task_loss = ce_val
                
            tv_loss = total_variation_loss(dummy_img)
            # L2 Sparsity Loss
            l2_loss = torch.norm(dummy_img, p=2) if (is_mnist and l2_weight > 0) else torch.tensor(0.0, device=device)
            
            loss = task_loss + tv_weight * tv_loss + l2_weight * l2_loss
            
            loss.backward()
            optimizer.step()
            
            # Projected Gradient Descent: clamp values to valid image range
            with torch.no_grad():
                dummy_img.clamp_(0, 1)
            
            if i % 100 == 0 or i == num_iters - 1:
                metrics = {
                    "ce_loss": ce_val.item(),
                    "entropy": entropy_loss.item(),
                    "conf": conf_val.item(),
                    "output_mse": output_mse_val.item(),
                    "repr_mse": repr_mse_val.item() if repr_mse_val is not None else None
                }
                
                parts = [f"Iter {i:04d}", f"Total: {loss.item():.4f}"]
                if "internal_representation_mse" in criterion_choice and metrics["repr_mse"] is not None:
                    parts.append(f"ReprMSE: {metrics['repr_mse']:.4f}")
                elif "mse_loss" in criterion_choice:
                    parts.append(f"OutMSE: {metrics['output_mse']:.4f}")
                
                if "ce_loss" in criterion_choice:
                    parts.append(f"CELoss: {metrics['ce_loss']:.4f}")
                if "conf_loss" in criterion_choice:
                    parts.append(f"Conf: {metrics['conf']:.4f}")
                if "negative_entropy" in criterion_choice:
                    parts.append(f"NegEnt: {metrics['entropy']:.4f}")
                    
                print(" | ".join(parts))
                
                intermediate_imgs.append((i, dummy_img.clone().detach(), metrics))
        
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
                if "internal_representation_mse" in criterion_choice and metrics_dict.get("repr_mse") is not None:
                    lines.append(f"Repr MSE: {metrics_dict['repr_mse']:.4f}")
                elif "mse_loss" in criterion_choice:
                    lines.append(f"Out MSE: {metrics_dict['output_mse']:.4f}")
                    
                if "ce_loss" in criterion_choice:
                    lines.append(f"CE Loss: {metrics_dict['ce_loss']:.4f}")
                if "conf_loss" in criterion_choice:
                    lines.append(f"Conf: {metrics_dict['conf']:.4f}")
                if "negative_entropy" in criterion_choice:
                    lines.append(f"Neg Ent: {metrics_dict['entropy']:.4f}")
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
    parser.add_argument('--tv_weight', type=float, default=1e-3, help="Weight for Total Variation loss.")
    parser.add_argument('--l2_weight', type=float, default=0.01, help="Weight for L2 sparsity loss.")
    parser.add_argument('--jitter', type=int, default=2, help="Max random pixel shift for robust optimization.")
    parser.add_argument('--criterion', type=str, default='ce_loss', choices=[
        'ce_loss', 'negative_entropy_loss', 'internal_representation_mse_loss', 
        'internal_representation_mse_loss_and_negative_entropy_loss', 'ce_loss_and_negative_entropy_loss', 
        'conf_loss', 'conf_loss_and_negative_entropy_loss', 'mse_loss', 'mse_loss_and_negative_entropy_loss'
    ], help="Loss criterion for inversion.")
    parser.add_argument('--entropy_weight', type=float, default=0.1, help="Weight for entropy when combined with other losses.")
    
    args = parser.parse_args()
    
    run_model_inversion(args.run_id, args.step, args.target_class, args.lr, args.num_iters, args.tv_weight, args.l2_weight, args.jitter, args.criterion, args.entropy_weight)
