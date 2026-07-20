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
                
            if criterion_choice == "ce_loss":
                task_loss = criterion(logits, target_tensor)
            elif criterion_choice == "negative_entropy_loss":
                task_loss = entropy_loss
            elif criterion_choice == "internal_representation_mse_loss":
                task_loss = torch.nn.functional.mse_loss(feats, class_mean_features) if class_mean_features is not None else torch.tensor(0.0, device=device)
            elif criterion_choice == "internal_representation_mse_loss_and_negative_entropy_loss":
                mse_val = torch.nn.functional.mse_loss(feats, class_mean_features) if class_mean_features is not None else torch.tensor(0.0, device=device)
                task_loss = mse_val + entropy_weight * entropy_loss
            elif criterion_choice == "ce_loss_and_negative_entropy_loss":
                task_loss = criterion(logits, target_tensor) + entropy_weight * entropy_loss
            elif criterion_choice == "conf_loss":
                task_loss = -probs[:, target_class].mean()
            elif criterion_choice == "conf_loss_and_negative_entropy_loss":
                conf_val = -probs[:, target_class].mean()
                task_loss = conf_val + entropy_weight * entropy_loss
            elif criterion_choice == "mse_loss":
                target_one_hot = F.one_hot(target_tensor, num_classes=num_classes).float()
                task_loss = F.mse_loss(probs, target_one_hot)
            elif criterion_choice == "mse_loss_and_negative_entropy_loss":
                target_one_hot = F.one_hot(target_tensor, num_classes=num_classes).float()
                task_loss = F.mse_loss(probs, target_one_hot) + entropy_weight * entropy_loss
            else:
                task_loss = criterion(logits, target_tensor)
            tv_loss = total_variation_loss(dummy_img)
            # L2 Sparsity Loss
            l2_loss = torch.norm(dummy_img, p=2) if (is_mnist and l2_weight > 0) else torch.tensor(0.0, device=device)
            
            loss = task_loss + tv_weight * tv_loss + l2_weight * l2_loss
            
            loss.backward()
            optimizer.step()
            
            # Projected Gradient Descent: clamp values to valid image range
            with torch.no_grad():
                dummy_img.clamp_(0, 1)
            
            if i % 100 == 0:
                print(f"Iter {i:04d} | Loss: {loss.item():.4f} | TaskLoss: {task_loss.item():.4f} | Ent: {entropy_loss.item():.4f} | TV: {tv_loss.item():.4f} | L2: {l2_loss.item():.4f}")
                mlflow.log_metrics({
                    f"mi_c{target_class}_total_loss": loss.item(), 
                    f"mi_c{target_class}_task_loss": task_loss.item(), 
                    f"mi_c{target_class}_entropy": entropy_loss.item(),
                    f"mi_c{target_class}_tv_loss": tv_loss.item(),
                    f"mi_c{target_class}_l2_loss": l2_loss.item()
                }, step=i)
        
        # Save output
        with torch.no_grad():
            final_img = dummy_img.clone().detach().cpu()
            
            # calculate final entropy
            if norm_mean is not None:
                final_input = (dummy_img - norm_mean) / norm_std
            else:
                final_input = dummy_img
            
            final_out = model(final_input)
            final_logits = final_out[0] if isinstance(final_out, tuple) else final_out
            final_probs = F.softmax(final_logits, dim=1)
            final_entropy = -torch.sum(final_probs * torch.log(final_probs + 1e-8), dim=1).mean().item()
            
            # Image is already clamped to [0, 1], so we don't apply min/max scaling which artificially boosts noise
            
            # Plot
            plt.figure(figsize=(4, 4))
            if input_shape[0] == 1:
                plt.imshow(final_img[0, 0].numpy(), cmap='gray')
            else:
                plt.imshow(final_img[0].permute(1, 2, 0).numpy())
            plt.title(f"Inverted Image (Class {target_class})\nEnt: {final_entropy:.4f}")
            plt.axis('off')
            
            temp_dir = tempfile.mkdtemp()
            out_path = os.path.join(temp_dir, f"inverted_class_{target_class}.png")
            plt.savefig(out_path, bbox_inches='tight')
            
            mlflow.log_artifact(out_path)
            print(f"Reconstructed image saved to MLflow artifacts.")

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
