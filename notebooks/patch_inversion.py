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

def run_patch_inversion(run_id, step, img_index=0, patch_size=5, lr=0.1, num_iters=1000, dataset_split="train"):
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

    # The patch parameters we optimize
    patch = torch.randn((1, c, patch_size, patch_size), device=device, requires_grad=True)
    
    optimizer = optim.Adam([patch], lr=lr)
    criterion = config.loss(num_classes=num_classes)
    
    print(f"Starting Patch Inversion for True Label: {true_label}")
    
    with mlflow.start_run(run_id=run_id):
        mlflow.log_params({
            f"pi_{dataset_split}_{img_index}_checkpoint_step": step,
            f"pi_{dataset_split}_{img_index}_patch_size": patch_size,
            f"pi_{dataset_split}_{img_index}_lr": lr,
            f"pi_{dataset_split}_{img_index}_num_iters": num_iters
        })
        
        for i in range(num_iters):
            optimizer.zero_grad()
            
            # Reconstruct the full image
            full_img = masked_img.clone()
            full_img[:, :, start_y:start_y+patch_size, start_x:start_x+patch_size] = patch
            
            # Forward pass
            output = model(full_img)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
                
            task_loss = criterion(logits, target_tensor)
            
            # TV loss to make patch smoother
            tv_h = torch.pow(patch[:, :, 1:, :] - patch[:, :, :-1, :], 2).sum()
            tv_w = torch.pow(patch[:, :, :, 1:] - patch[:, :, :, :-1], 2).sum()
            tv_loss = (tv_h + tv_w) / (c * patch_size * patch_size)
            
            # small weight for TV loss
            loss = task_loss + 1e-4 * tv_loss
            
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0:
                print(f"Iter {i:04d} | Loss: {loss.item():.4f} | TaskLoss: {task_loss.item():.4f} | TV: {tv_loss.item():.4f}")
                mlflow.log_metrics({
                    f"pi_{dataset_split}_{img_index}_total_loss": loss.item(), 
                    f"pi_{dataset_split}_{img_index}_task_loss": task_loss.item(), 
                    f"pi_{dataset_split}_{img_index}_tv_loss": tv_loss.item()
                }, step=i)
        
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
            
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            cmap = 'gray' if c == 1 else None
            
            axes[0].imshow(to_numpy(true_img), cmap=cmap)
            axes[0].set_title("True Image")
            axes[0].axis('off')
            
            axes[1].imshow(to_numpy(masked_img), cmap=cmap)
            axes[1].set_title(f"Masked Image ({patch_size}x{patch_size} missing)")
            axes[1].axis('off')
            
            axes[2].imshow(to_numpy(final_img), cmap=cmap)
            axes[2].set_title(f"Reconstructed Image")
            axes[2].axis('off')
            
            temp_dir = tempfile.mkdtemp()
            out_path = os.path.join(temp_dir, f"patch_inversion_{img_index}.png")
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
    
    args = parser.parse_args()
    run_patch_inversion(args.run_id, args.step, args.img_index, args.patch_size, args.lr, args.num_iters, args.split)
