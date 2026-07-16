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
import torch.nn.functional as F

def run_image_inversion_dlg(run_id, step, img_index, lr=0.1, num_iters=1500, split='train', use_dip=False, layer_matching='all'):
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
    dataset = data_container.train if split == "train" else data_container.test
    true_img, true_label = dataset[img_index]
    
    # ensure it has a batch dimension
    true_img = true_img.unsqueeze(0).to(device)
    target_tensor = torch.tensor([true_label], device=device)

    _, c, h, w = true_img.shape
    
    # Initialize the whole dummy image with random noise
    noise_img = torch.randn((1, c, h, w), device=device) * 0.5

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
        dip_z = noise_img.clone()
        dummy_img = dip_model(dip_z)
    else:
        dummy_img = noise_img.clone().requires_grad_(True)
        optimizer = optim.Adam([dummy_img], lr=lr)
    
    def get_confidence(img_tensor):
        with torch.no_grad():
            if norm_mean is not None:
                img_tensor_in = (img_tensor - norm_mean) / norm_std
            else:
                img_tensor_in = img_tensor
            try:
                output = model(img_tensor_in, verbose=True)
                logits = output[0] if isinstance(output, tuple) else output
            except TypeError:
                output = model(img_tensor_in)
                logits = output[0] if isinstance(output, tuple) else output
            probs = F.softmax(logits, dim=1)
            return probs[0, true_label].item()

    conf_before = get_confidence(true_img)
    
    criterion = config.loss(num_classes=num_classes)
    
    print(f"Starting DLG Image Inversion for True Label: {true_label}")
    
    # Compute target gradients from the true image
    model.zero_grad()
    if norm_mean is not None:
        target_input = (true_img - norm_mean) / norm_std
    else:
        target_input = true_img
    try:
        target_output = model(target_input, verbose=True)
        target_logits = target_output[0] if isinstance(target_output, tuple) else target_output
    except TypeError:
        target_output = model(target_input)
        target_logits = target_output[0] if isinstance(target_output, tuple) else target_output
        
    target_loss = criterion(target_logits, target_tensor)
    
    params = [p for p in model.parameters() if p.requires_grad]
    target_dy_dx = torch.autograd.grad(target_loss, params)
    # Detach to prevent graph retention issues
    target_dy_dx = [g.detach() for g in target_dy_dx]
    
    target_grad_flat = torch.cat([g.flatten() for g in target_dy_dx])

    with mlflow.start_run(run_id=run_id):
        
        intermediate_imgs = []
        intermediate_dists = []
        
        for i in range(num_iters):
            if use_dip:
                dummy_img = dip_model(dip_z)
            
            model.zero_grad()
            if norm_mean is not None:
                model_input = (dummy_img - norm_mean) / norm_std
            else:
                model_input = dummy_img
            try:
                output = model(model_input, verbose=True)
                logits = output[0] if isinstance(output, tuple) else output
            except TypeError:
                output = model(model_input)
                logits = output[0] if isinstance(output, tuple) else output
                
            dummy_loss = criterion(logits, target_tensor)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, params, create_graph=True)
            
            num_layers = len(target_dy_dx)
            if args.layer_matching == 'all':
                active_layers = range(num_layers)
            else:
                steps_per_phase = max(1, num_iters // num_layers)
                phase = min(i // steps_per_phase, num_layers - 1)
                if args.layer_matching == 'forward':
                    active_layers = range(0, phase + 1)
                elif args.layer_matching == 'backward':
                    active_layers = range(num_layers - 1 - phase, num_layers)
                    
            grad_diff = 0
            for l_idx in active_layers:
                gx = dummy_dy_dx[l_idx]
                gy = target_dy_dx[l_idx]
                grad_diff += ((gx - gy) ** 2).sum()
            
            # TV loss to make image smoother
            tv_h = torch.pow(dummy_img[:, :, 1:, :] - dummy_img[:, :, :-1, :], 2).sum()
            tv_w = torch.pow(dummy_img[:, :, :, 1:] - dummy_img[:, :, :, :-1], 2).sum()
            tv_loss = (tv_h + tv_w) / (c * h * w)
            
            # Increase TV loss weight significantly because GradDiff starts around 400+
            loss = grad_diff + 1.0 * tv_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0:
                intermediate_imgs.append((i, dummy_img.clone().detach()))
                intermediate_dists.append(grad_diff.item())
                print(f"Iter {i:04d} | GradDiff: {grad_diff.item():.6f} | TV: {tv_loss.item():.4f}")
        
        # Plotting
        with torch.no_grad():
            final_img = dummy_img.clone()
            
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
            
            num_intermediates = len(intermediate_imgs)
            cols = 5
            rows_for_intermediates = (num_intermediates + cols - 1) // cols
            total_rows = 1 + rows_for_intermediates
            
            fig = plt.figure(figsize=(15, 3 * total_rows))
            gs = fig.add_gridspec(total_rows, cols)
            
            cmap = 'gray' if c == 1 else None
            
            # Top row: True, Initial, Reconstructed
            ax1 = fig.add_subplot(gs[0, 0])
            conf_after = get_confidence(final_img)
            mse_initial = torch.nn.functional.mse_loss(noise_img, true_img).item()
            mse_final = torch.nn.functional.mse_loss(final_img, true_img).item()
            
            ax1.imshow(to_numpy(true_img), cmap=cmap)
            ax1.set_title(f"True Image\nConf: {conf_before*100:.1f}%")
            ax1.axis('off')
            
            ax2 = fig.add_subplot(gs[0, 2])
            ax2.imshow(to_numpy(noise_img), cmap=cmap)
            ax2.set_title(f"Initial Noise\nMSE: {mse_initial:.4f}")
            ax2.axis('off')
            
            ax3 = fig.add_subplot(gs[0, 4])
            ax3.imshow(to_numpy(final_img), cmap=cmap)
            ax3.set_title(f"Reconstructed\nConf: {conf_after*100:.1f}% | MSE: {mse_final:.4f}")
            ax3.axis('off')
            
            # Intermediate steps
            for idx, (step_i, img_val) in enumerate(intermediate_imgs):
                grad_d = intermediate_dists[idx]
                r = 1 + idx // cols
                c_idx = idx % cols
                ax = fig.add_subplot(gs[r, c_idx])
                conf = get_confidence(img_val)
                mse_val = torch.nn.functional.mse_loss(img_val, true_img).item()
                ax.imshow(to_numpy(img_val), cmap=cmap)
                ax.set_title(f"Step {step_i}\nConf: {conf*100:.1f}% | MSE: {mse_val:.4f}\nGradDiff: {grad_d:.4f}")
                ax.axis('off')
            
            plt.tight_layout()
            
            temp_dir = tempfile.mkdtemp()
            out_path = os.path.join(temp_dir, f"image_inversion_dlg_{img_index}.png")
            plt.savefig(out_path, bbox_inches='tight')
            
            mlflow.log_artifact(out_path)
            print(f"DLG Image inversion result saved to MLflow artifacts.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True, help="MLflow run ID")
    parser.add_argument("--step", type=int, required=True, help="Checkpoint step (e.g., 150000)")
    parser.add_argument("--img_index", type=int, default=0, help="Index of the image in the dataset")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--layer_matching", type=str, default="all", choices=["all", "forward", "backward"], help="Layer matching strategy")
    parser.add_argument("--num_iters", type=int, default=1500, help="Number of optimization iterations")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"], help="Dataset split")
    parser.add_argument("--use_dip", action="store_true", help="Use Deep Image Prior (CNN) instead of raw pixels")
    
    args = parser.parse_args()
    run_image_inversion_dlg(args.run_id, args.step, args.img_index, args.lr, args.num_iters, args.split, args.use_dip, args.layer_matching)

"""
### How this script works:
This script performs 'Full Image Inversion' using Deep Leakage from Gradients (DLG).
1. It takes a true image and computes the 'true gradients' for that image.
2. It initializes a completely random noise image.
3. It iteratively tweaks every single pixel in the noise image to minimize the difference 
   between its gradients and the true gradients.
4. It uses Total Variation (TV) loss to encourage the image to be smooth (like a real photo) 
   rather than noisy. It can also optionally match gradients layer-by-layer to stabilize 
   the inversion for deeper networks.
"""
