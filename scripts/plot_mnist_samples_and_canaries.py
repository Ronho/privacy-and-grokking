import argparse
import os
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

from privacy_and_grokking.datasets.sets.base import CACHE_PATH
from privacy_and_grokking.datasets.sets.mnist import MNISTConfig
from privacy_and_grokking.datasets.canaries import (
    GaussianNoiseCanary,
    UniformNoiseCanary,
    SquareWatermarkCanary,
    LabelNoiseCanary,
    OODNaturalCanary,
)


def plot_mnist_grid_2x5(dataset, output_dir: Path, dpi: int = 300):
    """
    Finds one clean sample for each digit 0-9 and plots them in a 2x5 grid.
    """
    samples_per_digit = {}
    
    # Collect one sample per digit (0 to 9)
    for idx in range(len(dataset)):
        img, label = dataset[idx]
        if isinstance(label, torch.Tensor):
            label = int(label.item())
        else:
            label = int(label)
            
        if label not in samples_per_digit:
            samples_per_digit[label] = img
            
        if len(samples_per_digit) == 10:
            break
            
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.8))
    axes_flat = axes.flatten()
    
    for digit in range(10):
        ax = axes_flat[digit]
        img = samples_per_digit[digit]
        img_np = img.squeeze().cpu().numpy()
        
        ax.imshow(img_np, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Class {digit}", fontsize=11, fontweight="bold", pad=8)
        ax.axis("off")
        
    plt.tight_layout(h_pad=2.0)
    
    out_png = output_dir / "mnist_samples_2x5.png"
    out_pdf = output_dir / "mnist_samples_2x5.pdf"
    plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved 2x5 MNIST sample grid to:\n  - {out_png}\n  - {out_pdf}")


def plot_mnist_canary_examples(
    dataset,
    output_dir: Path,
    base_digit: int = 7,
    square_size: int = 5,
    seed: int = 42,
    dpi: int = 300,
):
    """
    Generates 1 example image for each canary type on MNIST:
      1. Clean / Original (for comparison)
      2. Square Watermark
      3. Label Noise
      4. Gaussian Noise
      5. Uniform Noise
      6. OOD Natural (FashionMNIST)

    Saves:
      - A combined 2x3 grid overview image
      - Individual image files for each canary
    """
    torch.manual_seed(seed)
    
    # Find a representative base sample matching base_digit
    base_img = None
    true_label = None
    for idx in range(len(dataset)):
        img, label = dataset[idx]
        lbl = int(label.item()) if isinstance(label, torch.Tensor) else int(label)
        if lbl == base_digit:
            base_img = img
            true_label = lbl
            break
            
    if base_img is None:
        base_img, label = dataset[0]
        true_label = int(label.item()) if isinstance(label, torch.Tensor) else int(label)

    dim = (1, 28, 28)
    
    # Instantiate canaries
    watermark_canary = SquareWatermarkCanary(dim=dim, square_size=square_size)
    label_noise_canary = LabelNoiseCanary(dim=dim)
    gaussian_canary = GaussianNoiseCanary(dim=dim)
    uniform_canary = UniformNoiseCanary(dim=dim)
    ood_canary = OODNaturalCanary(dim=dim)

    # Generate canary images
    img_clean = base_img.clone()
    img_watermark = watermark_canary(base_img.clone())
    img_label_noise = label_noise_canary(base_img.clone())
    deranged_label = (true_label + 1) % 10
    img_gaussian = gaussian_canary(base_img.clone())
    img_uniform = uniform_canary(base_img.clone())
    img_ood = ood_canary(base_img.clone())

    canary_items = [
        {
            "name": "Original (Clean)",
            "file_suffix": "clean",
            "img": img_clean,
            "title": f"Clean (Digit {true_label})",
            "subtitle": f"Ground Truth Label: {true_label}",
        },
        {
            "name": "Square Watermark",
            "file_suffix": "square_watermark",
            "img": img_watermark,
            "title": "Square Watermark",
            "subtitle": f"{square_size}x{square_size} px patch in corner",
        },
        {
            "name": "Label Noise",
            "file_suffix": "label_noise",
            "img": img_label_noise,
            "title": "Label Noise",
            "subtitle": f"True: {true_label} -> Flipped to: {deranged_label}",
        },
        {
            "name": "Gaussian Noise",
            "file_suffix": "gaussian_noise",
            "img": img_gaussian,
            "title": "Gaussian Noise",
            "subtitle": "Independent N(0, 1) noise",
        },
        {
            "name": "Uniform Noise",
            "file_suffix": "uniform_noise",
            "img": img_uniform,
            "title": "Uniform Noise",
            "subtitle": "Uniform U[0, 1] noise",
        },
        {
            "name": "OOD Natural",
            "file_suffix": "ood_natural",
            "img": img_ood,
            "title": "OOD Natural",
            "subtitle": "FashionMNIST (Sneaker/Clothing)",
        },
    ]

    # 1. Save individual canary images
    single_dir = output_dir / "individual_canaries"
    single_dir.mkdir(parents=True, exist_ok=True)
    
    for item in canary_items:
        fig, ax = plt.subplots(figsize=(3, 3))
        arr = item["img"].squeeze().cpu().numpy()
        # Clip Gaussian noise if needed for clean display
        if item["file_suffix"] == "gaussian_noise":
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{item['title']}\n({item['subtitle']})", fontsize=9, fontweight="bold")
        ax.axis("off")
        plt.tight_layout()
        single_path = single_dir / f"mnist_canary_{item['file_suffix']}.png"
        plt.savefig(single_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved individual canary images to: {single_dir}/")

    # 2. Save combined 2x3 overview
    fig, axes = plt.subplots(2, 3, figsize=(10.0, 7.0))
    axes_flat = axes.flatten()

    for idx, item in enumerate(canary_items):
        ax = axes_flat[idx]
        arr = item["img"].squeeze().cpu().numpy()
        if item["file_suffix"] == "gaussian_noise":
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
            
        ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{item['title']}\n{item['subtitle']}", fontsize=11, fontweight="bold", pad=8)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout(h_pad=2.5, w_pad=1.5)

    overview_png = output_dir / "mnist_canary_types_overview.png"
    overview_pdf = output_dir / "mnist_canary_types_overview.pdf"
    plt.savefig(overview_png, dpi=dpi, bbox_inches="tight")
    plt.savefig(overview_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined canary overview to:\n  - {overview_png}\n  - {overview_pdf}")


def main():
    parser = argparse.ArgumentParser(description="Generate 2x5 MNIST sample grid and 1 sample per canary type")
    parser.add_argument("--output_dir", "-o", type=str, default="plots/mnist_examples", help="Directory to save output plots")
    parser.add_argument("--base_digit", type=int, default=7, help="Digit to use as base sample for canaries (0-9)")
    parser.add_argument("--square_size", type=int, default=5, help="Size of square watermark in pixels")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saved PNG images")
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("Loading MNIST dataset...")
    data_container = MNISTConfig()()
    dataset = data_container.train

    print("\n--- Generating 2x5 MNIST Sample Grid ---")
    plot_mnist_grid_2x5(dataset, out_path, dpi=args.dpi)

    print("\n--- Generating MNIST Canary Type Examples ---")
    plot_mnist_canary_examples(
        dataset,
        out_path,
        base_digit=args.base_digit,
        square_size=args.square_size,
        seed=args.seed,
        dpi=args.dpi,
    )
    print("\nAll done!")


if __name__ == "__main__":
    main()
