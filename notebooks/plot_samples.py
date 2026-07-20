import argparse
import matplotlib.pyplot as plt
import torch

from privacy_and_grokking.datasets.sets.cifar10 import CIFAR10Config
from privacy_and_grokking.datasets.sets.mnist import MNISTConfig

def get_dataset_config(name: str):
    if name.lower() == "cifar10":
        return CIFAR10Config()
    elif name.lower() == "mnist":
        return MNISTConfig()
    else:
        raise ValueError(f"Unknown dataset: {name}")

def main():
    parser = argparse.ArgumentParser(description="Plot 10 samples from each class")
    parser.add_argument("--start-index", type=int, default=0, help="Index to start searching for samples")
    parser.add_argument("--split", type=str, choices=["train", "test"], default="train", help="Which split to use")
    parser.add_argument("--dataset", type=str, choices=["cifar10", "mnist"], default="cifar10", help="Dataset name")
    parser.add_argument("--output", type=str, default="samples.png", help="Output image file")
    args = parser.parse_args()

    config = get_dataset_config(args.dataset)
    data_container = config()
    
    dataset = data_container.train if args.split == "train" else data_container.test
    num_classes = data_container.num_classes

    samples_per_class = 10
    collected_samples = {i: [] for i in range(num_classes)}
    
    print(f"Collecting {samples_per_class} samples for each of {num_classes} classes from {args.dataset} {args.split} split starting at index {args.start_index}...")

    # Collect samples
    idx = args.start_index
    while True:
        if idx >= len(dataset):
            print(f"Warning: Reached end of dataset at index {idx}. Not all classes may have 10 samples.")
            break
            
        img, label = dataset[idx]
        if isinstance(label, torch.Tensor):
            label = label.item()
            
        if len(collected_samples[label]) < samples_per_class:
            if isinstance(img, torch.Tensor):
                # Images are usually [C, H, W], transpose to [H, W, C] for plotting
                img = img.permute(1, 2, 0).numpy()
            collected_samples[label].append((img, idx))
            
        # Check if we have enough samples for all classes
        if all(len(samples) == samples_per_class for samples in collected_samples.values()):
            break
            
        idx += 1

    # Plot
    fig, axes = plt.subplots(num_classes, samples_per_class, figsize=(samples_per_class * 1.5, num_classes * 1.5))
    
    for class_idx in range(num_classes):
        samples = collected_samples[class_idx]
        for col_idx in range(samples_per_class):
            ax = axes[class_idx, col_idx]
            if col_idx < len(samples):
                img, orig_idx = samples[col_idx]
                if img.shape[-1] == 1:
                    ax.imshow(img.squeeze(), cmap="gray")
                else:
                    ax.imshow(img)
                ax.set_xlabel(f"idx: {orig_idx}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            
            if col_idx == 0:
                ax.set_ylabel(f"Class {class_idx}", fontsize=12)

    plt.tight_layout()
    plt.savefig(args.output, bbox_inches="tight")
    print(f"Saved plot to {args.output}")

if __name__ == "__main__":
    main()
