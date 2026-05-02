import matplotlib.pyplot as plt
import numpy as np
import torch

from privacy_and_grokking.datasets.base import CanaryDataset
from privacy_and_grokking.datasets.masking.base import MaskingConfig
from privacy_and_grokking.logger import get_logger
from privacy_and_grokking.path_keeper import get_path_keeper

logger = get_logger()


def _visualize_datapoints_per_model(
    ax, mask: torch.Tensor, labels: torch.Tensor, num_classes: int, set_ylabel: bool = True
):
    num_models = mask.shape[1]

    model_class_counts = np.zeros((num_models, num_classes))
    for model_idx in range(num_models):
        model_mask = mask[:, model_idx]
        model_labels = labels[model_mask]
        for class_idx in range(num_classes):
            model_class_counts[model_idx, class_idx] = (model_labels == class_idx).sum().item()

    total_per_model = model_class_counts.sum(axis=1)
    sorted_indices = np.argsort(total_per_model)[::-1]  # High to low
    model_class_counts_sorted = model_class_counts[sorted_indices]

    x = np.arange(num_models)
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))

    bottom = np.zeros(num_models)
    for class_idx in range(num_classes):
        ax.bar(
            x,
            model_class_counts_sorted[:, class_idx],
            bottom=bottom,
            label=f"Class {class_idx}",
            color=colors[class_idx],
            width=0.8,
        )
        bottom += model_class_counts_sorted[:, class_idx]

    ax.set_xlabel("Model (sorted by total samples)", fontsize=8)
    if set_ylabel:
        ax.set_ylabel("Number of Samples", fontsize=8)
    ax.tick_params(axis="both", which="major", labelsize=7)

    ax.legend(fontsize=6, loc="upper right", ncol=2)

    if num_models <= 64:
        ax.set_xticks(x[:: max(1, num_models // 10)])
        ax.set_xticklabels(x[:: max(1, num_models // 10)])
    else:
        tick_positions = [0, num_models // 4, num_models // 2, 3 * num_models // 4, num_models - 1]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_positions)

    ax.grid(True, alpha=0.3, axis="y")


def _visualize_models_per_datapoint(ax, mask: torch.Tensor, set_ylabel: bool = True):
    num_samples = mask.shape[0]
    models_per_sample = mask.sum(dim=1).cpu().numpy()
    sorted_counts = np.sort(models_per_sample)
    x = np.arange(num_samples)

    ax.plot(x, sorted_counts, linewidth=0.5, alpha=0.7)
    ax.set_xlabel("Data Entry (sorted)", fontsize=8)
    if set_ylabel:
        ax.set_ylabel("Number of Models", fontsize=8)
    ax.tick_params(axis="both", which="major", labelsize=7)

    mean_count = models_per_sample.mean()
    ax.axhline(
        mean_count,
        color="red",
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label=f"Mean: {mean_count:.2f}",
    )
    ax.legend(fontsize=6, loc="upper left")

    if num_samples <= 1000:
        tick_step = max(1, num_samples // 5)
        tick_positions = np.arange(0, num_samples, tick_step)
    else:
        tick_positions = [
            0,
            num_samples // 4,
            num_samples // 2,
            3 * num_samples // 4,
            num_samples - 1,
        ]

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_positions)

    ax.grid(True, alpha=0.3, axis="y")


def vis_masking_strategy(
    dataset: CanaryDataset,
    masking_type: str,
    num_models_list: list[int],
    overwrite: bool = False,
):
    pk = get_path_keeper()
    filename = f"masking_{masking_type}.png"
    output_path = pk.IMAGE_FOLDER / filename

    if output_path.exists() and not overwrite:
        return

    labels = torch.Tensor([label for _, label in dataset])
    num_cols = len(num_models_list)
    fig, axes = plt.subplots(3, num_cols, figsize=(4 * num_cols, 12))
    if num_cols == 1:
        axes = axes.reshape(3, 1)

    for col_idx, num_models in enumerate(num_models_list):
        config = MaskingConfig(name=masking_type, num_models=num_models, p=0.5, seed=1)
        masking = config(num_samples=len(dataset), num_classes=dataset.num_classes)
        mask = masking(labels)
        axes[0, col_idx].set_title(f"{num_models} Models", fontsize=9, fontweight="bold")

        _visualize_datapoints_per_model(axes[0, col_idx], mask, labels, dataset.num_classes)

        _visualize_models_per_datapoint(axes[1, col_idx], mask, set_ylabel=(col_idx == 0))

        im = axes[2, col_idx].imshow(mask.cpu().numpy(), aspect="auto", cmap="viridis")
        axes[2, col_idx].set_xlabel("Model Index", fontsize=8)
        if col_idx == 0:
            axes[2, col_idx].set_ylabel("Datapoint Index", fontsize=8)
        fig.colorbar(im, ax=axes[2, col_idx], orientation="vertical", fraction=0.05, pad=0.02)

    fig.suptitle(
        f"{masking_type.upper().replace('_', ' ')}", fontsize=14, fontweight="bold", y=0.995
    )

    fig.text(
        0.02,
        0.83,
        "Samples per Model\n(by class)",
        fontsize=10,
        fontweight="bold",
        rotation=90,
        va="center",
        ha="center",
    )
    fig.text(
        0.02,
        0.5,
        "Models per\nData Entry",
        fontsize=10,
        fontweight="bold",
        rotation=90,
        va="center",
        ha="center",
    )
    fig.text(
        0.02,
        0.17,
        "Membership Heatmap",
        fontsize=10,
        fontweight="bold",
        rotation=90,
        va="center",
        ha="center",
    )

    plt.tight_layout(rect=[0.03, 0.01, 1, 0.99])

    pk = get_path_keeper()
    filename = f"masking_{masking_type}.png"
    fig.savefig(pk.IMAGE_FOLDER / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
