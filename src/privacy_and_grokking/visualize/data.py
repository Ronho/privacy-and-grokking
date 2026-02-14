import matplotlib.pyplot as plt
import rsatoolbox
import torch
from matplotlib.patches import Rectangle

from privacy_and_grokking.config import DatasetConfig
from privacy_and_grokking.datasets import (
    CanarySubset,
    Datasets,
    Maskings,
    Normalization,
    SquareWatermarkCanaryConfig,
    UniformNoiseCanaryConfig,
    generate_datasets,
)
from privacy_and_grokking.logger import get_logger
from privacy_and_grokking.path_keeper import get_path_keeper
from privacy_and_grokking.visualize.masks import vis_masking_strategy

logger = get_logger()


def _samples_by_class(dataset: CanarySubset, canary: bool = False):
    samples_by_class = {i: [] for i in range(dataset.num_classes)}
    indices = dataset.canary_indices if canary else dataset.subset_indices
    for idx in indices:
        img, label = dataset[idx]
        label = int(label.item())
        if len(samples_by_class[label]) < 5:
            samples_by_class[label].append(img)
        if all(len(samples) >= 5 for samples in samples_by_class.values()):
            break
    return samples_by_class


def _vis_samples_per_class(
    name: str,
    samples_by_class: dict[int, list[torch.Tensor]],
    num_classes: int,
    norm: Normalization,
    postfix="",
):
    pk = get_path_keeper()
    fig, axes = plt.subplots(num_classes, 6, figsize=(12, 2 * num_classes))
    for class_idx in range(num_classes):
        axes[class_idx, 0].text(
            0.5,
            0.5,
            f"Class {class_idx}",
            ha="right",
            va="center",
            fontsize=10,
            fontweight="bold",
            rotation=0,
        )
        axes[class_idx, 0].axis("off")

        for sample_idx in range(5):
            ax = axes[class_idx, sample_idx + 1]
            img = samples_by_class[class_idx][sample_idx]

            mean = torch.tensor(norm.mean).view(-1, 1, 1)
            std = torch.tensor(norm.std).view(-1, 1, 1)
            img_denorm = img * std + mean
            img_np = img_denorm.cpu().numpy()

            if img_np.shape[0] == 1:
                ax.imshow(img_np.squeeze(), cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(img_np.transpose(1, 2, 0).clip(0, 1))

            ax.axis("off")
    fig.text(0.02, 0.01, f"Dataset: {name.upper()}", fontsize=9, alpha=0.6)
    fig.tight_layout()
    fig.savefig(pk.IMAGE_FOLDER / f"samples_per_class{postfix}.png")
    plt.close(fig)


def _vis_rdm(name: str, image_ranges: dict[int, list[torch.Tensor]], postfix=""):
    pk = get_path_keeper()

    images = [img for imgs in image_ranges.values() for img in imgs]
    data_array = torch.stack(images).flatten(start_dim=1).cpu().numpy()
    dataset = rsatoolbox.data.Dataset(data_array)
    rdm = rsatoolbox.rdm.calc_rdm(dataset, method="correlation")
    rdm_matrix = rdm.get_matrices()[0]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(rdm_matrix, cmap="hot", interpolation="nearest")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Dissimilarity", rotation=90, labelpad=20)

    n_samples = rdm_matrix.shape[0]
    start = 0
    for idx, (label, images) in enumerate(image_ranges.items()):
        end = start + len(images) - 1
        mid = (start + end) / 2
        width = end - start + 1
        grey_shade = "lightgrey" if idx % 2 == 0 else "darkgrey"
        alpha = 0.3 if idx % 2 == 0 else 0.4
        rect_x = Rectangle(
            (start - 0.5, n_samples),
            width,
            2.5,
            facecolor=grey_shade,
            alpha=alpha,
            clip_on=False,
            transform=ax.transData,
            zorder=0,
        )
        ax.add_patch(rect_x)
        rect_y = Rectangle(
            (-3, start - 0.5),
            2.5,
            width,
            facecolor=grey_shade,
            alpha=alpha,
            clip_on=False,
            transform=ax.transData,
            zorder=0,
        )
        ax.add_patch(rect_y)
        ax.text(
            mid,
            n_samples + 1.25,
            label,
            rotation=0,
            va="center",
            ha="center",
            fontsize=9,
            fontweight="bold",
            clip_on=False,
        )
        ax.text(
            -1.5,
            mid,
            label,
            rotation=90,
            va="center",
            ha="center",
            fontsize=9,
            fontweight="bold",
            clip_on=False,
        )
        start = end + 1

    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Sample Index")
    ax.set_title("Representational Dissimilarity Matrix")

    fig.text(0.02, 0.01, f"Dataset: {name.upper()}", fontsize=9, alpha=0.6)
    fig.tight_layout()
    fig.savefig(pk.IMAGE_FOLDER / f"rdm{postfix}.png")
    plt.close(fig)


def vis_dataset(name: Datasets):
    logger.info("Visualizing dataset", extra={"dataset": name})
    pk = get_path_keeper()
    pk.set_params({"model": f"DATASET_{name.upper()}"})

    config = DatasetConfig(name=name, train_size=None, canary_share=0, canary_config=None, seed=1)
    train, test = generate_datasets(config=config)

    # Class Distributions
    def get_counts(dataset):
        return torch.tensor([y for _, y in dataset]).unique(return_counts=True)

    train_labels, train_counts = get_counts(train)
    test_labels, test_counts = get_counts(test)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.bar(train_labels, train_counts)
    ax1.set_xlabel("Class")
    ax1.set_ylabel("Number of Samples")
    ax1.set_title("Train Set Class Distribution")
    ax1.set_xticks(train_labels)
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(test_labels, test_counts)
    ax2.set_xlabel("Class")
    ax2.set_ylabel("Number of Samples")
    ax2.set_title("Test Set Class Distribution")
    ax2.set_xticks(test_labels)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.text(0.02, 0.01, f"Dataset: {name.upper()}", fontsize=9, alpha=0.6)
    fig.tight_layout()
    fig.savefig(pk.IMAGE_FOLDER / "class_distributions.png")
    plt.close(fig)

    # Samples per Class
    samples_by_class_train = _samples_by_class(dataset=train)
    samples_by_class_test = _samples_by_class(dataset=test)
    _vis_samples_per_class(name, samples_by_class_train, train.num_classes, train.norm)

    # Watermark Samples per Class
    config_watermark = DatasetConfig(
        name=name,
        train_size=None,
        canary_share=0.01,
        canary_config=SquareWatermarkCanaryConfig(square_size=3),
        seed=2,
    )
    train_watermark, _ = generate_datasets(config=config_watermark)
    samples_by_class_watermark = _samples_by_class(dataset=train_watermark, canary=True)
    _vis_samples_per_class(
        name,
        samples_by_class_watermark,
        train_watermark.num_classes,
        train_watermark.norm,
        postfix="_watermark",
    )

    # Noise Samples per Class
    config_noise = DatasetConfig(
        name=name,
        train_size=None,
        canary_share=0.01,
        canary_config=UniformNoiseCanaryConfig(),
        seed=2,
    )
    train_noise, _ = generate_datasets(config=config_noise)
    samples_by_class_noise = _samples_by_class(dataset=train_noise, canary=True)
    _vis_samples_per_class(
        name,
        samples_by_class_noise,
        train_noise.num_classes,
        train_noise.norm,
        postfix="_gaussian_noise",
    )

    # RDM of Images
    def sbc_to_images(sbc):
        return [img for class_samples in sbc.values() for img in class_samples]

    train_images = sbc_to_images(samples_by_class_train)
    test_images = sbc_to_images(samples_by_class_test)
    watermark_images = sbc_to_images(samples_by_class_watermark)
    noise_images = sbc_to_images(samples_by_class_noise)
    images = {
        "Train": train_images,
        "Test": test_images,
        "Watermark": watermark_images,
        "Gaussian Noise": noise_images,
    }
    _vis_rdm(name, images, postfix="_images")

    # RDM of Indices
    def sbc_to_classes(sbc):
        return torch.nn.functional.one_hot(
            torch.tensor(
                [class_idx for class_idx, class_samples in sbc.items() for _ in class_samples]
            ),
            train.num_classes,
        )

    train_classes = sbc_to_classes(samples_by_class_train)
    test_classes = sbc_to_classes(samples_by_class_test)
    watermark_classes = sbc_to_classes(samples_by_class_watermark)
    noise_classes = sbc_to_classes(samples_by_class_noise)
    images = {
        "Train": train_classes,
        "Test": test_classes,
        "Watermark": watermark_classes,
        "Gaussian Noise": noise_classes,
    }
    _vis_rdm(name, images, postfix="_indices")

    # Masks
    config.train_size = 2000
    train_mask, _ = generate_datasets(config=config)
    num_models_list = [2, 8, 64, 256, 512]
    for masking_type in Maskings:
        vis_masking_strategy(train_mask, masking_type, num_models_list=num_models_list)


def visualize():
    for dataset_name in list(Datasets):
        vis_dataset(dataset_name)
