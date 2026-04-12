import mlflow
import numpy as np
import torch
from matplotlib import pyplot as plt

from privacy_and_grokking.config import TrainConfig
from privacy_and_grokking.datasets import create_masking, generate_datasets, mask_dataset
from privacy_and_grokking.utils import Logger
from privacy_and_grokking.visualize.handler import DataHandler
from privacy_and_grokking.visualize.visualizations.shared import handle_missing_data


def class_distribution(ax: plt.Axes, dh: DataHandler):
    logger = Logger.get()
    logger.info("Creating class distribution plot.", extra={"run_id": dh.run_id})

    try:
        cfg = TrainConfig.model_validate(
            mlflow.artifacts.load_dict(f"runs:/{dh.run_id}/training_config.json")
        )
    except Exception as exc:
        logger.warning(
            "Could not load training config for class distribution.",
            extra={"run_id": dh.run_id, "error": str(exc)},
        )
        handle_missing_data(ax, dh.run_id, "class distribution")
        return

    train_ds, test_ds = generate_datasets(cfg.dataset)
    masking = create_masking(
        config=cfg.dataset_mask,
        num_samples=len(train_ds),
        num_classes=train_ds.num_classes,
    )
    train_subset = mask_dataset(masking, train_ds, cfg.dataset_mask_idx)

    train_labels = torch.tensor([y for _, y in train_subset])
    test_labels = torch.tensor([y for _, y in test_ds])

    classes = sorted(torch.cat([train_labels, test_labels]).unique().tolist())
    num_classes = len(classes)

    train_counts = np.array([(train_labels == c).sum().item() for c in classes])
    test_counts = np.array([(test_labels == c).sum().item() for c in classes])

    x = np.arange(num_classes)
    width = 0.4

    ax.bar(x - width / 2, train_counts, width, label="Train", color="tab:blue", alpha=0.8)
    ax.bar(x + width / 2, test_counts, width, label="Test", color="tab:red", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(c)) for c in classes], rotation=45, ha="right")
    ax.set_xlabel("Class")
    ax.set_ylabel("Sample Count")
    ax.legend(loc="best")

    logger.info("Created class distribution plot.", extra={"run_id": dh.run_id})
