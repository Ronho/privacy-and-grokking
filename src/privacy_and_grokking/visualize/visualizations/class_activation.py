import numpy as np
import torch
from matplotlib import pyplot as plt

from privacy_and_grokking.utils import Logger
from privacy_and_grokking.visualize.handler import DataHandler
from privacy_and_grokking.visualize.visualizations.shared import handle_missing_data


def class_activation(ax: plt.Axes, dh: DataHandler, layer: str) -> None:
    """Plot per-class mean±std activation profiles for one layer."""
    logger = Logger.get()
    logger.info(
        "Creating class activation plot.", extra={"run_id": dh.run_id, "layer": layer}
    )

    data = dh.load_activation_data()
    if data is None:
        handle_missing_data(ax, dh.run_id, f"class activation ({layer})")
        return

    train_acts: dict[str, torch.Tensor] = data.get("train_activations", {})
    test_acts: dict[str, torch.Tensor] = data.get("test_activations", {})
    train_labels: torch.Tensor = data["train_labels"]
    test_labels: torch.Tensor = data["test_labels"]

    if layer not in train_acts:
        handle_missing_data(ax, dh.run_id, f"class activation ({layer})")
        return

    tr_layer = train_acts[layer].float()
    te_layer = test_acts[layer].float()

    if tr_layer.shape[0] != len(train_labels):
        tr_layer = tr_layer[: len(train_labels)]
    if te_layer.shape[0] != len(test_labels):
        te_layer = te_layer[: len(test_labels)]

    classes = sorted(torch.cat([train_labels, test_labels]).unique().tolist())
    neuron_idx = np.arange(tr_layer.shape[1])
    cmap = plt.get_cmap("tab10") if len(classes) <= 10 else plt.get_cmap("tab20")

    for cls in classes:
        color = cmap(int(cls) % cmap.N)
        tr_cls = tr_layer[train_labels == cls]
        te_cls = te_layer[test_labels == cls]

        if tr_cls.shape[0] > 0:
            tr_mean = tr_cls.mean(dim=0).numpy()
            tr_std = tr_cls.std(dim=0).numpy()
            ax.plot(
                neuron_idx,
                tr_mean,
                color=color,
                linewidth=0.9,
                linestyle="-",
                label=f"Tr {int(cls)}",
            )
            ax.fill_between(
                neuron_idx, tr_mean - tr_std, tr_mean + tr_std, color=color, alpha=0.1
            )

        if te_cls.shape[0] > 0:
            te_mean = te_cls.mean(dim=0).numpy()
            te_std = te_cls.std(dim=0).numpy()
            ax.plot(
                neuron_idx,
                te_mean,
                color=color,
                linewidth=0.9,
                linestyle="--",
                label=f"Te {int(cls)}",
            )
            ax.fill_between(
                neuron_idx, te_mean - te_std, te_mean + te_std, color=color, alpha=0.05
            )

    ax.set_xlabel("Neuron index", fontsize=7)
    ax.set_ylabel("Activation", fontsize=7)
    ax.tick_params(labelsize=5)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best", fontsize=5, ncol=max(1, len(classes) // 2))

    logger.info(
        "Created class activation plot.", extra={"run_id": dh.run_id, "layer": layer}
    )
