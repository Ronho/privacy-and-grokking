import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE

from privacy_and_grokking.utils import Logger
from privacy_and_grokking.visualize.handler import DataHandler
from privacy_and_grokking.visualize.visualizations.shared import handle_missing_data


def _tsne_on_ax(
    ax: plt.Axes,
    train_acts: np.ndarray,
    test_acts: np.ndarray,
    train_lbl: np.ndarray,
    test_lbl: np.ndarray,
    *,
    perplexity: float = 30.0,
    random_state: int = 42,
    max_samples: int = 5000,
) -> None:
    n_train, n_test = len(train_acts), len(test_acts)
    total = n_train + n_test
    if total > max_samples:
        ratio = max_samples / total
        rng = np.random.default_rng(random_state)
        idx_tr = rng.choice(n_train, size=max(1, int(n_train * ratio)), replace=False)
        idx_te = rng.choice(n_test, size=max(1, int(n_test * ratio)), replace=False)
        train_acts, train_lbl = train_acts[idx_tr], train_lbl[idx_tr]
        test_acts, test_lbl = test_acts[idx_te], test_lbl[idx_te]

    combined = np.concatenate([train_acts, test_acts], axis=0)
    all_labels = np.concatenate([train_lbl, test_lbl], axis=0)
    is_train = np.array([True] * len(train_acts) + [False] * len(test_acts))

    n_pts = len(combined)
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, max(1.0, n_pts - 1)),
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    embedded = tsne.fit_transform(combined)

    classes = np.unique(all_labels)
    cmap = plt.get_cmap("tab10") if len(classes) <= 10 else plt.get_cmap("tab20")

    for cls in classes:
        cls_mask = all_labels == cls
        color = cmap(int(cls) % cmap.N)
        tr_mask = cls_mask & is_train
        te_mask = cls_mask & ~is_train
        if tr_mask.any():
            ax.scatter(
                embedded[tr_mask, 0],
                embedded[tr_mask, 1],
                s=12,
                alpha=0.65,
                color=color,
                marker="o",
                edgecolors="none",
            )
        if te_mask.any():
            ax.scatter(
                embedded[te_mask, 0],
                embedded[te_mask, 1],
                s=14,
                alpha=0.45,
                color=color,
                marker="^",
                edgecolors="none",
            )

    class_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=cmap(int(c) % cmap.N),
            markersize=6,
            label=f"Class {int(c)}",
        )
        for c in classes
    ]
    membership_handles = [
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor="gray", markersize=6, label="Train"
        ),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="gray", markersize=6, label="Test"),
    ]
    ax.legend(
        handles=class_handles + membership_handles,
        loc="best",
        fontsize=6,
        ncol=max(1, (len(classes) + 1) // 2),
    )
    ax.set_xlabel("t-SNE 1", fontsize=7)
    ax.set_ylabel("t-SNE 2", fontsize=7)
    ax.grid(True, alpha=0.2)


def tsne(ax: plt.Axes, dh: DataHandler, layer: str) -> None:
    """Run t-SNE on one layer's activations and plot on ax."""
    logger = Logger.get()
    logger.info("Creating t-SNE plot.", extra={"run_id": dh.run_id, "layer": layer})

    data = dh.load_activation_data()
    if data is None:
        handle_missing_data(ax, dh.run_id, f"t-SNE ({layer})")
        return

    train_layer_acts: dict[str, torch.Tensor] | None = data.get("train_layer_activations")
    test_layer_acts: dict[str, torch.Tensor] | None = data.get("test_layer_activations")
    train_labels: torch.Tensor = data["train_labels"]
    test_labels: torch.Tensor = data["test_labels"]

    if not train_layer_acts:
        train_acts_raw = data.get("train_activations")
        test_acts_raw = data.get("test_activations")
        if train_acts_raw is None:
            handle_missing_data(ax, dh.run_id, f"t-SNE ({layer})")
            return
        if isinstance(train_acts_raw, dict):
            train_layer_acts = train_acts_raw
            test_layer_acts = test_acts_raw if isinstance(test_acts_raw, dict) else {}
        else:
            train_layer_acts = {"activations": train_acts_raw}
            test_layer_acts = {"activations": test_acts_raw}

    if layer not in train_layer_acts:
        handle_missing_data(ax, dh.run_id, f"t-SNE ({layer})")
        return

    tr_acts = train_layer_acts[layer].float().numpy()
    te_acts = test_layer_acts[layer].float().numpy()
    train_lbl_np = train_labels.numpy()
    test_lbl_np = test_labels.numpy()

    if tr_acts.shape[0] != len(train_lbl_np):
        tr_acts = tr_acts[: len(train_lbl_np)]
    if te_acts.shape[0] != len(test_lbl_np):
        te_acts = te_acts[: len(test_lbl_np)]

    _tsne_on_ax(ax, tr_acts, te_acts, train_lbl_np, test_lbl_np)

    logger.info("Created t-SNE plot.", extra={"run_id": dh.run_id, "layer": layer})
