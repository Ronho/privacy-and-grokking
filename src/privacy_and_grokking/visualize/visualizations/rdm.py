import numpy as np
import torch
from matplotlib import pyplot as plt

from privacy_and_grokking.utils import Logger
from privacy_and_grokking.visualize.handler import DataHandler
from privacy_and_grokking.visualize.visualizations.shared import handle_missing_data


def _correlation_rdm(x: np.ndarray) -> np.ndarray:
    """Compute pairwise correlation-distance matrix (1 - pearson r) for rows of x."""
    x = x - x.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1e-10, norms)
    x_normed = x / norms
    corr = np.clip(x_normed @ x_normed.T, -1.0, 1.0)
    return 1.0 - corr


def rdm(ax: plt.Axes, dh: DataHandler, layer: str) -> None:
    """Render correlation-distance RDM for a single layer."""
    logger = Logger.get()
    logger.info("Creating RDM plot.", extra={"run_id": dh.run_id, "layer": layer})

    data = dh.load_activation_data()
    if data is None:
        handle_missing_data(ax, dh.run_id, f"RDM ({layer})")
        return

    train_acts: dict[str, torch.Tensor] = data.get("train_activations", {})
    test_acts: dict[str, torch.Tensor] = data.get("test_activations", {})
    train_labels: torch.Tensor = data["train_labels"]
    test_labels: torch.Tensor = data["test_labels"]

    if layer not in train_acts:
        handle_missing_data(ax, dh.run_id, f"RDM ({layer})")
        return

    n_per_class = 5
    classes = sorted(torch.cat([train_labels, test_labels]).unique().tolist())
    rng = np.random.default_rng(seed=0)

    def _sample_idx(labels: torch.Tensor, cls: int, n: int) -> np.ndarray:
        idx = (labels == cls).nonzero(as_tuple=False).squeeze(1).numpy()
        return rng.choice(idx, size=n, replace=False) if len(idx) >= n else idx

    tr_idx_per_class = [_sample_idx(train_labels, c, n_per_class) for c in classes]
    te_idx_per_class = [_sample_idx(test_labels, c, n_per_class) for c in classes]
    tr_order = np.concatenate(tr_idx_per_class)
    te_order = np.concatenate(te_idx_per_class)
    n_train_total = len(tr_order)

    tr_layer = train_acts[layer].float().numpy()
    te_layer = test_acts[layer].float().numpy()
    if tr_layer.shape[0] != len(train_labels):
        tr_layer = tr_layer[: len(train_labels)]
    if te_layer.shape[0] != len(test_labels):
        te_layer = te_layer[: len(test_labels)]

    acts = np.concatenate([tr_layer[tr_order], te_layer[te_order]], axis=0)
    rdm_matrix = _correlation_rdm(acts)
    im = ax.imshow(
        rdm_matrix, aspect="auto", cmap="RdBu_r", vmin=0.0, vmax=1.0, interpolation="none"
    )

    tick_positions: list[float] = []
    tick_names: list[str] = []
    cursor = 0
    for i, c in enumerate(classes):
        n_tr = len(tr_idx_per_class[i])
        tick_positions.append(cursor + n_tr / 2 - 0.5)
        tick_names.append(f"tr{int(c)}")
        cursor += n_tr
    for i, c in enumerate(classes):
        n_te = len(te_idx_per_class[i])
        tick_positions.append(cursor + n_te / 2 - 0.5)
        tick_names.append(f"te{int(c)}")
        cursor += n_te

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_names, rotation=90, fontsize=5)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_names, fontsize=5)

    cursor = 0
    for idx_list in tr_idx_per_class + te_idx_per_class:
        cursor += len(idx_list)
        pos = cursor - 0.5
        ax.axhline(pos, color="white", linewidth=0.4, alpha=0.7)
        ax.axvline(pos, color="white", linewidth=0.4, alpha=0.7)
    ax.axhline(n_train_total - 0.5, color="black", linewidth=1.2)
    ax.axvline(n_train_total - 0.5, color="black", linewidth=1.2)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=5)

    logger.info("Created RDM plot.", extra={"run_id": dh.run_id, "layer": layer})
