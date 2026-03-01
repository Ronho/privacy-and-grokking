"""t-SNE visualisation of penultimate-layer activations."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from sklearn.manifold import TSNE


def plot_tsne(
    train_activations: torch.Tensor,
    test_activations: torch.Tensor,
    *,
    title: str = "t-SNE of Penultimate-Layer Activations",
    perplexity: float = 30.0,
    random_state: int = 42,
    max_samples: int = 5000,
    figsize: tuple[float, float] = (8, 6),
) -> Figure:
    train_np = train_activations.numpy()
    test_np = test_activations.numpy()

    n_train, n_test = len(train_np), len(test_np)
    total = n_train + n_test

    # Sub-sample if necessary
    if total > max_samples:
        ratio = max_samples / total
        idx_tr = np.random.default_rng(random_state).choice(
            n_train,
            size=max(1, int(n_train * ratio)),
            replace=False,
        )
        idx_te = np.random.default_rng(random_state + 1).choice(
            n_test,
            size=max(1, int(n_test * ratio)),
            replace=False,
        )
        train_np = train_np[idx_tr]
        test_np = test_np[idx_te]

    combined = np.concatenate([train_np, test_np], axis=0)
    labels = np.array(
        [1] * len(train_np) + [0] * len(test_np),
    )

    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, max(1.0, len(combined) - 1)),
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    embedded = tsne.fit_transform(combined)

    fig, ax = plt.subplots(figsize=figsize)
    member_mask = labels == 1

    ax.scatter(
        embedded[~member_mask, 0],
        embedded[~member_mask, 1],
        s=12,
        alpha=0.5,
        label="Non-members (test)",
        color="#dc2626",
        edgecolors="none",
    )
    ax.scatter(
        embedded[member_mask, 0],
        embedded[member_mask, 1],
        s=12,
        alpha=0.5,
        label="Members (train)",
        color="#2563eb",
        edgecolors="none",
    )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def plot_tsne_on_ax(
    ax: plt.Axes,
    train_activations: torch.Tensor,
    test_activations: torch.Tensor,
    *,
    title: str = "",
    perplexity: float = 30.0,
    random_state: int = 42,
    max_samples: int = 5000,
) -> None:
    train_np = train_activations.numpy()
    test_np = test_activations.numpy()

    n_train, n_test = len(train_np), len(test_np)
    total = n_train + n_test

    if total > max_samples:
        ratio = max_samples / total
        idx_tr = np.random.default_rng(random_state).choice(
            n_train,
            size=max(1, int(n_train * ratio)),
            replace=False,
        )
        idx_te = np.random.default_rng(random_state + 1).choice(
            n_test,
            size=max(1, int(n_test * ratio)),
            replace=False,
        )
        train_np = train_np[idx_tr]
        test_np = test_np[idx_te]

    combined = np.concatenate([train_np, test_np], axis=0)
    labels = np.array(
        [1] * len(train_np) + [0] * len(test_np),
    )

    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, max(1.0, len(combined) - 1)),
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    embedded = tsne.fit_transform(combined)

    member_mask = labels == 1

    ax.scatter(
        embedded[~member_mask, 0],
        embedded[~member_mask, 1],
        s=8,
        alpha=0.4,
        label="Non-members",
        color="#dc2626",
        edgecolors="none",
    )
    ax.scatter(
        embedded[member_mask, 0],
        embedded[member_mask, 1],
        s=8,
        alpha=0.4,
        label="Members",
        color="#2563eb",
        edgecolors="none",
    )

    if title:
        ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("t-SNE 1", fontsize=8)
    ax.set_ylabel("t-SNE 2", fontsize=8)
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, alpha=0.2)
