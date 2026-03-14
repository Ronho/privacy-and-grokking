from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
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


def make_tsne_video(
    step_activations: dict[int, dict[str, torch.Tensor]],
    out_path: Path,
    *,
    fps: int = 2,
    perplexity: float = 30.0,
    random_state: int = 42,
    max_samples: int = 2000,
    figsize: tuple[float, float] = (8, 6),
    title_prefix: str = "t-SNE",
) -> None:
    import av
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    steps = sorted(step_activations.keys())
    if not steps:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=figsize, dpi=120)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    canvas.draw()
    w, h = canvas.get_width_height()
    # h264 requires even dimensions
    w = w if w % 2 == 0 else w - 1
    h = h if h % 2 == 0 else h - 1

    container = av.open(str(out_path), mode="w")
    stream = container.add_stream("h264", rate=fps)
    stream.width = w
    stream.height = h
    stream.pix_fmt = "yuv420p"

    for step in steps:
        data = step_activations[step]
        ax.clear()
        plot_tsne_classes_on_ax(
            ax,
            data["train_activations"],
            data["test_activations"],
            data["train_labels"],
            data["test_labels"],
            title=f"{title_prefix} – Step {step}",
            perplexity=perplexity,
            random_state=random_state,
            max_samples=max_samples,
        )
        canvas.draw()
        rgba = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        rgb = np.ascontiguousarray(rgba[:h, :w, :3])  # crop to even dims, drop alpha
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    # flush encoder
    for packet in stream.encode():
        container.mux(packet)

    container.close()
    plt.close(fig)


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


def plot_tsne_classes_on_ax(
    ax: plt.Axes,
    train_activations: torch.Tensor,
    test_activations: torch.Tensor,
    train_labels: torch.Tensor,
    test_labels: torch.Tensor,
    *,
    title: str = "t-SNE (Classes)",
    perplexity: float = 30.0,
    random_state: int = 42,
    max_samples: int = 5000,
) -> None:
    train_np = train_activations.numpy()
    test_np = test_activations.numpy()
    train_lbl = train_labels.numpy()
    test_lbl = test_labels.numpy()

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
        train_np, train_lbl = train_np[idx_tr], train_lbl[idx_tr]
        test_np, test_lbl = test_np[idx_te], test_lbl[idx_te]

    combined = np.concatenate([train_np, test_np], axis=0)
    all_labels = np.concatenate([train_lbl, test_lbl], axis=0)
    is_member = np.array([True] * len(train_np) + [False] * len(test_np))

    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, max(1.0, len(combined) - 1)),
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

        member_cls = cls_mask & is_member
        nonmember_cls = cls_mask & ~is_member

        if member_cls.any():
            ax.scatter(
                embedded[member_cls, 0],
                embedded[member_cls, 1],
                s=12,
                alpha=0.6,
                color=color,
                marker="o",
                edgecolors="none",
            )
        if nonmember_cls.any():
            ax.scatter(
                embedded[nonmember_cls, 0],
                embedded[nonmember_cls, 1],
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
            markersize=7,
            label=f"Class {c}",
        )
        for c in classes
    ]
    membership_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markersize=7,
            label="Members (train)",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="gray",
            markersize=7,
            label="Non-members (test)",
        ),
    ]
    n_cols = max(1, (len(classes) + 1) // 2)
    ax.legend(
        handles=class_handles + membership_handles,
        loc="best",
        fontsize=7,
        ncol=n_cols,
    )

    if title:
        ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("t-SNE 1", fontsize=8)
    ax.set_ylabel("t-SNE 2", fontsize=8)
    ax.grid(True, alpha=0.2)


def plot_tsne_classes(
    train_activations: torch.Tensor,
    test_activations: torch.Tensor,
    train_labels: torch.Tensor,
    test_labels: torch.Tensor,
    *,
    title: str = "t-SNE of Penultimate-Layer Activations – Coloured by Class",
    perplexity: float = 30.0,
    random_state: int = 42,
    max_samples: int = 5000,
    figsize: tuple[float, float] = (8, 6),
) -> Figure:
    train_np = train_activations.numpy()
    test_np = test_activations.numpy()
    train_lbl = train_labels.numpy()
    test_lbl = test_labels.numpy()

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
        train_np, train_lbl = train_np[idx_tr], train_lbl[idx_tr]
        test_np, test_lbl = test_np[idx_te], test_lbl[idx_te]

    combined = np.concatenate([train_np, test_np], axis=0)
    all_labels = np.concatenate([train_lbl, test_lbl], axis=0)
    is_member = np.array([True] * len(train_np) + [False] * len(test_np))

    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, max(1.0, len(combined) - 1)),
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    embedded = tsne.fit_transform(combined)

    classes = np.unique(all_labels)
    cmap = plt.get_cmap("tab10") if len(classes) <= 10 else plt.get_cmap("tab20")

    fig, ax = plt.subplots(figsize=figsize)

    for cls in classes:
        cls_mask = all_labels == cls
        color = cmap(int(cls) % cmap.N)

        member_cls = cls_mask & is_member
        nonmember_cls = cls_mask & ~is_member

        if member_cls.any():
            ax.scatter(
                embedded[member_cls, 0],
                embedded[member_cls, 1],
                s=12,
                alpha=0.6,
                color=color,
                marker="o",
                edgecolors="none",
            )
        if nonmember_cls.any():
            ax.scatter(
                embedded[nonmember_cls, 0],
                embedded[nonmember_cls, 1],
                s=14,
                alpha=0.45,
                color=color,
                marker="^",
                edgecolors="none",
            )

    # Build a combined legend: one entry per class (colour) + membership markers
    class_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=cmap(int(c) % cmap.N),
            markersize=7,
            label=f"Class {c}",
        )
        for c in classes
    ]
    membership_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markersize=7,
            label="Members (train)",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="gray",
            markersize=7,
            label="Non-members (test)",
        ),
    ]
    n_cols = max(1, (len(classes) + 1) // 2)
    ax.legend(
        handles=class_handles + membership_handles,
        loc="best",
        fontsize=7,
        ncol=n_cols,
    )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def plot_tsne_layer_on_ax(
    ax: plt.Axes,
    train_activations: torch.Tensor,
    test_activations: torch.Tensor,
    train_labels: torch.Tensor,
    test_labels: torch.Tensor,
    *,
    title: str = "",
    perplexity: float = 30.0,
    random_state: int = 42,
    max_samples: int = 5000,
) -> None:
    plot_tsne_classes_on_ax(
        ax,
        train_activations,
        test_activations,
        train_labels,
        test_labels,
        title=title,
        perplexity=perplexity,
        random_state=random_state,
        max_samples=max_samples,
    )


def plot_tsne_all_layers(
    train_layer_activations: dict[str, torch.Tensor],
    test_layer_activations: dict[str, torch.Tensor],
    train_labels: torch.Tensor,
    test_labels: torch.Tensor,
    *,
    title_prefix: str = "t-SNE",
    perplexity: float = 30.0,
    random_state: int = 42,
    max_samples: int = 5000,
    figsize_per_layer: tuple[float, float] = (6, 5),
) -> Figure:
    layer_names = list(train_layer_activations.keys())
    n_layers = len(layer_names)
    if n_layers == 0:
        fig, ax = plt.subplots(figsize=figsize_per_layer)
        ax.text(0.5, 0.5, "No layer activations", transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
        return fig

    fig, axes = plt.subplots(
        1,
        n_layers,
        figsize=(figsize_per_layer[0] * n_layers, figsize_per_layer[1]),
        squeeze=False,
    )

    for col, layer_name in enumerate(layer_names):
        ax = axes[0, col]
        plot_tsne_layer_on_ax(
            ax,
            train_layer_activations[layer_name],
            test_layer_activations[layer_name],
            train_labels,
            test_labels,
            title=f"{title_prefix} – {layer_name}",
            perplexity=perplexity,
            random_state=random_state,
            max_samples=max_samples,
        )

    fig.tight_layout()
    return fig
