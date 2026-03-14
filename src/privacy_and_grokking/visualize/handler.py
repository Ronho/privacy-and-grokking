import tempfile
from pathlib import Path

import mlflow
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from mlflow import MlflowClient
from sklearn.manifold import TSNE

from privacy_and_grokking.utils import Logger, setup_mlflow


class DataHandler:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.mlflow_client = MlflowClient()

    def get_metric_history(self, metric_name: str):
        history = self.mlflow_client.get_metric_history(self.run_id, metric_name)
        steps = [m.step for m in history]
        values = [m.value for m in history]
        return {"steps": steps, "values": values}

    def discover_keys(self, prefix: str) -> list[str]:
        run = self.mlflow_client.get_run(self.run_id)
        return sorted(k for k in run.data.metrics if k.startswith(prefix))

    def load_weight_trajectory(self) -> dict[int, np.ndarray]:
        artifacts = self.mlflow_client.list_artifacts(self.run_id, path="checkpoints")
        steps = sorted(
            int(a.path.split("/")[-1]) for a in artifacts if a.path.split("/")[-1].isdigit()
        )
        result: dict[int, np.ndarray] = {}
        for step in steps:
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    mlflow.artifacts.download_artifacts(
                        artifact_uri=f"runs:/{self.run_id}/checkpoints/{step}/model.pth",
                        dst_path=tmpdir,
                    )
                    state_dict = torch.load(
                        Path(tmpdir) / "model.pth",
                        map_location="cpu",
                        weights_only=True,
                    )
                    result[step] = np.concatenate(
                        [p.float().numpy().ravel() for p in state_dict.values()]
                    )
            except Exception:
                continue
        return result

    def load_activation_data(self) -> dict | None:
        artifacts = self.mlflow_client.list_artifacts(self.run_id, path="activations")
        steps = sorted(int(Path(a.path).stem) for a in artifacts if Path(a.path).stem.isdigit())
        if not steps:
            return None
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                mlflow.artifacts.download_artifacts(
                    artifact_uri=f"runs:/{self.run_id}/activations/{steps[-1]}.pt",
                    dst_path=tmpdir,
                )
                return torch.load(
                    Path(tmpdir) / f"{steps[-1]}.pt",
                    map_location="cpu",
                    weights_only=True,
                )
        except Exception:
            return None


STEP_LABEL = "Optimization Step"


def _accuracy_over_steps(ax, dh: DataHandler):
    logger = Logger.get()
    logger.info("Creating accuracy over steps plot.", extra={"run_id": dh.run_id})

    train = dh.get_metric_history("validation.train.accuracy")
    test = dh.get_metric_history("validation.test.accuracy")

    ax.set_xlabel(STEP_LABEL)
    ax.set_ylabel("Accuracy")
    ax.plot(train["steps"], train["values"], label="Train", color="tab:blue")
    ax.plot(test["steps"], test["values"], label="Test", color="tab:red")
    ax.legend(loc="best")

    logger.info("Created accuracy over steps plot.", extra={"run_id": dh.run_id})


def _loss_over_steps(ax, dh: DataHandler):
    logger = Logger.get()
    logger.info("Creating loss over steps plot.", extra={"run_id": dh.run_id})

    train_mean = dh.get_metric_history("extraction.train.loss.mean")
    train_std = dh.get_metric_history("extraction.train.loss.std")
    test_mean = dh.get_metric_history("extraction.test.loss.mean")
    test_std = dh.get_metric_history("extraction.test.loss.std")
    overlap = dh.get_metric_history("extraction.loss.overlap")

    train_steps = train_mean["steps"]
    train_values = train_mean["values"]
    train_std_values = train_std["values"]
    test_steps = test_mean["steps"]
    test_values = test_mean["values"]
    test_std_values = test_std["values"]

    ax.set_xlabel(STEP_LABEL)
    ax.set_ylabel("Loss")

    ax.plot(train_steps, train_values, label="Train (mean)", color="tab:blue")
    ax.fill_between(
        train_steps,
        [m - s for m, s in zip(train_values, train_std_values)],
        [m + s for m, s in zip(train_values, train_std_values)],
        alpha=0.2,
        color="tab:blue",
        label="Train (±std)",
    )

    ax.plot(test_steps, test_values, label="Test (mean)", color="tab:red")
    ax.fill_between(
        test_steps,
        [m - s for m, s in zip(test_values, test_std_values)],
        [m + s for m, s in zip(test_values, test_std_values)],
        alpha=0.2,
        color="tab:red",
        label="Test (±std)",
    )

    ax2 = ax.twinx()
    ax2.set_ylabel("Overlap", color="tab:orange")
    ax2.plot(
        overlap["steps"], overlap["values"], label="Overlap", color="tab:orange", linestyle="--"
    )
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="best")

    logger.info("Created loss over steps plot.", extra={"run_id": dh.run_id})


_NORM_LAYER_COLORS = [
    "tab:blue",
    "tab:red",
    "tab:green",
    "tab:orange",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:cyan",
]
_NORM_TOTAL_COLOR = "black"


def _plot_norms_over_steps(ax, dh: DataHandler, prefix: str, ylabel: str):
    all_keys = dh.discover_keys(prefix)
    total_key = f"{prefix}total"
    layer_keys = [k for k in all_keys if k != total_key]

    seen: dict[str, int] = {}
    for key in layer_keys:
        name = key[len(prefix) :]
        base = name.removesuffix(".weight").removesuffix(".bias")
        if base not in seen:
            seen[base] = len(seen)

    for key in layer_keys:
        name = key[len(prefix) :]
        base = name.removesuffix(".weight").removesuffix(".bias")
        color = _NORM_LAYER_COLORS[seen[base] % len(_NORM_LAYER_COLORS)]
        linestyle = "--" if name.endswith(".bias") else "-"
        data = dh.get_metric_history(key)
        ax.plot(
            data["steps"],
            data["values"],
            label=name,
            color=color,
            linestyle=linestyle,
            linewidth=1,
            alpha=0.8,
        )

    if total_key in all_keys:
        data = dh.get_metric_history(total_key)
        ax.plot(data["steps"], data["values"], label="total", color=_NORM_TOTAL_COLOR, linewidth=2)

    ax.set_xlabel(STEP_LABEL)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")


def _weight_norms_over_steps(ax, dh: DataHandler):
    logger = Logger.get()
    logger.info("Creating weight norms over steps plot.", extra={"run_id": dh.run_id})
    _plot_norms_over_steps(ax, dh, prefix="weight_norm/", ylabel="Weight Norm")
    logger.info("Created weight norms over steps plot.", extra={"run_id": dh.run_id})


def _gradient_norms_over_steps(ax, dh: DataHandler):
    logger = Logger.get()
    logger.info("Creating gradient norms over steps plot.", extra={"run_id": dh.run_id})
    _plot_norms_over_steps(ax, dh, prefix="grad_norm/", ylabel="Gradient Norm")
    logger.info("Created gradient norms over steps plot.", extra={"run_id": dh.run_id})


_MIA_NICE_NAMES: dict[str, str] = {
    "mia_prob/auc": "Prob",
    "mia_logit/auc": "Logit",
    "mia_ce_loss/auc": "CE Loss",
    "mia_mse_loss/auc": "MSE Loss",
    "mia_correctness/auc": "Correct",
    "mia_merlin_morgan_ce/auc": "MM CE",
    "mia_merlin_morgan_mse/auc": "MM MSE",
}

_MIA_COLORS = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
]


def _mia_auc_over_steps(ax, dh: DataHandler):
    logger = Logger.get()
    logger.info("Creating MIA AUC over steps plot.", extra={"run_id": dh.run_id})

    auc_keys = [k for k in dh.discover_keys("mia_") if k.endswith("/auc")]

    for i, key in enumerate(auc_keys):
        data = dh.get_metric_history(key)
        if not data["steps"]:
            continue
        label = _MIA_NICE_NAMES.get(key, key)
        color = _MIA_COLORS[i % len(_MIA_COLORS)]
        ax.plot(data["steps"], data["values"], label=label, color=color, linewidth=1.5)

    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.5, label="Random (0.5)")
    ax.set_xlabel(STEP_LABEL)
    ax.set_ylabel("AUC")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="best")

    logger.info("Created MIA AUC over steps plot.", extra={"run_id": dh.run_id})


def _class_distribution(ax, dh: DataHandler):
    logger = Logger.get()
    logger.info("Creating class distribution plot.", extra={"run_id": dh.run_id})

    data = dh.load_activation_data()
    if data is None:
        ax.text(
            0.5,
            0.5,
            "No activation data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        logger.warning(
            "No activation data found for class distribution.", extra={"run_id": dh.run_id}
        )
        return

    train_labels: torch.Tensor = data["train_labels"]
    test_labels: torch.Tensor = data["test_labels"]

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


def _training_trajectory(ax, dh: DataHandler):
    logger = Logger.get()
    logger.info("Creating training trajectory plot.", extra={"run_id": dh.run_id})

    traj = dh.load_weight_trajectory()

    if len(traj) < 3:
        ax.text(
            0.5,
            0.5,
            "Insufficient checkpoint data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        logger.warning("Not enough checkpoints for trajectory plot.", extra={"run_id": dh.run_id})
        return

    sorted_steps = sorted(traj.keys())
    weight_matrix = np.stack([traj[s] for s in sorted_steps], axis=0)  # (T, D)

    w_centred = weight_matrix - weight_matrix.mean(axis=0, keepdims=True)

    if np.allclose(w_centred, 0, atol=1e-9):
        ax.text(0.5, 0.5, "All weights identical", ha="center", va="center", transform=ax.transAxes)
        return

    u_matrix, singular_values, _ = np.linalg.svd(w_centred, full_matrices=False)
    coords = u_matrix[:, :2] * singular_values[:2]  # (T, 2)

    steps_arr = np.array(sorted_steps, dtype=float)
    norm_steps = (steps_arr - steps_arr.min()) / max(steps_arr.max() - steps_arr.min(), 1.0)

    for i in range(len(coords) - 1):
        dx = coords[i + 1, 0] - coords[i, 0]
        dy = coords[i + 1, 1] - coords[i, 1]
        if dx * dx + dy * dy < 1e-18:
            continue
        ax.annotate(
            "",
            xy=(coords[i + 1, 0], coords[i + 1, 1]),
            xytext=(coords[i, 0], coords[i, 1]),
            arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=0.8, alpha=0.5),
            zorder=2,
        )

    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=norm_steps,
        cmap="viridis",
        s=25,
        zorder=3,
        edgecolors="none",
    )

    ax.scatter(
        [coords[0, 0]],
        [coords[0, 1]],
        color="#22c55e",
        s=100,
        zorder=4,
        marker="o",
        label=f"Start (step {sorted_steps[0]})",
    )
    ax.scatter(
        [coords[-1, 0]],
        [coords[-1, 1]],
        color="#ef4444",
        s=100,
        zorder=4,
        marker="*",
        label=f"End (step {sorted_steps[-1]})",
    )

    var_total = float((singular_values**2).sum())
    var1 = float(singular_values[0] ** 2) / var_total * 100
    var2 = float(singular_values[1] ** 2) / var_total * 100
    ax.set_xlabel(f"PC1 ({var1:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var2:.1f}% var)")
    ax.legend(loc="best", fontsize=7)

    cbar = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label("Training progress")
    cbar.ax.tick_params(labelsize=7)

    logger.info("Created training trajectory plot.", extra={"run_id": dh.run_id})


VISUALIZATIONS = {
    "accuracy_over_steps": _accuracy_over_steps,
    "loss_over_steps": _loss_over_steps,
    "weight_norms_over_steps": _weight_norms_over_steps,
    "gradient_norms_over_steps": _gradient_norms_over_steps,
    "mia_auc_over_steps": _mia_auc_over_steps,
    "class_distribution": _class_distribution,
    "training_trajectory": _training_trajectory,
}


def _class_layer_activation_grid(dh: DataHandler) -> plt.Figure | None:
    logger = Logger.get()
    logger.info("Creating class-layer activation grid.", extra={"run_id": dh.run_id})

    data = dh.load_activation_data()
    if data is None:
        logger.warning("No activation data for grid plot.", extra={"run_id": dh.run_id})
        return None

    train_acts: dict[str, torch.Tensor] = data.get("train_activations", {})
    test_acts: dict[str, torch.Tensor] = data.get("test_activations", {})
    train_labels: torch.Tensor = data["train_labels"]
    test_labels: torch.Tensor = data["test_labels"]

    layers = sorted(train_acts.keys())
    classes = sorted(torch.cat([train_labels, test_labels]).unique().tolist())
    n_layers = len(layers)
    n_classes = len(classes)

    cell_w, cell_h = 2.2, 1.8
    fig, axes = plt.subplots(
        n_layers,
        n_classes,
        figsize=(cell_w * n_classes, cell_h * n_layers),
        squeeze=False,
    )

    run = mlflow.get_run(dh.run_id)
    run_name = run.data.tags.get("mlflow.runName", dh.run_id)
    fig.suptitle(run_name, fontsize=10, y=1.01)

    for row, layer in enumerate(layers):
        tr_layer: torch.Tensor = train_acts[layer].float()  # (N_train, D)
        te_layer: torch.Tensor = test_acts[layer].float()  # (N_test, D)
        n_neurons = tr_layer.shape[1]
        neuron_idx = np.arange(n_neurons)

        for col, cls in enumerate(classes):
            ax = axes[row][col]

            tr_mask = train_labels == cls
            te_mask = test_labels == cls

            tr_cls = tr_layer[tr_mask]  # (N_cls, D)
            te_cls = te_layer[te_mask]

            if tr_cls.shape[0] > 0:
                tr_mean = tr_cls.mean(dim=0).numpy()
                tr_std = tr_cls.std(dim=0).numpy()
                ax.plot(neuron_idx, tr_mean, color="tab:blue", linewidth=0.9, label="Train")
                ax.fill_between(
                    neuron_idx, tr_mean - tr_std, tr_mean + tr_std, color="tab:blue", alpha=0.15
                )

            if te_cls.shape[0] > 0:
                te_mean = te_cls.mean(dim=0).numpy()
                te_std = te_cls.std(dim=0).numpy()
                ax.plot(neuron_idx, te_mean, color="tab:red", linewidth=0.9, label="Test")
                ax.fill_between(
                    neuron_idx, te_mean - te_std, te_mean + te_std, color="tab:red", alpha=0.15
                )

            ax.set_xticks([])
            ax.set_yticks([])
            ax.tick_params(labelsize=5)
            ax.grid(True, alpha=0.2)

            if row == 0:
                ax.set_title(f"Class {int(cls)}", fontsize=7)
            if col == 0:
                ax.set_ylabel(layer, fontsize=6, rotation=0, ha="right", va="center", labelpad=4)
            if row == n_layers - 1 and col == n_classes - 1:
                handles, labels = ax.get_legend_handles_labels()
                fig.legend(
                    handles,
                    labels,
                    loc="lower right",
                    fontsize=7,
                    ncol=2,
                    bbox_to_anchor=(1.0, 0.0),
                )

    fig.tight_layout()
    logger.info("Created class-layer activation grid.", extra={"run_id": dh.run_id})
    return fig


def _correlation_rdm(x: np.ndarray) -> np.ndarray:
    """Compute pairwise correlation-distance matrix (1 - pearson r) for rows of x."""
    x = x - x.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1e-10, norms)
    x_normed = x / norms
    corr = np.clip(x_normed @ x_normed.T, -1.0, 1.0)
    return 1.0 - corr


def _rdm_per_layer(dh: DataHandler) -> plt.Figure | None:
    logger = Logger.get()
    logger.info("Creating RDM per layer plot.", extra={"run_id": dh.run_id})

    data = dh.load_activation_data()
    if data is None:
        logger.warning("No activation data for RDM plot.", extra={"run_id": dh.run_id})
        return None

    train_acts: dict[str, torch.Tensor] = data.get("train_activations", {})
    test_acts: dict[str, torch.Tensor] = data.get("test_activations", {})
    train_labels: torch.Tensor = data["train_labels"]
    test_labels: torch.Tensor = data["test_labels"]

    if not train_acts:
        logger.warning("train_activations is empty.", extra={"run_id": dh.run_id})
        return None

    n_per_class = 5
    classes = sorted(torch.cat([train_labels, test_labels]).unique().tolist())
    layers = sorted(train_acts.keys())

    # --- Build sorted sample index: train (class 0..N), then test (class 0..N) ---
    rng = np.random.default_rng(seed=0)

    def _sample_indices(labels: torch.Tensor, cls: int, n: int) -> np.ndarray:
        idx = (labels == cls).nonzero(as_tuple=False).squeeze(1).numpy()
        if len(idx) >= n:
            return rng.choice(idx, size=n, replace=False)
        return idx  # fewer than requested — use all

    tr_idx_per_class = [_sample_indices(train_labels, c, n_per_class) for c in classes]
    te_idx_per_class = [_sample_indices(test_labels, c, n_per_class) for c in classes]

    # Ordered index arrays into the full train / test activation tensors
    tr_order = np.concatenate(tr_idx_per_class)
    te_order = np.concatenate(te_idx_per_class)

    # Class label for every row in the ordered array (for tick annotation)
    tr_class_labels = np.concatenate(
        [[c] * len(tr_idx_per_class[i]) for i, c in enumerate(classes)]
    )
    te_class_labels = np.concatenate(
        [[c] * len(te_idx_per_class[i]) for i, c in enumerate(classes)]
    )
    all_class_labels = np.concatenate([tr_class_labels, te_class_labels])
    n_total = len(all_class_labels)

    # --- Ideal RDM: 0 if same class, 1 otherwise ---
    ideal = (all_class_labels[:, None] != all_class_labels[None, :]).astype(float)

    n_cols = 1 + len(layers)  # ideal + one per layer
    cell = max(3.0, n_total * 0.06)
    fig, axes = plt.subplots(1, n_cols, figsize=(cell * n_cols, cell + 1.2), squeeze=False)

    run = mlflow.get_run(dh.run_id)
    run_name = run.data.tags.get("mlflow.runName", dh.run_id)
    fig.suptitle(run_name, fontsize=10)

    n_train_total = len(tr_order)

    def _annotate_rdm(ax: plt.Axes, rdm: np.ndarray, title: str) -> None:
        im = ax.imshow(rdm, aspect="auto", cmap="RdBu_r", vmin=0.0, vmax=1.0, interpolation="none")
        ax.set_title(title, fontsize=7)

        # Tick at mid-point of each class block for train rows/cols
        tick_positions = []
        tick_names = []
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

        # Grid lines at class boundaries
        boundary_positions = []
        cursor = 0
        for idx_list in tr_idx_per_class + te_idx_per_class:
            cursor += len(idx_list)
            boundary_positions.append(cursor - 0.5)
        for pos in boundary_positions:
            ax.axhline(pos, color="white", linewidth=0.4, alpha=0.7)
            ax.axvline(pos, color="white", linewidth=0.4, alpha=0.7)

        # Train/test separator
        ax.axhline(n_train_total - 0.5, color="black", linewidth=1.2)
        ax.axvline(n_train_total - 0.5, color="black", linewidth=1.2)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=5)

    _annotate_rdm(axes[0][0], ideal, "Ideal")

    for col, layer in enumerate(layers, start=1):
        tr_layer = train_acts[layer].float().numpy()  # (N_train, D)
        te_layer = test_acts[layer].float().numpy()  # (N_test, D)

        acts = np.concatenate([tr_layer[tr_order], te_layer[te_order]], axis=0)
        rdm = _correlation_rdm(acts)
        _annotate_rdm(axes[0][col], rdm, layer)

    fig.tight_layout()
    logger.info("Created RDM per layer plot.", extra={"run_id": dh.run_id})
    return fig


def _tsne_on_ax(
    ax: plt.Axes,
    train_acts: np.ndarray,
    test_acts: np.ndarray,
    train_lbl: np.ndarray,
    test_lbl: np.ndarray,
    *,
    title: str = "",
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
    if title:
        ax.set_title(title, fontsize=8)
    ax.set_xlabel("t-SNE 1", fontsize=7)
    ax.set_ylabel("t-SNE 2", fontsize=7)
    ax.grid(True, alpha=0.2)


def _tsne_per_layer(dh: DataHandler) -> plt.Figure | None:
    logger = Logger.get()
    logger.info("Creating t-SNE per layer plot.", extra={"run_id": dh.run_id})

    data = dh.load_activation_data()
    if data is None:
        logger.warning("No activation data for t-SNE plot.", extra={"run_id": dh.run_id})
        return None

    train_layer_acts: dict[str, torch.Tensor] | None = data.get("train_layer_activations")
    test_layer_acts: dict[str, torch.Tensor] | None = data.get("test_layer_activations")
    train_labels: torch.Tensor = data["train_labels"]
    test_labels: torch.Tensor = data["test_labels"]

    # Fall back to the single flattened activation if per-layer data is absent
    if not train_layer_acts:
        train_acts = data.get("train_activations")
        test_acts = data.get("test_activations")
        if train_acts is None:
            logger.warning("No layer activations found.", extra={"run_id": dh.run_id})
            return None
        train_layer_acts = {"activations": train_acts}
        test_layer_acts = {"activations": test_acts}

    run = mlflow.get_run(dh.run_id)
    run_name = run.data.tags.get("mlflow.runName", dh.run_id)

    layers = list(train_layer_acts.keys())
    n_layers = len(layers)
    fig, axes = plt.subplots(1, n_layers, figsize=(6.0 * n_layers, 5.0), squeeze=False)

    train_lbl_np = train_labels.numpy()
    test_lbl_np = test_labels.numpy()

    for col, layer in enumerate(layers):
        _tsne_on_ax(
            axes[0][col],
            train_layer_acts[layer].float().numpy(),
            test_layer_acts[layer].float().numpy(),
            train_lbl_np,
            test_lbl_np,
            title=f"{run_name} – {layer}",
        )

    fig.tight_layout()
    logger.info("Created t-SNE per layer plot.", extra={"run_id": dh.run_id})
    return fig


FIGURE_VISUALIZATIONS: dict[str, object] = {
    "class_layer_activation_grid": _class_layer_activation_grid,
    "rdm_per_layer": _rdm_per_layer,
    "tsne_per_layer": _tsne_per_layer,
}

SINGLE_VIZ_NAMES: list[str] = list(VISUALIZATIONS) + list(FIGURE_VISUALIZATIONS)
MULTI_VIZ_NAMES: list[str] = list(VISUALIZATIONS)


def _save_figure_to_mlflow(fig, filename: str, run_id: str, filetype: str = "png"):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / f"{filename}.{filetype}"
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        mlflow.log_artifact(str(path), artifact_path="visualizations", run_id=run_id)


def _single_image_handler(dh: DataHandler, plot_func, filename: str):
    logger = Logger.get()
    logger.info(f"Creating {filename} plot.", extra={"run_id": dh.run_id, "filename": filename})

    run = mlflow.get_run(dh.run_id)
    run_name = run.data.tags.get("mlflow.runName")

    fig, ax = plt.subplots(figsize=(6, 5))
    plot_func(ax, dh)
    ax.set_yscale("linear")
    ax.set_xscale("linear")
    ax.grid(True, alpha=0.3, which="major", axis="both")
    fig.suptitle(run_name)
    fig.tight_layout()

    _save_figure_to_mlflow(fig, filename, run_id=dh.run_id)

    plt.close(fig)
    logger.info(f"Created {filename} plot.", extra={"run_id": dh.run_id, "filename": filename})


def visualization_single_handler(
    exp_name: str,
    run_id: str,
    include: list[str] | None = None,
):
    setup_mlflow(exp_name)

    all_names = list(VISUALIZATIONS) + list(FIGURE_VISUALIZATIONS)
    visualizations = [k for k in all_names if include is None or k in include]

    with Logger() as logger:
        logger.info(
            "Starting visualization.",
            run_id=run_id,
            visualizations=sorted(visualizations),
        )
        dh = DataHandler(run_id)
        for viz_name in visualizations:
            if viz_name in VISUALIZATIONS:
                _single_image_handler(dh, VISUALIZATIONS[viz_name], viz_name)
            elif viz_name in FIGURE_VISUALIZATIONS:
                fig_func = FIGURE_VISUALIZATIONS[viz_name]
                fig = fig_func(dh)
                if fig is not None:
                    _save_figure_to_mlflow(fig, viz_name, run_id=run_id)
                    plt.close(fig)
        logger.info("Visualization complete.", run_id=run_id)


def visualization_multi_handler(
    exp_name: str,
    run_ids: list[str],
    include: list[str] | None = None,
):
    setup_mlflow(exp_name)

    viz_names = [k for k in VISUALIZATIONS if include is None or k in include]
    if not viz_names or not run_ids:
        return

    n_rows = len(viz_names)
    n_cols = len(run_ids)

    col_w, row_h = 6.0, 5.0
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(col_w * n_cols, row_h * n_rows),
        squeeze=False,
    )

    with Logger() as logger:
        logger.info(
            "Starting multi-run visualization.",
            run_ids=run_ids,
            visualizations=sorted(viz_names),
        )

        # Resolve run names upfront
        run_names: list[str] = []
        for rid in run_ids:
            run = mlflow.get_run(rid)
            run_names.append(run.data.tags.get("mlflow.runName", rid))

        data_handlers = [DataHandler(rid) for rid in run_ids]

        for row, viz_name in enumerate(viz_names):
            plot_func = VISUALIZATIONS[viz_name]
            for col, dh in enumerate(data_handlers):
                ax = axes[row][col]
                try:
                    plot_func(ax, dh)
                except Exception as exc:
                    ax.text(
                        0.5,
                        0.5,
                        f"Error:\n{exc}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=7,
                        color="tab:red",
                        wrap=True,
                    )
                ax.set_yscale("linear")
                ax.set_xscale("linear")
                ax.grid(True, alpha=0.3, which="major", axis="both")

                # Column header on the top row: bold run name + faded run ID
                if row == 0:
                    ax.set_title(
                        run_names[col],
                        fontsize=9,
                        fontweight="bold",
                        loc="center",
                    )
                    ax.text(
                        0.5,
                        1.02,
                        run_ids[col],
                        transform=ax.transAxes,
                        fontsize=6,
                        color="#888888",
                        ha="center",
                        va="bottom",
                    )

                # Row label on the left column
                if col == 0:
                    ax.set_ylabel(
                        f"{viz_name}\n{ax.get_ylabel()}",
                        fontsize=7,
                    )

        fig.tight_layout()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "multi_run_comparison.png"
            fig.savefig(str(path), dpi=150, bbox_inches="tight")
            for rid in run_ids:
                mlflow.log_artifact(str(path), artifact_path="visualizations", run_id=rid)

        plt.close(fig)
        logger.info("Multi-run visualization complete.", run_ids=run_ids)
