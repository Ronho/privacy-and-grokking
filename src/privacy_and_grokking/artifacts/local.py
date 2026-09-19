"""Local filesystem paths for generated artifacts.

Covers cached data files (``cache/``), plot outputs (``plots/``),
and final results (``results/``).  None of these belong in version
control — they are all covered by ``.gitignore``.

Local artifact tree
====================

::

    cache/
    ├── {experiment_name}_mlflow_export.parquet      → mlflow_export_cache_path
    ├── {experiment_name}_runs.parquet               → runs_list_cache_path
    ├── {experiment_name}_runs_keep.parquet          → runs_keep_cache_path
    ├── {run_id}_loss_full.parquet                   → loss_full_cache_path
    ├── {run_id}_pca.parquet                         → pca_cache_path
    ├── {run_id}_hard_samples.npz                    → hard_samples_cache_path
    ├── downloaded_artifacts/{run_id}/               ─┐
    │   └── checkpoints/{step}/model.pth             │ downloaded_checkpoint_path
    │                                               ─┘
    ├── runs/{run_id}/                              ─┐
    │   ├── training_config.json                     │ local_run_dir,
    │   ├── checkpoints/{step}/...                   │ local_run_training_config
    │   └── trajectories/                            │
    │       ├── grid.json                            │ trajectory_grid_path,
    │       └── projection.json                     ─┘ trajectory_projection_path
    └── {experiment_name}/                           → informia_cache_dir
        └── informia_results.parquet

    plots/
    ├── hyper_sweep/                                → plots_hyper_sweep_dir
    ├── hyper_sweep_heatmaps/                       → plots_hyper_sweep_heatmaps_dir
    ├── hyper_sweep_dynamics/                       → plots_hyper_sweep_dynamics_dir
    ├── canary_accuracy/                            → plots_canary_accuracy_dir
    ├── reproduction_nc_grokking-{version}/         → plots_reproduction_dir
    ├── mnist_examples/                             → plots_mnist_examples_dir
    ├── info_rmia/                                  → plots_info_rmia_dir
    ├── informia_vs_loss_overlap/                   → plots_informia_overlap_dir
    ├── 3d_loss_landscape_{run_id}.png              → plot_3d_landscape_path
    ├── logits_vs_loss_{run_id}.png                 → plot_logits_path
    └── run_{run_id}_metrics.png                    → plot_run_metrics_path

    results/
    └── {name}_{date}-{index}.pdf                   → result_pdf_path
"""

from __future__ import annotations

from pathlib import Path


# -------------------------------------------------------------------
#  Project root directories
# -------------------------------------------------------------------

def get_project_root() -> Path:
    """Get the root directory of the project."""
    return Path(__file__).resolve().parent.parent.parent.parent


def get_cache_dir() -> Path:
    """Root directory for all cached / intermediate data."""
    return get_project_root() / "cache"


def get_plots_dir() -> Path:
    """Root directory for generated plot outputs."""
    return get_project_root() / "plots"


def get_results_dir() -> Path:
    """Root directory for final paper-ready outputs."""
    return get_project_root() / "results"


# ===================================================================
#  Cache — per-run files
# ===================================================================

def loss_full_cache_path(run_id: str) -> Path:
    """Per-sample loss parquet for a run."""
    return get_cache_dir() / f"{run_id}_loss_full.parquet"


def pca_cache_path(run_id: str) -> Path:
    """PCA projections parquet for a run."""
    return get_cache_dir() / f"{run_id}_pca.parquet"


def hard_samples_cache_path(run_id: str) -> Path:
    """Hard-sample indices and losses for a run."""
    return get_cache_dir() / f"{run_id}_hard_samples.npz"


def downloaded_checkpoint_path(run_id: str, step: int) -> Path:
    """Locally cached checkpoint model file (downloaded from MLflow)."""
    return (
        get_cache_dir()
        / "downloaded_artifacts"
        / run_id
        / "checkpoints"
        / str(step)
        / "model.pth"
    )


# --- local run mirrors (cache/runs/{run_id}/...) ---

def local_run_dir(run_id: str) -> Path:
    """Root of the locally cached run directory."""
    return get_cache_dir() / "runs" / run_id


def local_run_training_config(run_id: str) -> Path:
    """Locally cached training_config.json for a run."""
    return local_run_dir(run_id) / "training_config.json"


def trajectory_grid_path(run_id: str) -> Path:
    """Loss landscape grid JSON for a run."""
    return local_run_dir(run_id) / "trajectories" / "grid.json"


def trajectory_projection_path(run_id: str) -> Path:
    """PCA projection trajectory JSON for a run."""
    return local_run_dir(run_id) / "trajectories" / "projection.json"


# ===================================================================
#  Cache — per-experiment files
# ===================================================================

def mlflow_export_cache_path(experiment_name: str) -> Path:
    """Exported MLflow metrics parquet for an experiment."""
    return get_cache_dir() / f"{experiment_name}_mlflow_export.parquet"


def runs_list_cache_path(experiment_name: str) -> Path:
    """Run listing parquet for an experiment."""
    return get_cache_dir() / f"{experiment_name}_runs.parquet"


def runs_keep_cache_path(experiment_name: str) -> Path:
    """Filtered (kept) runs parquet for an experiment."""
    return get_cache_dir() / f"{experiment_name}_runs_keep.parquet"


def informia_cache_dir(experiment_name: str) -> Path:
    """InfORMIA results directory for an experiment."""
    return get_cache_dir() / experiment_name


# ===================================================================
#  Plots — output directories
# ===================================================================

def plots_hyper_sweep_dir() -> Path:
    """Output directory for hyper sweep plots."""
    return get_plots_dir() / "hyper_sweep"


def plots_hyper_sweep_heatmaps_dir() -> Path:
    """Output directory for hyper sweep heatmap plots."""
    return get_plots_dir() / "hyper_sweep_heatmaps"


def plots_hyper_sweep_dynamics_dir() -> Path:
    """Output directory for hyper sweep dynamics plots."""
    return get_plots_dir() / "hyper_sweep_dynamics"


def plots_canary_accuracy_dir() -> Path:
    """Output directory for canary accuracy plots."""
    return get_plots_dir() / "canary_accuracy"


def plots_mnist_examples_dir() -> Path:
    """Output directory for MNIST example plots."""
    return get_plots_dir() / "mnist_examples"


def plots_info_rmia_dir() -> Path:
    """Output directory for InfoRMIA plots."""
    return get_plots_dir() / "info_rmia"


def plots_informia_overlap_dir() -> Path:
    """Output directory for InfoRMIA vs Loss Overlap plots."""
    return get_plots_dir() / "informia_vs_loss_overlap"


def plots_reproduction_dir(version: str = "v1") -> Path:
    """Output directory for reproduction plots."""
    return get_plots_dir() / f"reproduction_nc_grokking-{version}"


def plot_3d_landscape_path(run_id: str) -> Path:
    """3D loss landscape PNG for a specific run."""
    return get_plots_dir() / f"3d_loss_landscape_{run_id}.png"


def plot_logits_path(run_id: str) -> Path:
    """Logits-vs-loss scatter PNG for a specific run."""
    return get_plots_dir() / f"logits_vs_loss_{run_id}.png"


def plot_run_metrics_path(run_id: str) -> Path:
    """Per-run metrics summary PNG."""
    return get_plots_dir() / f"run_{run_id}_metrics.png"


# ===================================================================
#  Results — final paper-ready outputs
# ===================================================================

def result_pdf_path(name: str, date: str, index: int = 0) -> Path:
    """Paper-ready PDF in results/."""
    return get_results_dir() / f"{name}_{date}-{index}.pdf"
