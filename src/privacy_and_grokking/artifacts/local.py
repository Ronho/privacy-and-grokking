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
    ├── reproduction_nc_grokking-{version}/         ─┐
    │   ├── general_metrics.pdf                     │ plots_reproduction_general_metrics_path
    │   └── nc_metrics.pdf                          ─┘ plots_reproduction_nc_metrics_path
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
    return get_cache_dir() / "plots"


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
        get_cache_dir() / "downloaded_artifacts" / run_id / "checkpoints" / str(step) / "model.pth"
    )


# --- local run mirrors (cache/runs/{run_id}/...) ---


def local_run_dir(run_id: str) -> Path:
    """Root of the locally cached run directory."""
    return get_cache_dir() / "runs" / run_id


def local_run_training_config(run_id: str) -> Path:
    """Locally cached training_config.json for a run."""
    return local_run_dir(run_id) / "training_config.json"

def local_run_margin_attack_data(run_id: str, checkpoint: str) -> Path:
    """Locally cached margin attack data for a run."""
    return local_run_dir(run_id) / "margin_attack" / f"{checkpoint}.parquet"

def trajectory_grid_path(run_id: str) -> Path:
    """Loss landscape grid JSON for a run."""
    return local_run_dir(run_id) / "trajectories" / "grid.json"


def trajectory_projection_path(run_id: str) -> Path:
    """PCA projection trajectory JSON for a run."""
    return local_run_dir(run_id) / "trajectories" / "projection.json"


def filter_normalized_grid_path(run_id: str, step: int | str = "final") -> Path:
    """Filter-normalized loss landscape grid JSON for a run and checkpoint step."""
    step_str = f"step_{step}" if isinstance(step, int) else str(step)
    return local_run_dir(run_id) / "filter_normalized" / f"grid_{step_str}.json"


def interpolation_1d_path(run_id: str) -> Path:
    """1D linear parameter interpolation JSON for a run's trajectory."""
    return local_run_dir(run_id) / "interpolation" / "trajectory_interpolation.json"


def interpolation_1d_pair_path(
    run_id: str, start_step: int | str = 0, end_step: int | str = "final"
) -> Path:
    """1D linear parameter interpolation JSON between two checkpoints."""
    return local_run_dir(run_id) / "interpolation" / f"interpolation_{start_step}_{end_step}.json"


def jump_planes_grid_path(run_id: str) -> Path:
    """2D local subspace trajectory and loss landscape grid JSON for all jumps."""
    return local_run_dir(run_id) / "jump_planes" / "jumps_2d_planes.json"


def checkpoint_distance_cache_path(run_id_1: str, run_id_2: str) -> Path:
    """Euclidean distance matrix JSON between checkpoints of two runs."""
    return local_run_dir(run_id_1) / "distances" / f"{run_id_2}_distance.json"


def joint_trajectories_dir(group_id: str) -> Path:
    """Directory for joint multi-run trajectory artifacts."""
    return get_cache_dir() / "joint_trajectories" / group_id


def joint_trajectory_projection_path(group_id: str) -> Path:
    """Joint PCA projection JSON across multiple runs."""
    return joint_trajectories_dir(group_id) / "projection.json"


def joint_trajectory_grid_path(group_id: str) -> Path:
    """Joint 2D loss landscape grid JSON across multiple runs."""
    return joint_trajectories_dir(group_id) / "grid.json"


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


def canary_margins_cache_dir(experiment_name: str) -> Path:
    """Directory for cached per-sample canary margin data."""
    return get_cache_dir() / "canary_margins" / experiment_name


def canary_margins_cache_path(experiment_name: str, run_id: str, step: int) -> Path:
    """Cached per-run, per-checkpoint canary margin parquet."""
    return canary_margins_cache_dir(experiment_name) / f"{run_id}_{step}.parquet"


# ===================================================================
#  Plots — output directories
# ===================================================================


def plots_hyper_sweep_dir() -> Path:
    """Output directory for hyper sweep plots."""
    return get_plots_dir() / "hyper_sweep"


def plots_optimizer_sweep_overview_path() -> Path:
    """PDF plot path for the optimizer sweep overview."""
    return get_plots_dir() / "optimizer_sweep_overview.pdf"


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


def plots_reproduction_general_metrics_path(version: str = "v1") -> Path:
    """PDF plot path for general metrics reproduction overview."""
    return plots_reproduction_dir(version) / "general_metrics.pdf"


def plots_reproduction_nc_metrics_path(version: str = "v1") -> Path:
    """PDF plot path for neural collapse metrics reproduction overview."""
    return plots_reproduction_dir(version) / "nc_metrics.pdf"


def plot_3d_landscape_path(run_id: str) -> Path:
    """3D loss landscape PNG for a specific run."""
    return get_plots_dir() / f"3d_loss_landscape_{run_id}.png"


def plot_logits_path(run_id: str) -> Path:
    """Logits-vs-loss scatter PNG for a specific run."""
    return get_plots_dir() / f"logits_vs_loss_{run_id}.png"


def plot_run_metrics_path(run_id: str) -> Path:
    """Per-run metrics summary PNG."""
    return get_plots_dir() / f"run_{run_id}_metrics.png"


def plots_optimizer_sweep_dynamics_dir() -> Path:
    """Output directory for optimizer sweep dynamics plots."""
    return get_plots_dir() / "optimizer_sweep_dynamics"


def plots_optimizer_sweep_dynamics_path(
    model_tag: str = "GROK_MNIST_CE_MLP", fmt: str = "pdf"
) -> Path:
    """Canonical plot path for optimizer sweep dynamics multi-column visualization."""
    return plots_optimizer_sweep_dynamics_dir() / f"os_dynamics_{model_tag}.{fmt}"


def plots_checkpoint_distance_dir() -> Path:
    """Output directory for checkpoint distance heatmap plots."""
    return get_plots_dir() / "checkpoint_distances"


def plots_checkpoint_distance_path(run_id_1: str, run_id_2: str) -> Path:
    """PDF plot path for checkpoint distance heatmap between two runs."""
    return plots_checkpoint_distance_dir() / f"distance_{run_id_1[:8]}_{run_id_2[:8]}.pdf"


def plots_checkpoint_distance_grid_path(name: str = "distance_matrix_grid") -> Path:
    """PDF plot path for multi-run checkpoint distance grid heatmap."""
    return plots_checkpoint_distance_dir() / f"{name}.pdf"


def plots_pca_loss_landscape_dir() -> Path:
    """Output directory for PCA loss landscape visualizations."""
    return get_plots_dir() / "pca_loss_landscape"


def plots_pca_loss_landscape_path(run_id: str, fmt: str = "pdf") -> Path:
    """Canonical plot path for a run's PCA loss landscape visualization."""
    return plots_pca_loss_landscape_dir() / f"pca_loss_landscape_{run_id}.{fmt}"


def plots_filter_normalized_landscape_dir() -> Path:
    """Output directory for filter-normalized random direction loss landscapes."""
    return get_plots_dir() / "filter_normalized_landscape"


def plots_filter_normalized_landscape_path(
    run_id: str, step: int | str = "final", fmt: str = "pdf"
) -> Path:
    """Canonical plot path for a run's filter-normalized loss landscape visualization."""
    step_str = f"step_{step}" if isinstance(step, int) else str(step)
    return (
        plots_filter_normalized_landscape_dir()
        / f"filter_normalized_landscape_{run_id}_{step_str}.{fmt}"
    )


def plots_1d_interpolation_dir() -> Path:
    """Output directory for 1D checkpoint interpolation plots."""
    return get_plots_dir() / "interpolation_1d"


def plots_1d_interpolation_path(run_id: str, fmt: str = "pdf") -> Path:
    """Canonical plot path for a run's 1D linear interpolation visualization."""
    return plots_1d_interpolation_dir() / f"interpolation_1d_{run_id}.{fmt}"


def plots_1d_pair_path(
    run_id: str,
    start_step: int | str = 0,
    end_step: int | str = "final",
    fmt: str = "pdf",
) -> Path:
    """Canonical plot path for a 1D linear interpolation between two checkpoints."""
    return plots_1d_interpolation_dir() / f"interpolation_1d_{run_id}_{start_step}_{end_step}.{fmt}"


def plots_1d_jumps_grid_path(run_id: str, fmt: str = "pdf") -> Path:
    """Canonical plot path for 1D linear interpolations across all trajectory jumps."""
    return plots_1d_interpolation_dir() / f"interpolation_1d_{run_id}_jumps_grid.{fmt}"


def plots_jump_planes_dir() -> Path:
    """Output directory for 2D local jump plane loss landscape visualizations."""
    return get_plots_dir() / "jump_planes"


def plots_jump_planes_path(run_id: str, fmt: str = "pdf", short_descriptor: str | None = None) -> Path:
    """Canonical plot path for 2D local jump plane loss landscapes."""
    if short_descriptor:
        return plots_jump_planes_dir() / f"jump_planes_2d_{short_descriptor}_{run_id}.{fmt}"
    return plots_jump_planes_dir() / f"jump_planes_2d_{run_id}.{fmt}"


def plots_jump_planes_legend_path(run_id: str, short_descriptor: str | None = None) -> Path:
    """Canonical path for the exported legend text for jump planes."""
    if short_descriptor:
        return plots_jump_planes_dir() / f"jump_planes_legend_{short_descriptor}_{run_id}.txt"
    return plots_jump_planes_dir() / f"jump_planes_legend_{run_id}.txt"


def plots_joint_trajectories_dir() -> Path:
    """Output directory for joint multi-trajectory loss landscape visualizations."""
    return get_plots_dir() / "joint_trajectories"


def plots_joint_trajectories_path(group_id: str, fmt: str = "pdf") -> Path:
    """Canonical plot path for a joint multi-trajectory loss landscape visualization."""
    return plots_joint_trajectories_dir() / f"joint_loss_landscape_{group_id}.{fmt}"


# ===================================================================
#  Results — final paper-ready outputs
# ===================================================================


def result_pdf_path(name: str, date: str, index: int = 0) -> Path:
    """Paper-ready PDF in results/."""
    return get_results_dir() / f"{name}_{date}-{index}.pdf"
