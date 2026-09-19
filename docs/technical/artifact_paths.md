# Artifact Paths

To keep track of where files are saved (both locally and remotely via MLflow), we strictly use centralized functions defined in `src/privacy_and_grokking/artifacts/`.
Hardcoded string literals for paths are strictly forbidden in the main codebase.

## Local Filesystem (`local.py`)

The local artifact tree covers cached data files (`cache/`), plot outputs (`plots/`), and final results (`results/`). None of these belong in version control.

```text
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
```


## Remote MLflow Paths (`remote.py`)

Functions whose name ends in `_uri` return a full `runs:/` URI suitable for `mlflow.artifacts.download_artifacts`.
Functions whose name ends in `_artifact_path` return the relative segment used as `artifact_path=` in `mlflow.log_artifact` or `RunTracker.log_artifact`.

```text
runs:/<run_id>/
├── training_config.json                        → training_config_uri
├── git/
│   └── restart_{restart_index}.json            → git_changes_artifact_path,
│                                                 git_changes_filename
├── checkpoints/{step}/                         ─┐
│   ├── model.pth                                │ checkpoint_artifact_path,
│   ├── optimizer.pth                            │ checkpoint_model_uri,
│   └── rng_state.pth                           ─┘ checkpoint_optimizer_uri,
│                                                  checkpoint_rng_state_uri
├── activations/
│   └── {step}.pt                               → activation_uri
├── profiler/                                   → profiler_artifact_path
├── training_logs/                              → training_logs_artifact_path
├── visualizations/                             → visualizations_artifact_path
└── audit_plots/
    └── canary_audit_{step}.png                 → audit_plots_artifact_path,
                                                  audit_plot_filename
```
