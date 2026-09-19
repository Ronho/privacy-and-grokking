"""MLflow artifact paths (``runs:/`` URIs and ``artifact_path`` segments).

Every function makes its required parameters explicit.  Functions whose
name ends in ``_uri`` return a full ``runs:/`` URI suitable for
``mlflow.artifacts.download_artifacts`` or ``mlflow.artifacts.load_dict``.
Functions whose name ends in ``_artifact_path`` return the relative
segment used as ``artifact_path=`` in ``mlflow.log_artifact``.

Artifact tree of a training run
================================

::

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
"""

from __future__ import annotations


# -------------------------------------------------------------------
#  Static artifact_path definitions
# -------------------------------------------------------------------

def profiler_artifact_path() -> str:
    """``artifact_path`` for PyTorch profiler traces."""
    return "profiler"


def training_logs_artifact_path() -> str:
    """``artifact_path`` for structured log files."""
    return "training_logs"


def visualizations_artifact_path() -> str:
    """``artifact_path`` for matplotlib figure PDFs."""
    return "visualizations"


def audit_plots_artifact_path() -> str:
    """``artifact_path`` for canary audit histogram PNGs."""
    return "audit_plots"


# -------------------------------------------------------------------
#  Training config
# -------------------------------------------------------------------

def training_config_uri(run_id: str) -> str:
    """``runs:/`` URI for the serialized TrainConfig JSON."""
    return f"runs:/{run_id}/training_config.json"


# -------------------------------------------------------------------
#  Git changes
# -------------------------------------------------------------------

def git_changes_artifact_path() -> str:
    """``artifact_path`` for git diff snapshots."""
    return "git"


def git_changes_filename(restart_index: int) -> str:
    """Filename for the git diff at a specific restart."""
    return f"restart_{restart_index}.json"


# -------------------------------------------------------------------
#  Checkpoints (model + optimizer + rng_state always travel together)
# -------------------------------------------------------------------

def checkpoint_artifact_path(step: int) -> str:
    """``artifact_path`` for ``mlflow.log_artifact`` of checkpoint files."""
    return f"checkpoints/{step}"


def checkpoint_model_uri(run_id: str, step: int) -> str:
    """``runs:/`` URI for the model weights at *step*."""
    return f"runs:/{run_id}/{checkpoint_artifact_path(step)}/model.pth"


def checkpoint_optimizer_uri(run_id: str, step: int) -> str:
    """``runs:/`` URI for the optimizer state at *step*."""
    return f"runs:/{run_id}/{checkpoint_artifact_path(step)}/optimizer.pth"


def checkpoint_rng_state_uri(run_id: str, step: int) -> str:
    """``runs:/`` URI for the RNG state at *step*."""
    return f"runs:/{run_id}/{checkpoint_artifact_path(step)}/rng_state.pth"


# -------------------------------------------------------------------
#  Activations
# -------------------------------------------------------------------

def activation_uri(run_id: str, step: int) -> str:
    """``runs:/`` URI for an activation snapshot."""
    return f"runs:/{run_id}/activations/{step}.pt"


# -------------------------------------------------------------------
#  Audit plots
# -------------------------------------------------------------------

def audit_plot_filename(step: int) -> str:
    """Filename for a canary audit histogram at *step*."""
    return f"canary_audit_{step}.png"
