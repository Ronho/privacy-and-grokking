# Visualization Guidelines

All visualization and plotting scripts (located in `visualizations/` and plotting scripts in `scripts/`) must adhere to the standardized structure described below.

---

## 1. Module Docstring Specification

Every visualization script must start with a descriptive top-level docstring containing three required sections:
1. **Summary**: A high-level description of what the visualization illustrates.
2. **Required Inputs**: An explicit list of all prerequisite data files and dependencies needed to run the script (e.g. specific parquet exports in `cache/`, MLflow runs, or local model checkpoints).
3. **Generated Outputs**: The file formats, resolutions, and destination paths (using canonical paths from `privacy_and_grokking.artifacts.local`).

### Example
```python
"""plot_comparison_grid.py.

Generates an all-in-one publication-quality 3x3 multi-panel grid comparing grokking models.

Required Inputs:
  - cache/runs/{run_id}/trajectories/grid.json: 2D loss landscape grid data.
  - cache/runs/{run_id}/trajectories/projection.json: PCA trajectory projections.
  - MLflow server (or local cache): Metric history for train/test loss, accuracy, and weight norms.

Generated Outputs:
  - plots/model_comparison_grid.pdf (vector)
  - plots/model_comparison_grid.png (300 dpi raster)
"""
```

---

## 2. CLI Interface with Typer

Visualization scripts must implement a clean command-line interface using `typer.Typer()`.

### Rules for CLI Options
- **Only include options that are actually useful:** Do not add unused flags (for example, if a visualization operates purely on local Parquet files and never queries the MLflow server, do not include `--tracking-uri`).
- **Standardized Option Names:** When an option is used, always use the canonical name and short option:
  - `--tracking-uri`, `-u`: MLflow tracking URI (default `http://localhost:5051` or `os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5051")`).
  - `--output`, `-o`: Target output file path (defaulting to the canonical path in `privacy_and_grokking.artifacts.local`).
  - `--cache-dir`: Local cache root directory (defaulting to `get_cache_dir()`).
  - Script-specific options (e.g. `--experiment-name`, `--workers`, `--dpi`).

### Example CLI Template
```python
import typer
from pathlib import Path
from typing import Annotated, Optional
from privacy_and_grokking.artifacts.local import get_cache_dir, plots_optimizer_sweep_overview_path
from privacy_and_grokking.artifacts.cache import DEFAULT_TRACKING_URI

app = typer.Typer(pretty_exceptions_enable=False)

@app.command()
def main(
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Destination path for generated PDF."),
    ] = None,
    tracking_uri: Annotated[
        str,
        typer.Option("--tracking-uri", "-u", help="MLflow server URI."),
    ] = DEFAULT_TRACKING_URI,
):
    target_output = output or plots_optimizer_sweep_overview_path()
    ...

if __name__ == "__main__":
    app()
```

---

## 3. Data Processing & Artifact Caching

- **Tabular Data:** Use `polars` for fast, memory-efficient metric processing, joining, and filtering.
- **Run Artifacts:** Never re-implement ad-hoc artifact downloading or hardcode remote `runs:/` URIs. Always use `privacy_and_grokking.artifacts.cache`:
  - `get_artifact_path(run_id, artifact_path, tracking_uri=...)`
  - `get_artifact_json(run_id, artifact_path, tracking_uri=...)`
  - `get_training_config(run_id, tracking_uri=...)`
- These functions check `cache/runs/{run_id}/{artifact_path}` first and only fetch from the remote MLflow server when missing, persisting downloads for subsequent offline execution.

---

## 4. Output Formats & Canonical Paths

- Target paths must be obtained from `privacy_and_grokking.artifacts.local`.
- If a new visualization requires a dedicated path, add a corresponding path helper function to `src/privacy_and_grokking/artifacts/local.py` (e.g. `plots_optimizer_sweep_overview_path()`).
- Where applicable, save both vector format (`.pdf`) and high-resolution preview raster (`.png`, 300 DPI).
