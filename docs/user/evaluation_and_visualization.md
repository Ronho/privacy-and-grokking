# Evaluation and Visualization

This guide explains how to extract metrics from experiments, evaluate results, and generate plots.

> [!IMPORTANT]
> To successfully run visualization scripts, you must follow the correct pipeline dependency order. Plotting scripts expect specific artifacts and pre-calculated metrics to already exist.

*(This section will be expanded as the visualization scripts are refactored to use the new Tracking & Artifact abstractions).*

## 1. Extracting Metrics
Before you can plot results, you must extract and format the metrics from your training runs. 
This is usually done by running the `evaluate` command or specific metric extraction scripts.

```bash
uv run pag evaluate v2.3.0
```
*(Example only - precise scripts required for extraction will be listed here).*

### Required Artifacts for Extraction
The extraction scripts typically require:
- Model Checkpoints (`model.pth`)
- Training metrics (`metrics.jsonl` or MLflow API access)

## 2. Running Visualizations
Once metrics are extracted, you can generate plots.

*(Details on visualization scripts and their required input files will be added here).*
