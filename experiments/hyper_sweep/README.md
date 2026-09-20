# Hyper Sweep Experiment

**Status:** [Planned | Running | Completed | Abandoned]
**Date:** [YYYY-MM-DD]
**Author:** [Name]
**MLflow Experiment:** `[mlflow-experiment-name]`

## 1. Goal & Hypothesis
This experiment conducts a large-scale hyperparameter sweep to investigate the interplay of `initialization_scale`, `weight_decay`, and `train_size` (dataset size) on the grokking behavior of models.
**Hypothesis:** [TODO: Formulate exact hypothesis. e.g. The grokking phase transition boundary depends heavily on the ratio of initialization scale and dataset size.]

## 2. Methodology
The sweep is typically executed as "one_fixed" (2D planes in the 3D parameter space) or "two_fixed" (1D lines), where unused parameters remain at their configuration defaults.

## 3. Important Metrics
- **Test Accuracy vs Train Accuracy:** Evaluated across the parameter space to visualize where Grokking (Train=100%, Test rises later) occurs compared to Comprehension (both rise simultaneously) or Memorization (only Train rises).
- **Log Frequency:** Regular logging of test metrics (e.g., every 100-500 steps).

## 4. Reproduction Steps
Because this sweep is extensive, the Python script splits the command list into multiple files (`hyper_sweep_0.txt` to `hyper_sweep_4.txt`). The Slurm job is configured as an array job (`--array=0-4`) to execute them in parallel.
```bash
# Generation
uv run python experiments/hyper_sweep/hyper_sweep.py
# Execution
sbatch experiments/hyper_sweep/hyper_sweep.sbatch
```

## 5. Tracking & Results Location
[TODO: Where can the results be found? Provide the MLflow experiment name and relevant tags used to filter these runs in the MLflow UI.]

## 6. Conclusions & Next Steps
[TODO: Brief summary of the outcome: Did the results support the hypothesis? What are the actionable takeaways?]
