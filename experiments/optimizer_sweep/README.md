# Optimizer Sweep Experiment

**Status:** [Planned | Running | Completed | Abandoned]
**Date:** [YYYY-MM-DD]
**Author:** [Name]
**MLflow Experiment:** `[mlflow-experiment-name]`

## 1. Goal & Hypothesis
This experiment compares the behavior of different optimizers (AdamW, RMSProp, SGD) and their internal hyperparameters (like Momentum, Betas, Epsilon) regarding the grokking phenomenon.
**Hypothesis:** [TODO: Formulate exact hypothesis. e.g. Adaptive optimizers like AdamW accelerate the grokking phase compared to SGD due to momentum and variance scaling differences.]

## 2. Methodology
The setup systematically substitutes the optimizer settings for a fixed baseline configuration to evaluate their impact on generalization delay.

## 3. Important Metrics
- **Optimizer-specific Metrics:** E.g., Weight updates, gradients, parameter variance. These are logged at a high frequency (`optimizer_metrics_log_frequency=100`) to trace the training dynamics precisely.
- **Accuracy and Loss:** When and how quickly do the respective configurations generalize?

## 4. Reproduction Steps
This generates a custom job list for every combination of optimizer settings.
```bash
# Generation
uv run python experiments/optimizer_sweep/optimizer_sweep.py
# Execution
sbatch experiments/optimizer_sweep/optimizer_sweep.sbatch
```

## 5. Tracking & Results Location
[TODO: Where can the results be found? Provide the MLflow experiment name and relevant tags used to filter these runs in the MLflow UI.]

## 6. Conclusions & Next Steps
[TODO: Brief summary of the outcome: Did the results support the hypothesis? What are the actionable takeaways?]
