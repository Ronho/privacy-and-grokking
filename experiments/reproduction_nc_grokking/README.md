# Reproduction NC Grokking Experiment


## 1. Goal & Hypothesis
This experiment aims to reproducibly and statistically map the well-known Neural Collapse (NC) and grokking effects.
**Hypothesis:** [TODO: Formulate exact hypothesis. e.g. Standard models will consistently exhibit delayed generalization (grokking) and subsequent neural collapse across multiple random seeds without requiring canaries or hyperparameter sweeps.]

## 2. Methodology
Standard models are trained multiple times across different random seeds to establish a robust baseline for grokking behavior (without additional canaries or hyperparameter sweeps).

## 3. Important Metrics
- **Train vs Test Accuracy & Loss:** The classic grokking graph to identify delayed generalization.
- **Neural Collapse Metrics:** Feature variance, equiangularity of class centers (if these are logged).

## 4. Reproduction Steps
Exact commands required to generate jobs and run the experiment.
```bash
# Generation
uv run python experiments/reproduction_nc_grokking/reproduction_nc_grokking.py
# Execution
sbatch experiments/reproduction_nc_grokking/reproduction_nc_grokking.sbatch
```

## 5. Tracking & Results Location
[TODO: Where can the results be found? Provide the MLflow experiment name and relevant tags used to filter these runs in the MLflow UI.]

## 6. Conclusions & Next Steps
[TODO: Brief summary of the outcome: Did the results support the hypothesis? What are the actionable takeaways?]
