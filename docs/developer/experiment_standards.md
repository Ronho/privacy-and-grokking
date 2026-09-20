# Experiment Tracking Standards

To ensure long-term reproducibility and clarity of intent, this repository adheres to an experiment tracking standard.

This standard enforces a strict separation between the **definition** of an experiment which lives in this Git repository and the **results/metrics** of an experiment which are stored on different servers (Visualizations in Nextcloud, Descriptions in Notion).

## 1. Programmatic Logging
Every training run must log sufficient information to be 100% reproducible. The `privacy_and_grokking.tracking` module and the training loop (`train.py`) automatically handle this.

When a run is executed, the following must be logged to MLflow as artifacts/params:
- **`training_config.json`**: The fully resolved Pydantic configuration.
- **`git/restart_{N}.json`**: The exact Git commit hash and any uncommitted (staged/unstaged) diffs.
- **`env/pip_freeze_{N}.txt`**: The exact Python environment state (e.g. from `uv pip freeze`).
- **`env/command_{N}.txt`**: The exact command line arguments (`sys.argv`) used to launch the job.

## 2. Experiment Definition Template
Every directory inside `experiments/` MUST contain a `README.md` written in **English**. This file defines the experiment's goal, setup, and how to reproduce it, but does **not** contain raw metrics or graphs.

When creating a new experiment, copy the following template into `experiments/<name>/README.md`:

```markdown
# [Experiment Name]

## 1. Goal & Hypothesis
What is the purpose of this experiment? What are we trying to prove or disprove? What is the expected outcome?

## 2. Methodology
What is the setup? Which models, datasets, or specific configurations are being used? How does it differ from the baseline?

## 3. Important Metrics
Which metrics should be monitored to validate or invalidate the hypothesis?

## 4. Reproduction Steps
Exact commands required to generate jobs and run the experiment.
\`\`\`bash
# Generation
uv run python experiments/.../... .py
# Execution
sbatch experiments/.../... .sbatch
\`\`\`

## 5. Tracking & Results Location
Where can the results be found? Provide the MLflow experiment name and any relevant tags (e.g., `mlflow.set_tag("experiment", "...")`) used to filter these runs in the MLflow UI. Do not store actual metrics or graphs here.

## 6. Conclusions & Next Steps
Brief summary of the outcome: Did the results support the hypothesis? What are the actionable takeaways? (Refer to external results for proof).
```
