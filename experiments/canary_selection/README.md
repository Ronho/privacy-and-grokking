# Canary Selection Experiment

**Status:** [Planned | Running | Completed | Abandoned]
**Date:** [YYYY-MM-DD]
**Author:** [Name]
**MLflow Experiment:** `[mlflow-experiment-name]`

## 1. Goal & Hypothesis
This experiment investigates the selection and injection of "canaries" (distinctive, atypical, or noisy training examples) into the training dataset. 
**Hypothesis:** [TODO: Formulate exact hypothesis. e.g. Different types of canaries will be memorized at different stages of training relative to the grokking phase.]

## 2. Methodology
Different types of canaries (Uniform Noise, Square Watermark, Gaussian Noise, Label Noise, OOD Natural) are injected into the standard training dataset. The model's ability to memorize these specific examples is tracked throughout the training process, specifically observing the timeline relative to grokking.

## 3. Important Metrics
- **Train/Test Loss & Accuracy:** Used to identify general model behavior and the grokking point (delay between train and test accuracy).
- **Canary Loss / Memorization Metric:** Evaluates how strongly and at what exact step the canaries are memorized compared to regular data.
- **Log Frequency:** High frequency logging (e.g., every 100-500 steps) is recommended to precisely capture the canary memorization point.

## 4. Reproduction Steps
Exact commands required to generate jobs and run the experiment.
```bash
# Generation
uv run python experiments/canary_selection/canary_selection.py
# Execution
sbatch experiments/canary_selection/canary_selection.sbatch
```

## 5. Tracking & Results Location
[TODO: Where can the results be found? Provide the MLflow experiment name and relevant tags used to filter these runs in the MLflow UI.]

## 6. Conclusions & Next Steps
[TODO: Brief summary of the outcome: Did the results support the hypothesis? What are the actionable takeaways?]
