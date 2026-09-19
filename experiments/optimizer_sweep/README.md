# Optimizer Sweep Experiment

## Übersicht
Dieses Experiment vergleicht das Verhalten verschiedener Optimizer (AdamW, RMSProp, SGD) mit verschiedenen internen Hyperparametern (wie Momentum, Betas, Epsilon) bezüglich des Grokking-Phänomens. 

## Wichtige Metriken
- **Optimizer-spezifische Metriken:** Z.B. Weight-Updates, Gradienten, Varianz der Parameter. Es wird in hoher Frequenz geloggt (`optimizer_metrics_log_frequency=100`), um das Trainingsverhalten exakt nachzuvollziehen.
- **Accuracy und Loss:** Wann und wie schnell generalisieren die jeweiligen Konfigurationen?

## Ausführung
Da wir für jede Kombination die Optimizer-Settings austauschen, wird hier eine eigene Job-Liste generiert.
```bash
uv run python experiments/optimizer_sweep/optimizer_sweep.py
```

Anschließend wird das SLURM Array-Skript aufgerufen:
```bash
sbatch experiments/optimizer_sweep/optimizer_sweep.sbatch
```
