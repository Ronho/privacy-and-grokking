# Hyper Sweep Experiment

## Übersicht
Dieses Experiment führt einen großen Hyperparameter-Sweep durch, um das Zusammenspiel von `initialization_scale`, `weight_decay` und `train_size` (Datenmenge) auf das Grokking-Verhalten der Modelle zu untersuchen. Der Sweep wird meist als "one_fixed" (2D-Ebenen im 3D-Parameterraum) oder "two_fixed" (1D-Linien) durchgeführt, wobei ungenutzte Parameter auf ihren Konfigurations-Defaults bleiben.

## Wichtige Metriken
- **Test Accuracy vs Train Accuracy:** Über den Parameterraum hinweg um zu visualisieren, wo Grokking (Train=100%, Test steigt später) im Vergleich zu Comprehension (beide steigen zeitgleich) oder Memorization (nur Train steigt) auftritt.
- **Log Frequency:** Regelmäßiges Logging der Test-Metriken (z.B. alle 100-500 Steps).

## Ausführung
Da dieser Sweep sehr umfangreich ist, teilt das Python-Skript die Command-Liste auf mehrere Dateien (`hyper_sweep_0.txt` bis `hyper_sweep_4.txt`) auf.
```bash
uv run python experiments/hyper_sweep/hyper_sweep.py
```
Der Slurm-Job ist als Array-Job (`--array=0-4`) konfiguriert und führt die aufgeteilten Listen parallel aus:
```bash
sbatch experiments/hyper_sweep/hyper_sweep.sbatch
```
