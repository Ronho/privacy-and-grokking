# Find Grokking Experiment

## Übersicht
Dieses Experiment dient dazu, den Bereich im Hyperparameter-Raum (Initialization Scale, Weight Decay, Train Size) zu finden, in dem der Grokking-Effekt für die bereitgestellten Konfigurationen (gekennzeichnet mit `F_`) auftritt. Es führt in der Regel einen 1-dimensionalen Sweep aus (jeweils eine Dimension weicht von der Basis-Konfiguration ab).

## Wichtige Metriken
- **Test Accuracy vs Train Accuracy:** Der Fokus liegt darauf, Modelle zu identifizieren, die verspätet generalisieren (Grokking).
- **Train Loss:** Stabile Konvergenz zu 0 (bzw. 100% Accuracy).
- **Log Frequency:** Regelmäßiges Logging der Test-Metriken (z.B. alle 100-500 Steps), um den genauen "Generalization Point" zu erfassen.

## Ausführung
Zunächst die Konfigurationen generieren (dies legt eine `.txt` Datei im `jobs/` Verzeichnis ab):
```bash
uv run python experiments/find_grokking/find_grokking.py
```
Anschließend den Slurm-Job starten (aus dem Root-Verzeichnis des Repos):
```bash
sbatch experiments/find_grokking/find_grokking.sbatch
```
