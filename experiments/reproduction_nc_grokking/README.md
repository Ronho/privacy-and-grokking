# Reproduction NC Grokking Experiment

## Übersicht
Dieses Experiment zielt darauf ab, den bekannten Neural Collapse (NC) und Grokking-Effekt reproduzierbar und statistisch belastbar abzubilden. Hier werden Standardmodelle mehrmals trainiert, um eine Baseline für das Grokking-Verhalten (ohne zusätzliche Canaries oder Hyper-Sweeps) zu erhalten.

## Wichtige Metriken
- **Train vs Test Accuracy & Loss:** Der klassische Grokking-Graph.
- **Neural Collapse Metriken:** Varianz der Features, Equiangularity der Klassen-Zentren (falls diese geloggt werden).

## Ausführung
Zunächst die Konfigurationen generieren (dies legt eine `.txt` Datei im `jobs/` Verzeichnis ab):
```bash
uv run python experiments/reproduction_nc_grokking/reproduction_nc_grokking.py
```
Anschließend den Slurm-Job starten (aus dem Root-Verzeichnis des Repos):
```bash
sbatch experiments/reproduction_nc_grokking/reproduction_nc_grokking.sbatch
```
