# Canary Selection Experiment

## Übersicht
In diesem Experiment wird die Selektion und Injektion von "Canaries" (markante, atypische oder verrauschte Trainingsbeispiele) in die Trainingsdatenbank untersucht. Ziel ist es herauszufinden, wie verschiedene Arten von Canaries (Uniform Noise, Square Watermark, Gaussian Noise, Label Noise, OOD Natural) im Laufe des Trainings – insbesondere im Hinblick auf den Grokking-Effekt – memoriert werden.

## Wichtige Metriken
- **Train/Test Loss & Accuracy:** Um das generelle Modellverhalten und den Grokking-Zeitpunkt (Verzögerung zwischen Train- und Test-Accuracy) festzustellen.
- **Canary Loss / Memorization Metric:** Zu evaluieren, wie stark und zu welchem Zeitpunkt die Canaries im Vergleich zu den regulären Daten gelernt werden.
- **Log Frequency:** Empfohlen ist ein häufiges Logging (z.B. alle 100-500 Steps), um den genauen Zeitpunkt der Canary-Memorisation aufzulösen.

## Ausführung
Zunächst die Konfigurationen generieren (dies legt eine `.txt` Datei im `jobs/` Verzeichnis ab):
```bash
uv run python experiments/canary_selection/canary_selection.py
```
Anschließend den Slurm-Job starten (aus dem Root-Verzeichnis des Repos):
```bash
sbatch experiments/canary_selection/canary_selection.sbatch
```
