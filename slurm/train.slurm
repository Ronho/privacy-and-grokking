#!/bin/bash
#SBATCH --job-name=pag_train
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

# ------------------------------------------------------------------------------
# PFLICHT-EINGABE: Prüft, ob $1 (die IP) beim Aufruf übergeben wurde.
# Wenn nicht, bricht der Job sofort ab und zeigt den Hinweistext an.
# ------------------------------------------------------------------------------
MLFLOW_IP="${1:?FEHLER: Du musst deine VPN-IP als Argument uebergeben! Nutzung: sbatch train.slurm <DEINE-VPN-IP>}"

# MLflow Tracking URI dynamisch zusammensetzen
export MLFLOW_TRACKING_URI="http://${MLFLOW_IP}:5000"

# Weitere Pfade
export PATH="$HOME/storage/personal/bin:$PATH"
export UV_CACHE_DIR="$HOME/storage/personal/.cache/uv"

# Ausführung
cd $SLURM_SUBMIT_DIR
mkdir -p logs

uv run pag train v5.0.0 my_config.yaml 10000000 42 --seed 42 --run-name my_run_stem --checkpoint-frequency 25000