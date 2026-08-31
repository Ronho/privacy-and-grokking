#!/bin/bash
#
#SBATCH --job-name=export_mlflow_metrics
#SBATCH --output=logs/export_mlflow_metrics_%j.log
#SBATCH --error=logs/export_mlflow_metrics_%j.err
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --container-image=nvcr.io\#nvidia/pytorch:26.07-py3
#SBATCH --container-mounts=/home/ronholzapfel/privacy-and-grokking:/workspace,/home/ronholzapfel:/home/ronholzapfel
#SBATCH --container-workdir=/workspace

mkdir -p logs cache

echo "=========================================================="
echo "Starting Export Job $SLURM_JOB_ID on $HOSTNAME"
echo "Date: $(date)"
echo "=========================================================="

export MLFLOW_TRACKING_URI="file:///workspace/mlruns"
export PATH="$HOME/storage/personal/bin:$PATH"
export UV_CACHE_DIR="$HOME/storage/personal/.cache/uv"
export PYTHONUNBUFFERED=1

uv pip install --system --break-system-packages -e .

if [ $# -eq 0 ]; then
    echo "Usage: sbatch commands/export_metrics.sbatch <EXPERIMENT_NAME> [--history] [--output <OUTPUT_FILE>]"
    echo ""
    echo "Available experiments in $MLFLOW_TRACKING_URI:"
    uv run python -c "import mlflow; mlflow.set_tracking_uri('$MLFLOW_TRACKING_URI'); exps = mlflow.search_experiments(); print('\n'.join([f'  - {e.name} (ID: {e.experiment_id})' for e in exps]) if exps else '  (No experiments found)')"
    exit 1
fi

echo "Running: uv run python scripts/export_mlflow_metrics_local.py $@"
uv run python scripts/export_mlflow_metrics_local.py "$@"

echo "=========================================================="
echo "Job finished at $(date)"
echo "=========================================================="
