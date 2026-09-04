#!/bin/bash
#
#SBATCH --job-name=list_mlflow_runs
#SBATCH --output=logs/list_mlflow_runs_%j.log
#SBATCH --error=logs/list_mlflow_runs_%j.err
#SBATCH --partition=defq
#SBATCH --nodelist=BCM-SM-A100-03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --container-image=nvcr.io\#nvidia/pytorch:26.07-py3
#SBATCH --container-mounts=/home/ronholzapfel/privacy-and-grokking:/workspace,/home/ronholzapfel:/home/ronholzapfel
#SBATCH --container-workdir=/workspace

mkdir -p logs

echo "=========================================================="
echo "Starting List Runs Job $SLURM_JOB_ID on $HOSTNAME"
echo "Date: $(date)"
echo "=========================================================="

export MLFLOW_TRACKING_URI="file:///workspace/mlruns"
export PATH="$HOME/storage/personal/bin:$PATH"
export UV_CACHE_DIR="$HOME/storage/personal/.cache/uv"
export PYTHONUNBUFFERED=1

uv pip install --system --break-system-packages -e .

# sbatch list_runs.sbatch reproduction-nc-grokking -o /workspace/reproduction-nc-grokking_runs.parquet
echo "Running: uv run python scripts/list_mlflow_runs.py $@"
uv run python scripts/list_mlflow_runs.py "$@"

echo "=========================================================="
echo "Job finished at $(date)"
echo "=========================================================="
