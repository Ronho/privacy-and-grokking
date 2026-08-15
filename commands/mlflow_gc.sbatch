#!/bin/bash
#
#SBATCH --job-name=mlflow_gc
#SBATCH --output=logs/mlflow_gc_%j.log
#SBATCH --error=logs/mlflow_gc_%j.err
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --container-image=nvcr.io#nvidia/pytorch:26.07-py3
#SBATCH --container-mounts=/home/ronholzapfel/privacy-and-grokking:/workspace,/home/ronholzapfel:/home/ronholzapfel
#SBATCH --container-workdir=/workspace
#SBATCH --container-writable

mkdir -p logs

echo "Starting job $SLURM_JOB_ID on $HOSTNAME"

export MLFLOW_TRACKING_URI="file:///workspace/mlruns"
export PATH="$HOME/storage/personal/bin:$PATH"
export UV_CACHE_DIR="$HOME/storage/personal/.cache/uv"

uv pip install --system --break-system-packages -e .

echo "Running mlflow gc..."
mlflow gc --backend-store-uri $MLFLOW_TRACKING_URI

echo "Done."
