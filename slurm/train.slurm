#!/bin/bash
#
#SBATCH --job-name=grid-search-sgd
#SBATCH --output=logs/grid-search-sgd_%j.out
#SBATCH --error=logs/grid-search-sgd_%j.err
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --container-image=nvcr.io\#nvidia/pytorch:26.07-py3
#SBATCH --container-mounts=/home/ronholzapfel/privacy-and-grokking:/workspace,/home/ronholzapfel:/home/ronholzapfel
#SBATCH --container-workdir=/workspace
#SBATCH --container-writable

mkdir -p logs

echo "Running on host: $(hostname)"
nvidia-smi

export MLFLOW_TRACKING_URI="http://${1}:5051"
export PATH="$HOME/storage/personal/bin:$PATH"
export UV_CACHE_DIR="$HOME/storage/personal/.cache/uv"
export PYTHONUNBUFFERED=1

uv pip install --system --break-system-packages -e .

pag process commands/sgd-grid-search.txt 4

echo "All training runs in the queue have completed!"