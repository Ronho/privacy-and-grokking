#!/bin/bash
#
#SBATCH --job-name=train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --container-image=nvcr.io#nvidia/pytorch:23.12-py3
#SBATCH --container-mounts=/home/ronholzapfel/privacy-and-grokking:/workspace,/home/ronholzapfel:/home/ronholzapfel
#SBATCH --container-workdir=/workspace

mkdir -p logs

echo "Running on host: $(hostname)"
nvidia-smi

export MLFLOW_TRACKING_URI="http://${1}:5050"
export PATH="$HOME/storage/personal/bin:$PATH"
export UV_CACHE_DIR="$HOME/storage/personal/.cache/uv"
export PYTHONUNBUFFERED=1

# Install package dependencies into the container's system Python
uv pip install --system -e .

# Run the command using the container's environment (without uv run)
pag train slurm-test MSE_CNNEXTENDED_MNIST_GROK.json 10000 0 --seed 0 --run-name my-slurm-test --checkpoint-frequency 25000