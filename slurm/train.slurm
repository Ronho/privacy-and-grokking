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

if [ -z "$1" ]; then
    echo "====================================================" >&2
    echo "ERROR: NO IP ADDRESS!" >&2
    echo "USAGE: sbatch train.slurm <IP-ADDRESS>" >&2
    echo "====================================================" >&2
    exit 1
fi

export MLFLOW_TRACKING_URI="http://${1}:5000"
export PATH="$HOME/storage/personal/bin:$PATH"
export UV_CACHE_DIR="$HOME/storage/personal/.cache/uv"

cd $SLURM_SUBMIT_DIR
mkdir -p logs

uv run pag train slurm-test MSE_CNNEXTENDED_MNIST_GROK.json 10000 0 --seed 0 --run-name my-slurm-test --checkpoint-frequency 25000