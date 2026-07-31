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

cd $SLURM_SUBMIT_DIR
mkdir -p logs

export MLFLOW_TRACKING_URI="http://${1}:5050"
export PATH="$HOME/storage/personal/bin:$PATH"
export UV_CACHE_DIR="$HOME/storage/personal/.cache/uv"

uv run pag train slurm-test MSE_CNNEXTENDED_MNIST_GROK.json 10000 0 --seed 0 --run-name my-slurm-test --checkpoint-frequency 25000