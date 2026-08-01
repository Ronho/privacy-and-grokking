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

# Install package dependencies into the container's system Python
uv pip install --system --break-system-packages -e .

# Start 4 parallel training runs on the same GPU!
# The '&' puts the command in the background, allowing the loop to continue immediately.
pag train grid-search-sgd MSE_MLPEXTENDED_MNIST_GROK.json 150000 0 --run-name sgd-grid-1000-8 --checkpoint-frequency 10000 -o 'optimizer={"name": "SGD", "lr": 0.001, "weight_decay": 0}' -o model.initialization_scale=8.0 -o data.train_size=1000 &
pag train grid-search-sgd MSE_MLPEXTENDED_MNIST_GROK.json 150000 0 --run-name sgd-grid-500-8 --checkpoint-frequency 10000 -o 'optimizer={"name": "SGD", "lr": 0.001, "weight_decay": 0}' -o model.initialization_scale=8.0 -o data.train_size=500 &
pag train grid-search-sgd MSE_MLPEXTENDED_MNIST_GROK.json 150000 0 --run-name sgd-grid-250-8 --checkpoint-frequency 10000 -o 'optimizer={"name": "SGD", "lr": 0.001, "weight_decay": 0}' -o model.initialization_scale=8.0 -o data.train_size=250 &
pag train grid-search-sgd MSE_MLPEXTENDED_MNIST_GROK.json 150000 0 --run-name sgd-grid-150-8 --checkpoint-frequency 10000 -o 'optimizer={"name": "SGD", "lr": 0.001, "weight_decay": 0}' -o model.initialization_scale=8.0 -o data.train_size=150 &
pag train grid-search-sgd MSE_MLPEXTENDED_MNIST_GROK.json 150000 0 --run-name sgd-grid-100-8 --checkpoint-frequency 10000 -o 'optimizer={"name": "SGD", "lr": 0.001, "weight_decay": 0}' -o model.initialization_scale=8.0 -o data.train_size=100 &
pag train grid-search-sgd MSE_MLPEXTENDED_MNIST_GROK.json 150000 0 --run-name sgd-grid-50-8 --checkpoint-frequency 10000 -o 'optimizer={"name": "SGD", "lr": 0.001, "weight_decay": 0}' -o model.initialization_scale=8.0 -o data.train_size=50 &
pag train grid-search-sgd MSE_MLPEXTENDED_MNIST_GROK.json 150000 0 --run-name sgd-grid-10-8 --checkpoint-frequency 10000 -o 'optimizer={"name": "SGD", "lr": 0.001, "weight_decay": 0}' -o model.initialization_scale=8.0 -o data.train_size=10 &

# for i in {0..3}; do
#     echo "Starting run $i in the background..."
#     pag train slurm-test MSE_CNNEXTENDED_MNIST_GROK.json 10000 0 \
#         --seed $i \
#         --run-name "my-slurm-test-$i" \
#         --checkpoint-frequency 25000 &
# done

# 'wait' tells the Slurm script not to finish until all background tasks complete.
# Without this, Slurm would kill the job instantly after starting the loop!
wait

echo "All parallel training runs completed!"