#!/bin/bash
#
#SBATCH --job-name=info_rmia
#SBATCH --output=logs/info_rmia_%A.log
#SBATCH --error=logs/info_rmia_%A.err
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --container-image=nvcr.io\#nvidia/pytorch:26.07-py3
#SBATCH --container-mounts=/home/ronholzapfel/privacy-and-grokking:/workspace,/home/ronholzapfel:/home/ronholzapfel
#SBATCH --container-workdir=/workspace
#SBATCH --container-writable

mkdir -p logs cache

echo "Starting job $SLURM_JOB_ID on $HOSTNAME"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# Start MPS to allow parallel CUDA usage on the single GPU (if available)
if command -v nvidia-cuda-mps-control &> /dev/null; then
    nvidia-cuda-mps-control -d
fi

CONCURRENT_JOBS=20

export MLFLOW_TRACKING_URI="file:///workspace/mlruns"
export PATH="$HOME/storage/personal/bin:$PATH"
export UV_CACHE_DIR="$HOME/storage/personal/.cache/uv"
export PYTHONUNBUFFERED=1
unset VIRTUAL_ENV

# Limit threads per process to avoid CPU thrashing with concurrent PyTorch jobs
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export VECLIB_MAXIMUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

uv pip install --system --break-system-packages -e .

echo "Starting info_rmia.py with $CONCURRENT_JOBS concurrent workers..."
if [ $# -gt 0 ]; then
    python3 scripts/info_rmia.py --workers $CONCURRENT_JOBS "$@"
else
    python3 scripts/info_rmia.py --experiment-name hyper-sweep --workers $CONCURRENT_JOBS
fi

echo "All jobs finished. Shutting down MPS."
if command -v nvidia-cuda-mps-control &> /dev/null; then
    echo quit | nvidia-cuda-mps-control
fi
echo "Done."
