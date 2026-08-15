#!/bin/bash
#
#SBATCH --job-name=find_grokking
#SBATCH --output=logs/find_grokking_%j.log
#SBATCH --error=logs/find_grokking_%j.err
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

mkdir -p logs

echo "Starting job $SLURM_JOB_ID on $HOSTNAME"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi

nvidia-cuda-mps-control -d

CONCURRENT_JOBS=20

export MLFLOW_TRACKING_URI="file:///workspace/mlruns"
export PATH="$HOME/storage/personal/bin:$PATH"
export UV_CACHE_DIR="$HOME/storage/personal/.cache/uv"
export PYTHONUNBUFFERED=1

# Limit threads per process to avoid CPU thrashing with concurrent PyTorch jobs
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export VECLIB_MAXIMUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

uv pip install --system --break-system-packages -e .


echo "Starting $CONCURRENT_JOBS concurrent jobs via xargs..."

cat commands/find_grokking.txt | xargs -d '\n' -P $CONCURRENT_JOBS -I {} bash -c "sleep \$((RANDOM % 10)); {}"

echo "All jobs finished. Shutting down MPS."
echo quit | nvidia-cuda-mps-control
echo "Done."
