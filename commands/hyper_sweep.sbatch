#!/bin/bash
#
#SBATCH --job-name=hyper_sweep
#SBATCH --output=logs/hyper_sweep_%A_%a.log
#SBATCH --error=logs/hyper_sweep_%A_%a.err
#SBATCH --array=0-5
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

echo "Starting job $SLURM_JOB_ID (Array Task $SLURM_ARRAY_TASK_ID) on $HOSTNAME"
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


echo "Starting $CONCURRENT_JOBS concurrent jobs via xargs for task $SLURM_ARRAY_TASK_ID..."

cat commands/hyper_sweep_${SLURM_ARRAY_TASK_ID}.txt | xargs -d '\n' -P $CONCURRENT_JOBS -I {} bash -c "sleep \$((RANDOM % 10)); {}"

echo "All jobs finished. Shutting down MPS."
echo quit | nvidia-cuda-mps-control
echo "Done."
