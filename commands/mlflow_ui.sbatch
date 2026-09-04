#!/bin/bash
#
#SBATCH --job-name=mlflow_ui
#SBATCH --output=logs/mlflow_ui_%j.log
#SBATCH --error=logs/mlflow_ui_%j.err
#SBATCH --partition=defq
#SBATCH --nodelist=BCM-SM-A100-03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --container-image=nvcr.io\#nvidia/pytorch:26.07-py3
#SBATCH --container-mounts=/home/ronholzapfel/privacy-and-grokking:/workspace,/home/ronholzapfel:/home/ronholzapfel
#SBATCH --container-workdir=/workspace

mkdir -p logs

echo "=========================================================="
echo "MLflow UI is running on node: $HOSTNAME"
echo "=========================================================="
echo ""
echo "To view the UI locally, run this command on your LAPTOP:"
echo "ssh -N -L 5051:$HOSTNAME:5051 ronholzapfel@login.ai-lab.uni-luebeck.de"
echo ""
echo "Then open http://localhost:5051 in your browser."
echo "=========================================================="

export PATH="$HOME/storage/personal/bin:$PATH"
export UV_CACHE_DIR="$HOME/storage/personal/.cache/uv"

# Only install if needed, otherwise just start it
uv pip install --system --break-system-packages -e .

uv run mlflow ui --backend-store-uri file:///workspace/mlruns --host 0.0.0.0 --port 5051
