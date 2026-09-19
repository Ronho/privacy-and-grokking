# SLURM Cluster Usage

This guide contains specific commands and setups required to run the project on the AI-Lab cluster using SLURM.

## MLflow Port Forwarding & Running Jobs
To track experiments on the cluster directly to your local MLflow instance, you need to set up port forwarding.

1. Open your local terminal and run:
   ```bash
   uv run poe mlflow-host
   ```
2. Open another local terminal and reverse-forward the MLflow port to the login node:
   ```bash
   ssh -R 0.0.0.0:5050:127.0.0.1:5050 ronholzapfel@login.ai-lab.uni-luebeck.de
   ```
3. You are now on the login node. Get the IP and note it down:
   ```bash
   hostname -I
   ```
4. Listen and forward via `socat`:
   ```bash
   socat TCP-LISTEN:5051,bind=0.0.0.0,reuseaddr,fork TCP:127.0.0.1:5050
   ```
5. In yet another terminal, SSH to the login node:
   ```bash
   ssh ronholzapfel@login.ai-lab.uni-luebeck.de
   ```
6. If missing, copy the datasets (or specific MLflow exports):
   ```bash
   scp -r cache ronholzapfel@login.ai-lab.uni-luebeck.de:~/privacy-and-grokking/
   # Or for specific files:
   scp ronholzapfel@login.ai-lab.uni-luebeck.de:~/privacy-and-grokking/cache/reproduction-nc-grokking-v1_mlflow_export.parquet cache/reproduction-nc-grokking-v1_mlflow_export.parquet
   ```
7. Run the training:
   ```bash
   cd privacy-and-grokking/slurm
   sbatch train.slurm
   ```
8. Watch the logs:
   ```bash
   tail -f logs/train_...
   ```

## Dataset Preparation
To ensure datasets are downloaded locally before syncing to the cluster:
```python
uv run python -c "
from privacy_and_grokking.datasets.sets.cifar10 import CIFAR10Config
from privacy_and_grokking.datasets.sets.mnist import MNISTConfig
print('Checking CIFAR10...')
CIFAR10Config()()
print('Checking MNIST...')
MNISTConfig()()
"
```

## Useful SLURM Commands

**Check node resources (CPU/GPU/RAM):**
```bash
scontrol show node | awk '/^NodeName=/{n=substr($1,10); rm=0; fm=0; cc=0; cg=0; ac=0; ag=0} /^[ \t]*RealMemory=/{for(i=1;i<=NF;i++){if($i~/^RealMemory=/) rm=substr($i,12); if($i~/^FreeMem=/) fm=substr($i,9)}} /^[ \t]*CfgTRES=/{if(match($1,/cpu=[0-9]+/)) cc=substr($1,RSTART+4,RLENGTH-4); if(match($1,/gres\/gpu=[0-9]+/)) cg=substr($1,RSTART+9,RLENGTH-9)} /^[ \t]*AllocTRES=/{if(match($1,/cpu=[0-9]+/)) ac=substr($1,RSTART+4,RLENGTH-4); if(match($1,/gres\/gpu=[0-9]+/)) ag=substr($1,RSTART+9,RLENGTH-9); used=(fm>0?rm-fm:0); printf "%-18s | CPU: %3d / %-3d | GPU Avail: %d  [Alloc: %d, Tot: %d] | RAM: %4.0f / %-4.0f GiB (Free: %4.0f GiB)\n", n, ac, cc, cg-ag, ag, cg, used/1024, rm/1024, fm/1024}'
```

**Get an interactive GPU shell:**
```bash
srun --jobid=19462 --pty nvidia-smi
```

**Submit batch with exclusion:**
```bash
sbatch --exclude=BCM-DGX-H100-2 commands/info_rmia.sbatch --experiment-name canary-selection
```

## Troubleshooting MLflow

> [!WARNING]
> **MlflowException: tracking URI must be a valid http or https URI...**
> 
> Note: When this error occurs, make sure that you created the experiment and provided a proper path for the files in the creation process, i.e., `file:///workspace/mlruns/canary-selection`.

If you need to manually fetch MLflow data from the cluster using `rclone`:
```bash
rclone copy ":sftp,host=login.ai-lab.uni-luebeck.de,user=ronholzapfel,ask_password=true:privacy-and-grokking/mlruns" D:\privacy-and-grokking\cache\mlruns -P
```

## Running Scripts

Filter commands:
```bash
uv run python scripts/filter_commands.py commands/canary-selection.txt -r cache/canary-selection-v1_runs_keep.parquet
```
