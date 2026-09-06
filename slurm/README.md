# Commands:
1. Open Terminal and run `uv run poe mlflow-host`
1. Open Terminal and run `ssh -R 0.0.0.0:5050:127.0.0.1:5050 ronholzapfel@login.ai-lab.uni-luebeck.de`
    1. You should be on the login node now.
    1. Get the ip and note it down: `hostname -I`
    1. Then run: `socat TCP-LISTEN:5051,bind=0.0.0.0,reuseaddr,fork TCP:127.0.0.1:5050`
1. Open Terminal and run: `ssh ronholzapfel@login.ai-lab.uni-luebeck.de`
    1. If missing, copy the datasets: `scp -r cache ronholzapfel@login.ai-lab.uni-luebeck.de:~/privacy-and-grokking/`
    scp ronholzapfel@login.ai-lab.uni-luebeck.de:~/privacy-and-grokking/cache/reproduction-nc-grokking-v1_mlflow_export.parquet cache/reproduction-nc-grokking-v1_mlflow_export.parquet
    1. Change folder: `cd privacy-and-grokking/slurm`
    1. Run `sbatch train.slurm`
    1. Watch the logs `tail -f logs/train_`


# Download Datasets:
1. Run locally:
```bash
uv run python -c "
from privacy_and_grokking.datasets.sets.cifar10 import CIFAR10Config
from privacy_and_grokking.datasets.sets.mnist import MNISTConfig
# This will download the datasets to /workspace/cache if they don't exist yet.
print('Checking CIFAR10...')
CIFAR10Config()()
print('Checking MNIST...')
MNISTConfig()()
"
```

srun --jobid=19462 --pty nvidia-smi

Note: When the following error occurs, make sure that you created the experiment and provided a proper path for the files in the creation process, i.e. file:///workspace/mlruns/canary-selection

```
raise MlflowException(
mlflow.exceptions.MlflowException: When an mlflow-artifacts URI was supplied, the tracking URI must be a valid http or https URI, but it was currently set to file:///workspace/mlruns. Perhaps you forgot to set the tracking URI to the running MLflow server. To set the tracking URI, use either of the following methods:
1. Set the MLFLOW_TRACKING_URI environment variable to the desired tracking URI. `export MLFLOW_TRACKING_URI=http://localhost:5000`
2. Set the tracking URI programmatically by calling `mlflow.set_tracking_uri`. `mlflow.set_tracking_uri('http://localhost:5000')`
```

rclone copy ":sftp,host=login.ai-lab.uni-luebeck.de,user=ronholzapfel,ask_password=true:privacy-and-grokking/mlruns" D:\privacy-and-grokking\cache\mlruns -P

```bash
sbatch --exclude=BCM-DGX-H100-2 commands/info_rmia.sbatch --experiment-name canary-selection
```
```bash
scontrol show node | awk '/^NodeName=/{n=substr($1,10); rm=0; fm=0; cc=0; cg=0; ac=0; ag=0} /^[ \t]*RealMemory=/{for(i=1;i<=NF;i++){if($i~/^RealMemory=/) rm=substr($i,12); if($i~/^FreeMem=/) fm=substr($i,9)}} /^[ \t]*CfgTRES=/{if(match($1,/cpu=[0-9]+/)) cc=substr($1,RSTART+4,RLENGTH-4); if(match($1,/gres\/gpu=[0-9]+/)) cg=substr($1,RSTART+9,RLENGTH-9)} /^[ \t]*AllocTRES=/{if(match($1,/cpu=[0-9]+/)) ac=substr($1,RSTART+4,RLENGTH-4); if(match($1,/gres\/gpu=[0-9]+/)) ag=substr($1,RSTART+9,RLENGTH-9); used=(fm>0?rm-fm:0); printf "%-18s | CPU: %3d / %-3d | GPU Avail: %d  [Alloc: %d, Tot: %d] | RAM: %4.0f / %-4.0f GiB (Free: %4.0f GiB)\n", n, ac, cc, cg-ag, ag, cg, used/1024, rm/1024, fm/1024}'
```

uv run python scripts/filter_commands.py commands/canary-selection.txt -r cache/canary-selection-v1_runs_keep.parquet