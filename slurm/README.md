# Commands:
1. Open Terminal and run `uv run poe mlflow-host`
1. Open Terminal and run `ssh -R 0.0.0.0:5050:127.0.0.1:5050 ronholzapfel@login.ai-lab.uni-luebeck.de`
    1. You should be on the login node now.
    1. Get the ip and note it down: `hostname -I`
    1. Then run: `socat TCP-LISTEN:5051,bind=0.0.0.0,reuseaddr,fork TCP:127.0.0.1:5050`
1. Open Terminal and run: `ssh ronholzapfel@login.ai-lab.uni-luebeck.de`
    1. If missing, copy the datasets: `scp -r cache ronholzapfel@login.ai-lab.uni-luebeck.de:~/privacy-and-grokking/`
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