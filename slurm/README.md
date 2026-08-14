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