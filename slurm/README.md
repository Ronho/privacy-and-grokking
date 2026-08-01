# Commands:
1. Open Terminal and run `uv run poe mlflow-host`
1. Open Terminal and run `ssh -R 0.0.0.0:5050:127.0.0.1:5050 ronholzapfel@login.ai-lab.uni-luebeck.de`
    1. You should be on the login node now.
    1. Get the ip and note it down: `hostname -I`
    1. Then run: `socat TCP-LISTEN:5051,bind=0.0.0.0,reuseaddr,fork TCP:127.0.0.1:5050`
1. Open Terminal and run: `ssh ronholzapfel@login.ai-lab.uni-luebeck.de`
    1. Change folder: `cd privacy-and-grokking/slurm`
    1. Run `sbatch train.slurm`
    1. Watch the logs `tail -f logs/train_`
