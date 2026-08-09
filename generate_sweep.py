import itertools
from pathlib import Path

def main():
    amounts = [20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000, 10000, 25000, 50000]
    weight_decays = [1, 0.1, 0.01, 0.001, 0.0001]
    init_scales = [1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0]
    losses = ["mse", "cross_entropy"]
    devices = [0, 1, 3]

    commands = []
    
    for loss, wd, amount, init_scale in itertools.product(losses, weight_decays, amounts, init_scales):
        
        p_values = [None]
        if amount == 50000:
            p_values.append(1.0)
            
        for p_val in p_values:
            run_name = f"SWEEP_{loss}_{wd}_{amount}_{init_scale}"
            if p_val is not None:
                run_name += f"_p{p_val}"
                
            cuda_device = devices[len(commands) % len(devices)]
            
            cmd = (
                f"CUDA_VISIBLE_DEVICES={cuda_device} uv run pag train multi-dim-sweep "
                f"MSE_MLPEXTENDED_MNIST_GROK.json 150000 0 --seed 0 "
                f"--run-name {run_name} --checkpoint-frequency 5000 "
                f"-o data.train_size={amount} "
                f"-o optimizer.weight_decay={wd} "
                f"-o model.initialization_scale={init_scale} "
                f"-o loss.name={loss} "
                f"-o data.mask.name=balanced_stratified"
            )
            
            if p_val is not None:
                cmd += f" -o data.mask.p={p_val}"
                
            commands.append(cmd)

    output_file = Path("commands/sweep_commands.txt")
    output_file.parent.mkdir(exist_ok=True)
    output_file.write_text("\n".join(commands) + "\n")
    print(f"Generated {len(commands)} commands in {output_file}")

if __name__ == "__main__":
    main()
