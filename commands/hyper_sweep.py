from pathlib import Path
import random
import itertools

SCRIPT_DIR = Path(__file__).parent
configs = SCRIPT_DIR.parent / "configs"
command_file =  SCRIPT_DIR / "hyper_sweep.txt"
command_file.parent.mkdir(parents=True, exist_ok=True)

available_gpus = []
load_all_to_gpu = True
# Use this to avoid running out of memory.
shuffle = True
seed=4714

# Hyper Parameters
initialization_scale = [None, 2.0, 4.0, 8.0, 12.0, 16.0]
weight_decay = [1, 0.1, 0.01, 0.001, 0.0001]
train_size = [20_000, 2_000, 200, 20]

# Start of main script
random.seed(seed)

lines = []
configs_list = list(configs.glob("*.json"))
configs_list = [c for c in configs_list if not c.name.startswith("_") and c.name.startswith("+")] # TODO: Remove "and c.name.startswith("+")" once finished.

def cmd(config, seed, name_prefix="", postfix=None):
    # TODO: Remove the [1:] from config.stem once finished.
    cmd_str = f"pag train hyper-sweep {config.name} 150000 --run-name {name_prefix}{config.stem[1:]} -o seed={seed} -o data.seed={seed}"
    if load_all_to_gpu:
        cmd_str += " --load-all-to-gpu"
    if postfix is not None:
        cmd_str += postfix
    return cmd_str

for config in configs_list:
    for scale, decay, size in itertools.product(initialization_scale, weight_decay, train_size):
        seed = random.randint(0, 1000000)
        base = "data.mask=None" if size is None else f"-o data.mask.seed={seed}"

        lines.append(cmd(config, seed, postfix=f" {base} -o model.initialization_scale={scale} -o optimizer.weight_decay={decay} -o data.train_size={size}"))

if shuffle:
    random.shuffle(lines)

N_gpus = len(available_gpus)
if N_gpus > 0:
    for idx, line in enumerate(lines):
        line = f"CUDA_VISIBLE_DEVICES={available_gpus[idx % N_gpus]} " + line
        lines[idx] = line

command_file.write_text("\n".join(lines), encoding="utf-8")