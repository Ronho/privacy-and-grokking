from pathlib import Path
import random
import itertools

SCRIPT_DIR = Path(__file__).parent
configs = SCRIPT_DIR.parent / "configs"
command_file =  SCRIPT_DIR / "find_grokking.txt"
command_file.parent.mkdir(parents=True, exist_ok=True)

available_gpus = []
load_all_to_gpu = True
# Use this to avoid running out of memory.
shuffle = False
seed=4713

initialization_scale = [2.0, 3.0, 6.0, 9.0, 12.0]
weight_decay = [1.0, 0.5, 0.1, 0.05, 0.01, 0.001, 0.0001]
# train_size = [10_000, 5_000, 2_000, 1_000, 500]
train_size = [None, 10_000, 5_000, 2_000, 1_000, 500]


# Start of main script
random.seed(seed)
permutations = list(itertools.product(initialization_scale, weight_decay, train_size))

lines = []
configs_list = list(configs.glob("*.json"))
configs_list = [c for c in configs_list if c.name.startswith("-")]

def cmd(config, seed, name_prefix="", postfix=None):
    cmd_str = f"pag train find-grokking {config.name} 150000 --run-name {name_prefix}{config.stem[1:]} -o seed={seed} -o data.seed=4711"
    if load_all_to_gpu:
        cmd_str += " --load-all-to-gpu"
    if postfix is not None:
        cmd_str += postfix
    return cmd_str

# for config in configs_list:
#     for scale, decay, size in permutations:
#         seed = random.randint(0, 1000000)
#         lines.append(cmd(config, seed, postfix=f" -o data.mask.seed=4711 -o model.initialization_scale={scale} -o optimizer.weight_decay={decay} -o data.train_size={size}"))
for config in configs_list:
    for (key, val) in [("model.initialization_scale=", initialization_scale), ("optimizer.weight_decay=", weight_decay), ("data.train_size=", train_size)]:
        for v in val:
            seed = random.randint(0, 1000000)
            lines.append(cmd(config, seed, postfix=f" -o data.mask.seed=4711 -o {key}{v}"))

if shuffle:
    random.shuffle(lines)

N_gpus = len(available_gpus)
if N_gpus > 0:
    for idx, line in enumerate(lines):
        line = f"CUDA_VISIBLE_DEVICES={available_gpus[idx % N_gpus]} " + line
        lines[idx] = line

command_file.write_text("\n".join(lines), encoding="utf-8")