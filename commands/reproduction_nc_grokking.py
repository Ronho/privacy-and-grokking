from pathlib import Path
import random

SCRIPT_DIR = Path(__file__).parent
configs = SCRIPT_DIR.parent / "configs"
command_file =  SCRIPT_DIR / "reproduction_nc_grokking.txt"
command_file.parent.mkdir(parents=True, exist_ok=True)

available_gpus = []
load_all_to_gpu = True
# Use this to avoid running out of memory.
shuffle = True
num_repetitions=5
seed=4711


# Start of main script
random.seed(seed)

lines = []
configs_list = list(configs.glob("*.json"))
configs_list = [c for c in configs_list if not c.name.startswith("_") and c.name.startswith("+")] # TODO: Remove "and c.name.startswith("+")" once finished.

def cmd(config, seed, data_seed, model_index, name_prefix="", postfix=None):
    # TODO: Remove the [1:] from config.stem once finished.
    cmd_str = f"pag train reproduction-nc-grokking {config.name} 150000 --run-name {name_prefix}{config.stem[1:]} -o seed={seed} -o data.seed={data_seed} -o data.mask.seed={data_seed} -o data.mask.model_index={i}"
    if load_all_to_gpu:
        cmd_str += " --load-all-to-gpu"
    if postfix is not None:
        cmd_str += postfix
    return cmd_str

for config in configs_list:
    data_seed = random.randint(0, 1000000)
    for i in range(num_repetitions):
        seed = random.randint(0, 1000000)
        lines.append(cmd(config, seed, data_seed, i))

        # None grokking training i.e. no initialization scale, full train size.
        seed = random.randint(0, 1000000)
        lines.append(
            cmd(config, seed, data_seed, i, "NO_", postfix=" --override model.initialization_scale=None -o data.train_size=None")
        )

if shuffle:
    random.shuffle(lines)

N_gpus = len(available_gpus)
if N_gpus > 0:
    for idx, line in enumerate(lines):
        line = f"CUDA_VISIBLE_DEVICES={available_gpus[idx % N_gpus]} " + line
        lines[idx] = line

command_file.write_text("\n".join(lines), encoding="utf-8")