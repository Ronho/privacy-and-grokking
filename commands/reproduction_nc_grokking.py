from pathlib import Path
import random
import hashlib

SCRIPT_DIR = Path(__file__).parent
configs = SCRIPT_DIR.parent / "configs"
command_file =  SCRIPT_DIR / "reproduction_nc_grokking.txt"
command_file.parent.mkdir(parents=True, exist_ok=True)

available_gpus = []
load_all_to_gpu = True
# Use this to avoid running out of memory.
shuffle = False
num_repetitions=6
seed=4711


def get_deterministic_seed(*args, salt=seed):
    """Generates a deterministic integer seed from arguments to ensure idempotency."""
    s = str(salt) + "_" + "_".join(str(a) for a in args)
    return int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % 1000000

def config_priority(config_path):
    name = config_path.name.lstrip("+_")
    if "MNIST" in name and "MSE" in name and "MLP" in name:
        return 0
    if "MNIST" in name and "CE" in name and "MLP" in name:
        return 1
    if "VIT" in name:
        return 2
    if "MADD" in name and "CE" in name:
        return 3
    if "MADD" in name and "MSE" in name:
        return 4
    return 99

# Start of main script
random.seed(seed)

lines = []
configs_list = list(configs.glob("*.json"))
configs_list = [c for c in configs_list if not c.name.startswith("_")]
configs_list.sort(key=config_priority)


def cmd(config, seed, data_seed, model_index, name_prefix="", postfix=None):
    cmd_str = f"pag train reproduction-nc-grokking-v1 {config.name} 150000 --run-name {name_prefix}{config.stem} -o seed={seed} -o data.seed={data_seed} -o data.mask.seed={data_seed} -o data.mask.model_index={i}"
    if load_all_to_gpu:
        cmd_str += " --load-all-to-gpu"
    if postfix is not None:
        cmd_str += postfix
    return cmd_str

for config in configs_list:
    data_seed = get_deterministic_seed(config.name, "data_seed")
    for i in range(num_repetitions):
        run_seed = get_deterministic_seed(config.name, i, "base")
        lines.append(cmd(config, run_seed, data_seed, i))

if shuffle:
    random.shuffle(lines)

N_gpus = len(available_gpus)
if N_gpus > 0:
    for idx, line in enumerate(lines):
        line = f"CUDA_VISIBLE_DEVICES={available_gpus[idx % N_gpus]} " + line
        lines[idx] = line

command_file.write_text("\n".join(lines), encoding="utf-8")