import sys
import random
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.append(str(REPO_ROOT))
from experiments.utils import get_deterministic_seed, config_priority, AVAILABLE_GPUS, LOAD_ALL_TO_GPU

configs = REPO_ROOT / "configs"
command_file = SCRIPT_DIR / "jobs" / "reproduction_nc_grokking.txt"
command_file.parent.mkdir(parents=True, exist_ok=True)

num_repetitions=6
seed=4711
shuffle = False

# Start of main script
random.seed(seed)

lines = []
configs_list = list(configs.glob("*.json"))
configs_list = [c for c in configs_list if not c.name.startswith("_")]
configs_list.sort(key=config_priority)

def cmd(config, seed, data_seed, model_index, name_prefix="", postfix=None):
    cmd_str = f"pag train reproduction-nc-grokking-v1 {config.name} 150000 --run-name {name_prefix}{config.stem} -o seed={seed} -o data.seed={data_seed} -o data.mask.seed={data_seed} -o data.mask.model_index={model_index}"
    if LOAD_ALL_TO_GPU:
        cmd_str += " --load-all-to-gpu"
    if postfix is not None:
        cmd_str += postfix
    return cmd_str

for config in configs_list:
    data_seed = get_deterministic_seed(config.name, "data_seed", salt=seed)
    for i in range(num_repetitions):
        run_seed = get_deterministic_seed(config.name, i, "base", salt=seed)
        lines.append(cmd(config, run_seed, data_seed, i))

if shuffle:
    random.shuffle(lines)

N_gpus = len(AVAILABLE_GPUS)
if N_gpus > 0:
    for idx, line in enumerate(lines):
        line = f"CUDA_VISIBLE_DEVICES={AVAILABLE_GPUS[idx % N_gpus]} " + line
        lines[idx] = line

command_file.write_text("\n".join(lines), encoding="utf-8")
