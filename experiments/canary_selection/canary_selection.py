import sys
import json
import random
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.append(str(REPO_ROOT))
from experiments.utils import get_deterministic_seed, config_priority, get_steps, AVAILABLE_GPUS, LOAD_ALL_TO_GPU

configs = REPO_ROOT / "configs"
command_file = SCRIPT_DIR / "jobs" / "canary_selection.txt"
command_file.parent.mkdir(parents=True, exist_ok=True)

num_canaries = 100
canary_types = [
    "uniform_noise",
    "square_watermark",
    "gaussian_noise",
    "label_noise",
    "ood_natural"
]
num_repetitions=6
seed=4712
shuffle=False

# Start of main script
random.seed(seed)

lines = []
configs_list = list(configs.glob("*.json"))
configs_list = [c for c in configs_list if not c.name.startswith("_")]
configs_list.sort(key=config_priority)

def cmd(config, seed, data_seed, model_index, canary_json, name_prefix="", postfix=None):
    steps = get_steps(config)
    cmd_str = f"pag train canary-selection-v1 {config.name} {steps} --run-name {name_prefix}{config.stem} -o seed={seed} -o data.seed={data_seed} -o data.mask.seed={data_seed} -o data.mask.model_index={model_index} -o data.canary='{canary_json}'"
    if LOAD_ALL_TO_GPU:
        cmd_str += " --load-all-to-gpu"
    if postfix is not None:
        cmd_str += postfix
    return cmd_str

for config in configs_list:
    for canary_name in canary_types:
        data_seed = get_deterministic_seed(config.name, canary_name, "data_seed", salt=seed)
        for i in range(num_repetitions):
            c_num = 226 if "MADD" in config.name else num_canaries
            canary_dict = {"name": canary_name, "num": c_num}
            if canary_name == "square_watermark":
                canary_dict["square_size"] = 5
            canary_json = json.dumps(canary_dict)
            run_seed = get_deterministic_seed(config.name, canary_name, i, "base", salt=seed)
            lines.append(cmd(config, run_seed, data_seed, i, canary_json, name_prefix=f"{canary_name.upper()}_"))

if shuffle:
    random.shuffle(lines)

N_gpus = len(AVAILABLE_GPUS)
if N_gpus > 0:
    for idx, line in enumerate(lines):
        line = f"CUDA_VISIBLE_DEVICES={AVAILABLE_GPUS[idx % N_gpus]} " + line
        lines[idx] = line

command_file.write_text("\n".join(lines), encoding="utf-8")
