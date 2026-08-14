import json
import random
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
configs = SCRIPT_DIR.parent / "configs"
command_file = SCRIPT_DIR / "canary_selection.txt"
command_file.parent.mkdir(parents=True, exist_ok=True)

available_gpus = []
load_all_to_gpu = True
num_canaries = 50
canary_types = [
    "uniform_noise",
    "square_watermark",
    "gaussian_noise",
    "label_noise",
    "ood_natural"
]
num_repetitions=5
seed=4712
shuffle = False


# Start of main script
random.seed(seed)

lines = []
configs_list = list(configs.glob("*.json"))
configs_list = [c for c in configs_list if not c.name.startswith("_")]

def cmd(config, seed, name_prefix="", postfix=None):
    cmd_str = f"pag train canary-selection {config.name} 150000 --run-name {name_prefix}{config.stem} -o seed={seed} -o data.seed={seed}"
    if load_all_to_gpu:
        cmd_str += " --load-all-to-gpu"
    if postfix is not None:
        cmd_str += postfix
    return cmd_str

for config in configs_list:
    for canary_name in canary_types:
        canary_json = json.dumps({"name": canary_name, "num": num_canaries})
        seed = random.randint(0, 1000000)
        lines.append(cmd(config, seed, name_prefix=f"{canary_name.upper()}_", postfix=f" -o data.canary='{canary_json}' -o data.mask.seed={seed}"))

        # None grokking training i.e. no initialization scale, full train size, no mask.
        seed = random.randint(0, 1000000)
        lines.append(cmd(config, seed, name_prefix=f"{canary_name.upper()}_NO_", postfix=f" -o data.canary='{canary_json}' -o model.initialization_scale=None -o data.train_size=None -o data.mask=None"))

if shuffle:
    random.shuffle(lines)

N_gpus = len(available_gpus)
if N_gpus > 0:
    for idx, line in enumerate(lines):
        line = f"CUDA_VISIBLE_DEVICES={available_gpus[idx % N_gpus]} " + line
        lines[idx] = line

command_file.write_text("\n".join(lines), encoding="utf-8")
