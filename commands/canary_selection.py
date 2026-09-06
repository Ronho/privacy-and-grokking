import json
import random
import hashlib
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
configs = SCRIPT_DIR.parent / "configs"
command_file = SCRIPT_DIR / "canary_selection.txt"
command_file.parent.mkdir(parents=True, exist_ok=True)

available_gpus = []
load_all_to_gpu = True
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

def get_steps(config_path):
    name = config_path.name
    if "MADD" in name:
        return 50000
    if "MNIST" in name:
        return 150000
    return 150000

def get_deterministic_seed(*args, salt=seed):
    """Generates a deterministic integer seed from arguments to ensure idempotency."""
    s = str(salt) + "_" + "_".join(str(a) for a in args)
    return int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % 1000000

# Start of main script
random.seed(seed)

lines = []
configs_list = list(configs.glob("*.json"))
configs_list = [c for c in configs_list if not c.name.startswith("_")]
configs_list.sort(key=config_priority)

def cmd(config, seed, data_seed, model_index, canary_json, name_prefix="", postfix=None):
    steps = get_steps(config)
    cmd_str = f"pag train canary-selection-v1 {config.name} {steps} --run-name {name_prefix}{config.stem} -o seed={seed} -o data.seed={data_seed} -o data.mask.seed={data_seed} -o data.mask.model_index={model_index} -o data.canary='{canary_json}'"
    if load_all_to_gpu:
        cmd_str += " --load-all-to-gpu"
    if postfix is not None:
        cmd_str += postfix
    return cmd_str

for config in configs_list:
    for canary_name in canary_types:
        data_seed = get_deterministic_seed(config.name, canary_name, "data_seed")
        for i in range(num_repetitions):
            c_num = 226 if "MADD" in config.name else num_canaries
            canary_dict = {"name": canary_name, "num": c_num}
            if canary_name == "square_watermark":
                canary_dict["square_size"] = 5
            canary_json = json.dumps(canary_dict)
            run_seed = get_deterministic_seed(config.name, canary_name, i, "base")
            lines.append(cmd(config, run_seed, data_seed, i, canary_json, name_prefix=f"{canary_name.upper()}_"))

if shuffle:
    random.shuffle(lines)

N_gpus = len(available_gpus)
if N_gpus > 0:
    for idx, line in enumerate(lines):
        line = f"CUDA_VISIBLE_DEVICES={available_gpus[idx % N_gpus]} " + line
        lines[idx] = line

command_file.write_text("\n".join(lines), encoding="utf-8")
