import json
import random
import hashlib
import itertools
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
configs = SCRIPT_DIR.parent / "configs"
command_file = SCRIPT_DIR / "hyper_sweep.txt"
command_file.parent.mkdir(parents=True, exist_ok=True)

available_gpus = []
load_all_to_gpu = True
num_canaries = 100
canary_type = "label_noise"
initialization_scale = [2.0, 3.0, 6.0, 9.0, 12.0]
weight_decay = [1.0, 0.1, 0.01, 0.001, 0.0001]
train_size = [10_000, 5_000, 2_000, 1_000, 500]
train_size_madd = [10_170, 4_972, 2_034, 904, 452]
num_repetitions = 6
seed = 4712
shuffle = False


def get_deterministic_seed(*args, salt=seed):
    """Generates a deterministic integer seed from arguments to ensure idempotency."""
    s = str(salt) + "_" + "_".join(str(a) for a in args)
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16) % 1000000


# Start of main script
random.seed(seed)

lines = {i: [] for i in range(num_repetitions)}


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
        return 100000
    return 100000


configs_list = list(configs.glob("*.json"))
configs_list = [
    c for c in configs_list if not c.name.startswith("_") and not c.name.startswith("NO_")
]
configs_list.sort(key=config_priority)


def cmd(
    config,
    seed,
    data_seed,
    model_index,
    canary_json,
    scale,
    decay,
    size,
    steps=None,
    name_prefix="",
    postfix=None,
):
    if steps is None:
        steps = get_steps(config)
    cmd_str = (
        f"pag train hyper-sweep-v1 {config.name} {steps} "
        f"--run-name {name_prefix}{config.stem} "
        f"-o seed={seed} -o data.seed={data_seed} -o data.mask.seed={data_seed} "
        f"-o data.mask.model_index={model_index} -o data.canary='{canary_json}' "
        f"-o model.initialization_scale={scale} -o optimizer.weight_decay={decay} "
        f"-o data.train_size={size}"
    )
    if load_all_to_gpu:
        cmd_str += " --load-all-to-gpu"
    if postfix is not None:
        cmd_str += postfix
    return cmd_str


for config in configs_list:
    steps = get_steps(config)
    current_train_sizes = train_size_madd if "MADD" in config.name else train_size
    for scale, decay, size in itertools.product(
        initialization_scale, weight_decay, current_train_sizes
    ):
        data_seed = get_deterministic_seed(config.name, scale, decay, size, "data_seed")
        for i in range(num_repetitions):
            c_num = 226 if "MADD" in config.name else num_canaries
            canary_dict = {"name": f"{canary_type}", "num": c_num}
            canary_json = json.dumps(canary_dict)
            run_seed = get_deterministic_seed(config.name, scale, decay, size, i, "run_seed")
            lines[i].append(
                cmd(
                    config,
                    run_seed,
                    data_seed,
                    i,
                    canary_json,
                    scale=scale,
                    decay=decay,
                    size=size,
                    steps=steps,
                    name_prefix=f"{scale}_{decay}_{size}_",
                )
            )

N_gpus = len(available_gpus)
for i in range(num_repetitions):
    rep_lines = lines[i]
    if shuffle:
        random.shuffle(rep_lines)
    
    if N_gpus > 0:
        for idx, line in enumerate(rep_lines):
            line = f"CUDA_VISIBLE_DEVICES={available_gpus[idx % N_gpus]} " + line
            rep_lines[idx] = line

    cmd_file = SCRIPT_DIR / f"hyper_sweep_{i}.txt"
    cmd_file.write_text("\n".join(rep_lines), encoding="utf-8")
