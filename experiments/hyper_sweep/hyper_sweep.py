import sys
import json
import random
import itertools
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.append(str(REPO_ROOT))
from experiments.utils import get_deterministic_seed, config_priority, get_steps, AVAILABLE_GPUS, LOAD_ALL_TO_GPU

configs = REPO_ROOT / "configs"
command_file = SCRIPT_DIR / "jobs" / "hyper_sweep.txt"
command_file.parent.mkdir(parents=True, exist_ok=True)

num_canaries = 100
canary_type = "label_noise"
initialization_scale = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
weight_decay = [10.0, 1.0, 0.1, 0.01, 0.001, 0.0001]
train_size = [50_000, 25_000, 10_000, 5_000, 2_000, 1_000, 500]
train_size_madd = [10_170, 4_972, 2_034, 904, 452]
num_repetitions = 6
num_files = 5
seed = 4712
shuffle = False
sweep_mode = "one_fixed"  # "one_fixed" (2D slices, 1 param fixed) or "two_fixed" (1D ablations, 2 params fixed)


# Start of main script
random.seed(seed)

all_commands = []


def get_config_defaults(config_path):
    """Extract standard baseline parameters from the config JSON."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # 1. initialization_scale (defaults to 1.0 if not specified, e.g. for MADD Transformer)
    scale = cfg.get("model", {}).get("initialization_scale", 1.0)
    if scale is None:
        scale = 1.0

    # 2. weight_decay
    decay = cfg.get("optimizer", {}).get("weight_decay", 0.0)

    # 3. train_size
    train_size_cfg = cfg.get("data", {}).get("train_size")
    if train_size_cfg is None:
        if "MADD" in config_path.name:
            train_size_cfg = 113*90
        elif "MNIST" in config_path.name:
            train_size_cfg = 50_000
        else:
            raise ValueError(f"No train_size could be determined for {config_path.name}")

    return float(scale), float(decay), int(train_size_cfg)


def match_value_in_list(val, lst, tol=1e-6):
    """Snaps a value to the exact element in list if within tolerance."""
    for item in lst:
        if abs(item - val) <= tol:
            return item
    return val


def get_parameter_combinations(
    scales, decays, sizes, def_scale, def_decay, def_size, mode="one_fixed"
):
    """
    Generate parameter combinations.
    mode='one_fixed': 2D slices where at least one parameter is fixed to config baseline.
    mode='two_fixed': 1D slices where two parameters are fixed to config baseline (only 1 varies).
    """
    def_scale = match_value_in_list(def_scale, scales)
    def_decay = match_value_in_list(def_decay, decays)
    def_size = match_value_in_list(def_size, sizes)

    combinations = set()

    if mode == "one_fixed":
        # Slice 1: initialization_scale is fixed to default
        for d, s in itertools.product(decays, sizes):
            combinations.add((def_scale, d, s))

        # Slice 2: weight_decay is fixed to default
        for sc, s in itertools.product(scales, sizes):
            combinations.add((sc, def_decay, s))

        # Slice 3: train_size is fixed to default
        for sc, d in itertools.product(scales, decays):
            combinations.add((sc, d, def_size))

    elif mode == "two_fixed":
        # Vary scale (decay and size fixed)
        for sc in scales:
            combinations.add((sc, def_decay, def_size))
        # Vary decay (scale and size fixed)
        for d in decays:
            combinations.add((def_scale, d, def_size))
        # Vary size (scale and decay fixed)
        for s in sizes:
            combinations.add((def_scale, def_decay, s))
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Deterministic sorting by scale, decay, size
    return sorted(list(combinations), key=lambda x: (x[0], x[1], x[2]))


configs_list = list(configs.glob("*.json"))
configs_list = [
    c for c in configs_list#if c.name.startswith("GROK_")
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
        f"pag train hyper-sweep-v2 {config.name} {steps} "
        f"--run-name {name_prefix}{config.stem} "
        f"-o seed={seed} -o data.seed={data_seed} -o data.mask.seed={data_seed} "
        f"-o data.mask.model_index={model_index} -o data.canary='{canary_json}' "
        f"-o model.initialization_scale={scale} -o optimizer.weight_decay={decay} "
        f"-o data.train_size={size}"
    )
    if LOAD_ALL_TO_GPU:
        cmd_str += " --load-all-to-gpu"
    if postfix is not None:
        cmd_str += postfix
    return cmd_str


total_runs_per_rep = 0
config_combos = []
for config in configs_list:
    steps = get_steps(config)
    current_train_sizes = train_size_madd if "MADD" in config.name else train_size
    def_scale, def_decay, def_size = get_config_defaults(config)
    param_combos = get_parameter_combinations(
        initialization_scale,
        weight_decay,
        current_train_sizes,
        def_scale,
        def_decay,
        def_size,
        mode=sweep_mode,
    )
    print(
        f"Config {config.name}: {len(param_combos)} combinations "
        f"(defaults: scale={def_scale}, decay={def_decay}, train_size={def_size})"
    )
    total_runs_per_rep += len(param_combos)
    config_combos.append((config, steps, param_combos))

file_lines = {f: [] for f in range(num_files)}
start_file = 0

for i in range(num_repetitions):
    rep_commands = []
    for config, steps, param_combos in config_combos:
        for scale, decay, size in param_combos:
            data_seed = get_deterministic_seed(config.name, scale, decay, size, "data_seed", salt=seed)
            c_num = 226 if "MADD" in config.name else num_canaries
            canary_dict = {"name": f"{canary_type}", "num": c_num}
            canary_json = json.dumps(canary_dict)
            run_seed = get_deterministic_seed(config.name, scale, decay, size, i, "run_seed", salt=seed)
            rep_commands.append(
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

    if shuffle:
        random.shuffle(rep_commands)

    all_commands.extend(rep_commands)

    # Distribute this repetition's commands evenly across the files
    for idx, command in enumerate(rep_commands):
        file_lines[(start_file + idx) % num_files].append(command)
    start_file = (start_file + len(rep_commands)) % num_files

print(f"Total commands generated: {len(all_commands)}")

# Clean up any existing hyper_sweep_*.txt files in SCRIPT_DIR / "jobs"
for old_file in (SCRIPT_DIR / "jobs").glob("hyper_sweep_*.txt"):
    old_file.unlink()

N_gpus = len(AVAILABLE_GPUS)
for f_idx in range(num_files):
    rep_lines = file_lines[f_idx]
    if N_gpus > 0:
        for idx, line in enumerate(rep_lines):
            rep_lines[idx] = f"CUDA_VISIBLE_DEVICES={AVAILABLE_GPUS[idx % N_gpus]} " + line

    cmd_file = SCRIPT_DIR / "jobs" / f"hyper_sweep_{f_idx}.txt"
    cmd_file.write_text("\n".join(rep_lines), encoding="utf-8")
    print(f"Wrote {len(rep_lines)} commands to {cmd_file.name}")
