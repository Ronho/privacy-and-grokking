import sys
import json
import random
import itertools
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.append(str(REPO_ROOT))
from experiments.utils import get_deterministic_seed, config_priority, AVAILABLE_GPUS, LOAD_ALL_TO_GPU

configs = REPO_ROOT / "configs"

num_repetitions = 1
num_files = 1
seed = 4242
shuffle = False

random.seed(seed)
all_commands = []

configs_list = list(configs.glob("*.json"))
configs_list.sort(key=config_priority)

def cmd(
    config,
    seed,
    data_seed,
    model_index,
    name_prefix="",
    postfix=None,
):
    cmd_str = (
        f"pag train detailed-v1 {config.name} 150000 "
        f"--run-name {name_prefix}{config.stem} "
        f"-o seed={seed} -o data.seed={data_seed} -o data.mask.seed={data_seed} "
        f"-o data.mask.model_index={model_index} "
        f"-o metrics.optimizer_metrics_log_frequency=100"
    )
    if LOAD_ALL_TO_GPU:
        cmd_str += " --load-all-to-gpu"
    if postfix is not None:
        cmd_str += postfix
    return cmd_str

file_lines = {f: [] for f in range(num_files)}
start_file = 0

for i in range(num_repetitions):
    rep_commands = []
    for config in configs_list:
        data_seed = get_deterministic_seed(config.name, "data_seed", salt=seed)
        run_seed = get_deterministic_seed(config.name, i, "run_seed", salt=seed)

        rep_commands.append(
            cmd(
                config,
                run_seed,
                data_seed,
                i,
            )
        )

    if shuffle:
        random.shuffle(rep_commands)

    all_commands.extend(rep_commands)

    for idx, command in enumerate(rep_commands):
        file_lines[(start_file + idx) % num_files].append(command)
    start_file = (start_file + len(rep_commands)) % num_files

print(f"Total commands generated: {len(all_commands)}")

# Clean up any existing detailed_*.txt files
jobs_dir = SCRIPT_DIR / "jobs"
jobs_dir.mkdir(parents=True, exist_ok=True)
for old_file in jobs_dir.glob("detailed_*.txt"):
    old_file.unlink()

N_gpus = len(AVAILABLE_GPUS)
for f_idx in range(num_files):
    rep_lines = file_lines[f_idx]
    if N_gpus > 0:
        for idx, line in enumerate(rep_lines):
            rep_lines[idx] = f"CUDA_VISIBLE_DEVICES={AVAILABLE_GPUS[idx % N_gpus]} " + line

    cmd_file = jobs_dir / f"detailed_{f_idx}.txt"
    cmd_file.write_text("\n".join(rep_lines), encoding="utf-8")
    print(f"Wrote {len(rep_lines)} commands to {cmd_file.name}")
