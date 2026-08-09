import json
from pathlib import Path

# This gets the directory where this script lives (the 'commands' folder)
SCRIPT_DIR = Path(__file__).parent
configs = SCRIPT_DIR.parent / "configs"
command_file = SCRIPT_DIR / "canary_selection.txt"

available_gpus = [0, 1, 2, 3]
num_canaries = 50
canary_types = [
    "uniform_noise",
    "square_watermark",
    "gaussian_noise",
    "label_noise",
    "ood_natural"
]

lines = []

idx = 0
configs_list = list(configs.glob("*.json"))

for config in configs_list:
    for canary_name in canary_types:
        gpu = available_gpus[idx % len(available_gpus)]

        canary_json = json.dumps({"name": canary_name, "num": num_canaries})
        
        run_name = f"{config.stem}_{canary_name}"
        
        lines.append(
            f"CUDA_VISIBLE_DEVICES={gpu} uv run pag train canary-selection {config.name} 150000 --run-name {run_name} -o data.canary='{canary_json}'"
        )
        idx += 1

# None grokking training i.e. no initialization scale, full train size, no mask.
for config in configs_list:
    for canary_name in canary_types:
        gpu = available_gpus[idx % len(available_gpus)]

        canary_json = json.dumps({"name": canary_name, "num": num_canaries})
        
        run_name = f"NO{config.stem}_{canary_name}"
        
        lines.append(
            f"CUDA_VISIBLE_DEVICES={gpu} uv run pag train canary-selection {config.name} 150000 --run-name {run_name} -o data.canary='{canary_json}' --override model.initialization_scale=None --override data.train_size=None --override data.mask=None"
        )
        idx += 1

command_file.write_text("\n".join(lines), encoding="utf-8")
print(f"Generated {len(lines)} commands in {command_file}")
