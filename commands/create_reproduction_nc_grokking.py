from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
configs = SCRIPT_DIR.parent / "configs"
command_file =  SCRIPT_DIR / "reproduction_nc_grokking.txt"
command_file.parent.mkdir(parents=True, exist_ok=True)

available_gpus = [0, 1, 2, 3]
lines = []
configs_list = list(configs.glob("*.json"))

for idx, config in enumerate(configs_list):
    gpu = available_gpus[idx % len(available_gpus)]
    lines.append(
        f"CUDA_VISIBLE_DEVICES={gpu} uv run pag train reproduction-nc-grokking {config.name} 150000 --run-name {config.stem}"
    )

# None grokking training i.e. no initialization scale, full train size.
for idx, config in enumerate(configs_list):
    gpu = available_gpus[(idx + len(configs_list)) % len(available_gpus)]
    lines.append(
        f"CUDA_VISIBLE_DEVICES={gpu} uv run pag train reproduction-nc-grokking {config.name} 150000 --run-name NO{config.stem} --override model.initialization_scale=None --override data.train_size=None --override data.mask=None"
    )

command_file.write_text("\n".join(lines), encoding="utf-8")