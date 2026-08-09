from pathlib import Path

configs = Path("./configs")
command_file =  Path("./commands") / "reproduction_nc_grokking.txt"

available_gpus = [0, 1, 2, 3]
lines = []
for idx, config in enumerate(configs.glob("*.json")):
    gpu = available_gpus[idx % len(available_gpus)]
    lines.append(
        f"CUDA_VISIBLE_DEVICES={gpu} uv run pag train reproduction-nc-grokking {config.name} 150000 --run-name {config.stem}"
    )

command_file.write_text("\n".join(lines), encoding="utf-8")