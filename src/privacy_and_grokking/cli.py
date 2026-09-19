import ast
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, TypeVar

import typer
from pydantic import BaseModel
from typer import Typer

from privacy_and_grokking.config import TrainConfig
from privacy_and_grokking.training import RestartConfig
from privacy_and_grokking.training import train as training
from privacy_and_grokking.utils import Logger

app = Typer(name="Privacy and Grokking CLI", pretty_exceptions_enable=False)

T = TypeVar("T", bound=BaseModel)


def _dicts_differ(
    input_dict: dict[str, Any], validated_dict: dict[str, Any], path: str = ""
) -> str | None:
    for k, v in input_dict.items():
        if k not in validated_dict:
            return f"{path}{k}"
        if isinstance(v, dict) and isinstance(validated_dict[k], dict):
            res = _dicts_differ(v, validated_dict[k], path=f"{path}{k}.")
            if res:
                return res
    return None


def _parse_value(val_str: str) -> Any:
    if val_str.lower() == "true":
        return True
    if val_str.lower() == "false":
        return False
    if val_str.lower() in ("none", "null"):
        return None
    try:
        return json.loads(val_str)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(val_str)
        except (ValueError, SyntaxError):
            return val_str


def apply_overrides(cfg: T, overrides: list[str] | None) -> T:
    if not overrides:
        return cfg

    cfg_dict = cfg.model_dump(mode="json")
    parsed_overrides = []

    for override in overrides:
        if "=" not in override:
            raise typer.BadParameter(
                f"Invalid override format: {override}. Expected key.path=value."
            )

        path, val_str = override.split("=", 1)
        keys = path.split(".")

        val = _parse_value(val_str)
        parsed_overrides.append((path, keys))

        curr_dict = cfg_dict
        for key in keys[:-1]:
            if key not in curr_dict or curr_dict[key] is None:
                curr_dict[key] = {}
            curr_dict = curr_dict[key]

        curr_dict[keys[-1]] = val

    validated = type(cfg).model_validate(cfg_dict)
    validated_dict = validated.model_dump(mode="json")

    for path, keys in parsed_overrides:
        curr = validated_dict
        for key in keys[:-1]:
            if key not in curr or curr[key] is None:
                raise typer.BadParameter(f"Unknown config field: {path}")
            curr = curr[key]
        if keys[-1] not in curr:
            raise typer.BadParameter(f"Unknown config field: {path}")

    return validated


CONFIG_DIR = Path(__file__).parent.parent.parent / "configs"


@app.command()
def train(
    exp_name: str,
    model: str,
    total_steps: int,
    run_name: str | None = None,
    profile: bool = typer.Option(False, "--profile", help="Enable PyTorch profiler."),
    overrides: list[str] | None = typer.Option(
        None, "--override", "-o", help="Override config fields (e.g. data.batch_size=32)"
    ),
    load_all_to_gpu: bool = typer.Option(
        False, "--load-all-to-gpu", help="Load the whole dataset to the GPU"
    ),
):
    if profile:
        os.environ["PAG_PROFILE"] = "1"
    cfg = TrainConfig.model_validate_json((CONFIG_DIR / model).read_bytes())
    cfg = apply_overrides(cfg, overrides)
    training(
        exp_name=exp_name,
        total_steps=total_steps,
        cfg=cfg,
        run_name=run_name,
        load_all_to_gpu=load_all_to_gpu,
    )


@app.command()
def restart(
    exp_name: str,
    run_id: str,
    checkpoint: int,
    total_steps: int,
):
    cfg = RestartConfig(run_id=run_id, checkpoint=checkpoint)
    training(
        exp_name=exp_name,
        total_steps=total_steps,
        cfg=cfg,
        run_name="",
    )


def _handle(line):
    line = line.strip()
    if not line or line.startswith("#"):
        return
    with Logger() as logger:
        logger.info("Processing command.", command=line)
        subprocess.run(line, shell=True, stdin=subprocess.DEVNULL)


@app.command()
def process(path: str, num_workers: int):
    with open(path) as f, ThreadPoolExecutor(max_workers=num_workers) as executor:
        executor.map(_handle, f)


if __name__ == "__main__":
    app()
