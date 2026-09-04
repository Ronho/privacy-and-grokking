import argparse
import logging
import shlex
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlflow
import requests

from privacy_and_grokking.cli import apply_overrides
from privacy_and_grokking.config import TrainConfig

CONFIG_DIR = Path(__file__).parent.parent / "configs"


def fetch_json_artifact(tracking_uri: str, run_id: str, artifact_path: str) -> dict:
    base_uri = tracking_uri.rstrip("/")
    url = f"{base_uri}/get-artifact?path={artifact_path}&run_uuid={run_id}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name", type=str)
    parser.add_argument("commands_file", type=str)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file (defaults to overwriting the input file)",
    )
    parser.add_argument("--tracking-uri", type=str, default="http://localhost:5051")
    args = parser.parse_args()

    output_file = args.output if args.output else args.commands_file

    mlflow.set_tracking_uri(args.tracking_uri)

    try:
        experiment = mlflow.get_experiment_by_name(args.experiment_name)
        if experiment is None:
            logging.info(f"Experiment {args.experiment_name} not found. All commands will be run.")
            runs = []
        else:
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id], output_format="list"
            )
    except Exception as e:
        logging.info(f"Could not connect or fetch runs: {e}. All commands will be run.")
        runs = []

    def fetch_config(run):
        try:
            return fetch_json_artifact(args.tracking_uri, run.info.run_id, "training_config.json")
        except Exception:
            return None

    if runs:
        logging.info(f"Fetching configurations for {len(runs)} runs...")
        with ThreadPoolExecutor(max_workers=10) as executor:
            configs = list(executor.map(fetch_config, runs))
    else:
        configs = []

    run_configs = []
    for run, config in zip(runs, configs):
        if config is not None:
            run_configs.append(config)

    commands_file = Path(args.commands_file)
    if not commands_file.exists():
        logging.error(f"File not found: {args.commands_file}")
        sys.exit(1)

    filtered_commands = []

    with open(commands_file) as f:
        lines = f.readlines()

    for line in lines:
        line_s = line.strip()
        if not line_s or line_s.startswith("#"):
            continue

        parts = shlex.split(line_s)

        config_name = None
        overrides = []

        for i, part in enumerate(parts):
            if part.endswith(".json") and not part.startswith("--"):
                config_name = part
            elif part in ("-o", "--override") and i + 1 < len(parts):
                overrides.append(parts[i + 1])

        if config_name is None:
            filtered_commands.append(line_s)
            continue

        try:
            base_cfg_path = CONFIG_DIR / config_name
            if not base_cfg_path.exists():
                logging.warning(f"Config {config_name} not found, keeping command.")
                filtered_commands.append(line_s)
                continue

            cfg = TrainConfig.model_validate_json(base_cfg_path.read_bytes())
            cfg = apply_overrides(cfg, overrides)
            target_cfg_dict = cfg.model_dump(mode="json")

            already_run = False
            for existing_cfg in run_configs:
                if existing_cfg == target_cfg_dict:
                    already_run = True
                    break

            if not already_run:
                filtered_commands.append(line_s)
        except Exception as e:
            logging.error(f"Error processing command {line_s}: {e}")
            filtered_commands.append(line_s)

    with open(output_file, "w") as f:
        for cmd in filtered_commands:
            f.write(cmd + "\n")

    logging.info(
        f"Filtered {len(lines) - len(filtered_commands)} commands. {len(filtered_commands)} remaining to run."
    )


if __name__ == "__main__":
    main()
