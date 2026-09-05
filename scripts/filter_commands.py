import argparse
import logging
import os
import shlex
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlflow
import pandas as pd
import requests

from privacy_and_grokking.cli import apply_overrides
from privacy_and_grokking.config import TrainConfig

CONFIG_DIR = Path(__file__).parent.parent / "configs"
CACHE_DIR = Path(__file__).parent.parent / "cache"


def fetch_json_artifact(tracking_uri: str, run_id: str, artifact_path: str) -> dict:
    base_uri = tracking_uri.rstrip("/")
    url = f"{base_uri}/get-artifact?path={artifact_path}&run_uuid={run_id}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def check_command_in_parquet(cfg: TrainConfig, run_name: str | None, df: pd.DataFrame) -> bool:
    """Checks whether the given config matches any run in the DataFrame."""
    if df.empty:
        return False

    cond = pd.Series(True, index=df.index)

    # 1. Match run_name if specified and available
    if run_name:
        if "run_name" in df.columns:
            cond &= df["run_name"] == run_name
        elif "tags.mlflow.runName" in df.columns:
            cond &= df["tags.mlflow.runName"] == run_name

    # 2. Seed matching
    if cfg.seed is not None:
        if "seed" in df.columns:
            cond &= df["seed"] == cfg.seed
        elif "params.seed" in df.columns:
            cond &= df["params.seed"].astype(str) == str(cfg.seed)

    # 3. Data seed matching
    d_seed = getattr(cfg.data, "seed", None)
    if d_seed is not None:
        if "data.seed" in df.columns:
            cond &= df["data.seed"] == d_seed
        elif "params.data.seed" in df.columns:
            cond &= df["params.data.seed"].astype(str) == str(d_seed)

    # 4. Data mask seed and model index matching
    m_cfg = getattr(cfg.data, "mask", None)
    if m_cfg is not None:
        m_seed = getattr(m_cfg, "seed", None)
        m_idx = getattr(m_cfg, "model_index", None)
        if m_seed is not None:
            if "data.mask.seed" in df.columns:
                cond &= df["data.mask.seed"] == m_seed
            elif "params.data.mask.seed" in df.columns:
                cond &= df["params.data.mask.seed"].astype(str) == str(m_seed)
        if m_idx is not None:
            if "data.mask.model_index" in df.columns:
                cond &= df["data.mask.model_index"] == m_idx
            elif "params.data.mask.model_index" in df.columns:
                cond &= df["params.data.mask.model_index"].astype(str) == str(m_idx)

    # 5. Hyperparameter checks if columns exist in parquet
    if "params.name" in df.columns and getattr(cfg, "name", None) is not None:
        cond &= df["params.name"] == cfg.name
    if "params.model_name" in df.columns and getattr(cfg.model, "name", None) is not None:
        cond &= df["params.model_name"] == cfg.model.name
    if "params.loss_function" in df.columns and getattr(cfg.loss, "name", None) is not None:
        cond &= df["params.loss_function"] == cfg.loss.name
    if "params.optimizer" in df.columns and getattr(cfg.optimizer, "name", None) is not None:
        cond &= df["params.optimizer"] == cfg.optimizer.name
    if (
        "params.weight_decay" in df.columns
        and getattr(cfg.optimizer, "weight_decay", None) is not None
    ):
        cond &= df["params.weight_decay"].astype(str) == str(cfg.optimizer.weight_decay)
    if "params.learning_rate" in df.columns and getattr(cfg.optimizer, "lr", None) is not None:
        cond &= df["params.learning_rate"].astype(str) == str(cfg.optimizer.lr)
    if (
        "params.initialization_scale" in df.columns
        and getattr(cfg.model, "initialization_scale", None) is not None
    ):
        cond &= df["params.initialization_scale"].astype(str) == str(cfg.model.initialization_scale)
    if "params.batch_size" in df.columns and getattr(cfg, "batch_size", None) is not None:
        cond &= df["params.batch_size"].astype(str) == str(cfg.batch_size)

    return bool(cond.any())


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Filter a commands file by removing commands whose runs have completed.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Filter using an offline parquet export (no MLflow connection needed):
  python scripts/filter_commands.py commands/canary_selection.txt -r cache/runs.parquet
  python scripts/filter_commands.py commands/canary_selection.txt -r cache/runs.parquet -o todo.txt

  # Dry run to see what remains without modifying files:
  python scripts/filter_commands.py commands/canary_selection.txt -r cache/runs.parquet --dry-run

  # Filter with custom status filter (default is FINISHED):
  python scripts/filter_commands.py commands/canary_selection.txt -r cache/runs.parquet -s ALL

  # Backwards-compatible live MLflow querying:
  python scripts/filter_commands.py canary-selection-v1 commands/canary_selection.txt
""",
    )
    parser.add_argument(
        "arg1",
        type=str,
        help="Path to commands text file OR MLflow experiment name",
    )
    parser.add_argument(
        "arg2",
        type=str,
        nargs="?",
        default=None,
        help="Path to commands text file (if arg1 was experiment name)",
    )
    parser.add_argument(
        "--runs-file",
        "-r",
        "--parquet",
        dest="runs_file",
        type=str,
        default=None,
        help="Path to parquet file with exported runs (e.g. from list_mlflow_runs.py)",
    )
    parser.add_argument(
        "--status",
        "-s",
        type=str,
        default="FINISHED",
        help="Run statuses to filter out (e.g. 'FINISHED', 'ALL'; default: FINISHED)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file (defaults to overwriting the input commands file)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview filtering results without modifying or writing any files",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5051"),
        help="MLflow tracking URI when querying MLflow directly (default: MLFLOW_TRACKING_URI)",
    )
    args = parser.parse_args()

    # Determine commands_file and experiment_name
    if args.arg2 is not None:
        experiment_name = args.arg1
        commands_file_path = args.arg2
    else:
        commands_file_path = args.arg1
        experiment_name = None

    commands_file = Path(commands_file_path)
    if not commands_file.exists():
        logging.error(f"Commands file not found: {commands_file_path}")
        sys.exit(1)

    output_file = args.output if args.output else commands_file

    # Check for parquet file
    runs_file_path = args.runs_file
    if not runs_file_path and experiment_name:
        candidates = [
            CACHE_DIR / f"{experiment_name}_runs.parquet",
            Path(f"{experiment_name}_runs.parquet"),
        ]
        for cand in candidates:
            if cand.exists():
                runs_file_path = str(cand)
                logging.info(f"Auto-detected runs parquet file: {runs_file_path}")
                break

    use_parquet = runs_file_path is not None and os.path.isfile(runs_file_path)

    if use_parquet:
        logging.info(f"Loading runs from parquet file: {runs_file_path}")
        runs_df = pd.read_parquet(runs_file_path)
        total_runs_in_file = len(runs_df)

        if args.status and args.status.upper() not in ("ALL", "ANY"):
            target_statuses = [s.strip().upper() for s in args.status.split(",") if s.strip()]
            if "status" in runs_df.columns:
                runs_df = runs_df[runs_df["status"].str.upper().isin(target_statuses)]
            logging.info(
                f"Filtered runs in parquet by status {target_statuses}: "
                f"{len(runs_df)} / {total_runs_in_file} matching runs."
            )
        else:
            logging.info(f"Using all {len(runs_df)} runs from parquet regardless of status.")
    else:
        if not experiment_name:
            logging.error(
                "Neither a valid --runs-file was specified nor was an experiment_name "
                "provided for live MLflow querying."
            )
            sys.exit(1)

        logging.info(
            f"Connecting to MLflow at {args.tracking_uri} for experiment '{experiment_name}'..."
        )
        mlflow.set_tracking_uri(args.tracking_uri)

        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                logging.info(
                    f"Experiment {experiment_name} not found. All commands will be run."
                )
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
                return fetch_json_artifact(
                    args.tracking_uri, run.info.run_id, "training_config.json"
                )
            except Exception:
                return None

        if runs:
            logging.info(f"Fetching configurations for {len(runs)} runs...")
            with ThreadPoolExecutor(max_workers=10) as executor:
                configs = list(executor.map(fetch_config, runs))
        else:
            configs = []

        run_configs = [cfg for cfg in configs if cfg is not None]

    with open(commands_file) as f:
        lines = f.readlines()

    filtered_commands = []
    skipped_count = 0

    for line in lines:
        line_s = line.strip()
        if not line_s or line_s.startswith("#"):
            continue

        parts = shlex.split(line_s)

        config_name = None
        overrides = []
        run_name = None

        for i, part in enumerate(parts):
            if part.endswith(".json") and not part.startswith("--"):
                config_name = part
            elif part in ("-o", "--override") and i + 1 < len(parts):
                overrides.append(parts[i + 1])
            elif part in ("--run-name", "-r") and i + 1 < len(parts):
                run_name = parts[i + 1]

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

            if use_parquet:
                already_run = check_command_in_parquet(cfg, run_name, runs_df)
            else:
                target_cfg_dict = cfg.model_dump(mode="json")
                already_run = any(existing_cfg == target_cfg_dict for existing_cfg in run_configs)

            if already_run:
                skipped_count += 1
            else:
                filtered_commands.append(line_s)
        except Exception as e:
            logging.error(f"Error processing command {line_s}: {e}")
            filtered_commands.append(line_s)

    logging.info(
        f"Processed {len(lines)} lines: {skipped_count} commands filtered out (already run), "
        f"{len(filtered_commands)} remaining to run."
    )

    if args.dry_run:
        logging.info("[DRY RUN] No files written.")
        if filtered_commands:
            logging.info("First 3 remaining commands:\n" + "\n".join(filtered_commands[:3]))
    else:
        with open(output_file, "w") as f:
            for cmd in filtered_commands:
                f.write(cmd + "\n")
        logging.info(f"Wrote remaining commands to: {output_file}")


if __name__ == "__main__":
    main()
