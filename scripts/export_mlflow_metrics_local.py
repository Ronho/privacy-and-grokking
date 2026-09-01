import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "cache"))


def get_default_tracking_uri() -> str:
    """Returns the default tracking URI based on environment or available local storage."""
    env_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if env_uri:
        return env_uri
    workspace_mlruns = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "mlruns"))
    if os.path.isdir(workspace_mlruns):
        return f"file:///{workspace_mlruns.replace(os.sep, '/')}"
    return "http://localhost:5051"


def fetch_metric_for_run(client: MlflowClient, run_id: str, metric_key: str):
    try:
        history = client.get_metric_history(run_id, metric_key)
        return [
            {
                "run_id": run_id,
                "metric_name": metric_key,
                "value": m.value,
                "step": m.step,
                "timestamp": m.timestamp,
            }
            for m in history
        ]
    except Exception as e:
        tqdm.write(
            f"Warning: Failed to fetch history for run {run_id}, metric {metric_key}: {e}"
        )
        return []


def export_experiment(experiment_name: str, output_file: str, tracking_uri: str, history: bool, max_workers: int = 10):
    print(f"Connecting to MLflow tracking server at: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri)

    print(f"Searching for experiment '{experiment_name}'...")

    # Fetch experiments matching this name
    experiments = mlflow.search_experiments(filter_string=f"name = '{experiment_name}'")

    if not experiments:
        all_exps = mlflow.search_experiments()
        exp_names = [f"'{e.name}' (ID: {e.experiment_id})" for e in all_exps]
        print(f"Error: Experiment '{experiment_name}' not found.")
        if exp_names:
            print("Available experiments in tracking store:")
            for name in exp_names:
                print(f"  - {name}")
        else:
            print("No experiments found in the MLflow tracking store.")
        return

    experiment_ids = [exp.experiment_id for exp in experiments]

    if len(experiments) > 1:
        print(f"Found {len(experiments)} experiments with the name '{experiment_name}'.")
        print(f"Experiment IDs: {', '.join(experiment_ids)}")
        print("Will fetch and merge runs from ALL of these experiments.")
    else:
        print(f"Found experiment '{experiment_name}' (ID: {experiment_ids[0]})")

    # Fetch all runs as Pandas DataFrame
    print("Fetching runs...")
    runs_df = mlflow.search_runs(experiment_ids=experiment_ids)

    if runs_df.empty:
        print("No runs found for these experiments.")
        return

    print(f"Found {len(runs_df)} runs.")

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    existing_df = pd.DataFrame()
    if os.path.exists(output_file):
        print(f"Loading existing data from {output_file}...")
        try:
            existing_df = pd.read_parquet(output_file)
        except Exception as e:
            print(f"Warning: Could not read existing parquet file ({e}). A new file will be made.")
            existing_df = pd.DataFrame()

    existing_run_ids = set()
    if not existing_df.empty and "run_id" in existing_df.columns:
        existing_run_ids = set(existing_df["run_id"].unique())

    if not history:
        if not existing_df.empty:
            # Append and keep only the latest row per run_id
            runs_df = pd.concat([existing_df, runs_df]).drop_duplicates(
                subset=["run_id"], keep="last"
            )
        runs_df.to_parquet(output_file)
        print(f"Successfully saved runs summary to {output_file}")
        print(
            "Note: Contains only latest metric values. Use --history to get full metric history."
        )
        return

    runs_to_process = [row for _, row in runs_df.iterrows() if row["run_id"] not in existing_run_ids]
    skipped_count = len(runs_df) - len(runs_to_process)
    if skipped_count > 0:
        print(f"Skipping {skipped_count} runs already present in {output_file}.")

    if not runs_to_process:
        print("All runs are already present in the existing parquet file.")
        return

    metric_keys = [
        col.replace("metrics.", "") for col in runs_df.columns if col.startswith("metrics.")
    ]

    params_cols = [col for col in runs_df.columns if col.startswith("params.")]
    tag_cols = [col for col in runs_df.columns if col == "tags.mlflow.runName"]
    meta_df = runs_df[["run_id"] + params_cols + tag_cols]
    if "tags.mlflow.runName" in meta_df.columns:
        meta_df = meta_df.rename(columns={"tags.mlflow.runName": "run_name"})

    print(f"Fetching full metric history for {len(runs_to_process)} runs...")
    current_df = existing_df.copy()

    run_pbar = tqdm(runs_to_process, desc="Exporting runs", unit="run")
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for row in run_pbar:
                run_id = row["run_id"]
                run_name = str(row.get("tags.mlflow.runName") or run_id)
                if len(run_name) > 30:
                    run_name = run_name[:27] + "..."
                run_pbar.set_description(f"Run: {run_name}")

                # Only query metrics that were logged for this run to avoid unnecessary requests
                run_metric_keys = [k for k in metric_keys if pd.notna(row.get(f"metrics.{k}"))]
                if not run_metric_keys:
                    run_metric_keys = metric_keys

                run_metrics = []
                metric_pbar = tqdm(total=len(run_metric_keys), desc="Metrics", leave=False, unit="metric")

                future_to_metric = {
                    executor.submit(fetch_metric_for_run, client, run_id, k): k
                    for k in run_metric_keys
                }
                for future in as_completed(future_to_metric):
                    m_key = future_to_metric[future]
                    metric_pbar.set_postfix_str(m_key[:30])
                    res = future.result()
                    if res:
                        run_metrics.extend(res)
                    metric_pbar.update(1)
                metric_pbar.close()

                if run_metrics:
                    run_df = pd.DataFrame(run_metrics)
                    run_meta = meta_df[meta_df["run_id"] == run_id]
                    run_merged = pd.merge(run_df, run_meta, on="run_id", how="left")
                    if not current_df.empty:
                        current_df = pd.concat([current_df, run_merged], ignore_index=True)
                    else:
                        current_df = run_merged

                    # Atomically save to parquet after each completed run
                    tmp_output_file = f"{output_file}.tmp"
                    current_df.to_parquet(tmp_output_file)
                    os.replace(tmp_output_file, output_file)

                run_pbar.set_postfix(points=len(current_df))
    except KeyboardInterrupt:
        print(f"\nExport interrupted by user. Intermediate progress saved: {len(current_df)} rows in {output_file}.")
        return

    if not current_df.empty:
        print(f"Successfully saved {len(current_df)} metric data points to {output_file}")
    else:
        print("No metric history found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export MLflow metrics of all runs in an experiment to a Parquet file."
    )
    parser.add_argument("experiment_name", type=str, help="Name of the MLflow experiment")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output Parquet file path (e.g. data.parquet)",
    )
    parser.add_argument(
        "--uri",
        "-u",
        type=str,
        default=get_default_tracking_uri(),
        help="MLflow tracking URI (default: MLFLOW_TRACKING_URI or http://localhost:5051)",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Download the full metric history over time instead of just the final values.",
    )
    parser.add_argument(
        "--max-workers",
        "-w",
        type=int,
        default=10,
        help="Number of parallel worker threads for fetching metric history (default: 10).",
    )

    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(CACHE_DIR, f"{args.experiment_name}_mlflow_export.parquet")

    export_experiment(args.experiment_name, args.output, args.uri, args.history, args.max_workers)
