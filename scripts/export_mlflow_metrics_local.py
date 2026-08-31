import argparse
import os

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

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


def export_experiment(experiment_name: str, output_file: str, tracking_uri: str, history: bool):
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

    print("Fetching full metric history for all runs and metrics (this may take a while)...")
    all_metrics = []

    for _, row in runs_df.iterrows():
        run_id = row["run_id"]

        if run_id in existing_run_ids:
            print(f"Skipping run {run_id} as it is already in the existing parquet file.")
            continue

        # Extract all metric column names
        metric_keys = [
            col.replace("metrics.", "") for col in runs_df.columns if col.startswith("metrics.")
        ]

        for metric_key in metric_keys:
            try:
                # Fetch full history of this metric for the run
                metric_history = client.get_metric_history(run_id, metric_key)
                for m in metric_history:
                    all_metrics.append({
                        "run_id": run_id,
                        "metric_name": metric_key,
                        "value": m.value,
                        "step": m.step,
                        "timestamp": m.timestamp,
                    })
            except Exception as e:
                print(
                    f"Warning: Failed to fetch history for run {run_id}, metric {metric_key}: {e}"
                )

    if all_metrics:
        history_df = pd.DataFrame(all_metrics)
        # Join with params and tags from main runs_df for downstream analysis
        params_cols = [col for col in runs_df.columns if col.startswith("params.")]
        tag_cols = [col for col in runs_df.columns if col == "tags.mlflow.runName"]
        meta_df = runs_df[["run_id"] + params_cols + tag_cols]
        final_df = pd.merge(history_df, meta_df, on="run_id", how="left")

        if "tags.mlflow.runName" in final_df.columns:
            final_df = final_df.rename(columns={"tags.mlflow.runName": "run_name"})

        if not existing_df.empty:
            final_df = pd.concat([existing_df, final_df], ignore_index=True)

        final_df.to_parquet(output_file)
        print(f"Successfully saved {len(final_df)} metric data points to {output_file}")
    else:
        if not existing_df.empty:
            print(f"No new metric history found. Existing data has {len(existing_df)} rows.")
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

    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(CACHE_DIR, f"{args.experiment_name}_mlflow_export.parquet")

    export_experiment(args.experiment_name, args.output, args.uri, args.history)
