import argparse
import asyncio
import json
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from privacy_and_grokking.utils.mlflow import TRACKING_URI

METRICS_TO_FETCH = [
    "epoch",
    "eval/train/accuracy",
    "eval/test/accuracy",
    "eval/nc/rnc1/train",
    "eval/nc/rnc1/test",
    "eval/nc/rnc1_train_mean/test",
    "eval/attack/mse_loss/auc",
    "eval/attack/mse_loss/tpr-at-fpr/1",
    "eval/attack/mse_loss/tpr-at-fpr/5",
    "eval/attack/mse_loss/tpr-at-fpr/10",
    "eval/attack/margin_distance_lf/global/auc",
    "eval/attack/margin_distance_lf/global/tpr-at-fpr/1",
    "eval/attack/margin_distance_lf/global/tpr-at-fpr/5",
    "eval/attack/margin_distance_lf/global/tpr-at-fpr/10",
]


async def fetch_run_metrics(client, run_id):
    """Fetch all specified metrics for a run asynchronously."""
    run_metrics_data = []
    for metric_name in METRICS_TO_FETCH:
        try:
            # Run the synchronous mlflow client call in a thread
            history = await asyncio.to_thread(client.get_metric_history, run_id, metric_name)
            for m in history:
                run_metrics_data.append(
                    {
                        "run_id": run_id,
                        "step": m.step,
                        "metric_name": metric_name,
                        "value": m.value,
                        "timestamp": m.timestamp,
                    }
                )
        except Exception:
            # Metric might not exist for this run
            pass
    return run_metrics_data


async def fetch_run_config(client, run_id):
    """Fetch training config artifact asynchronously."""
    try:
        # Download artifact in a thread to avoid blocking
        config_path = await asyncio.to_thread(
            client.download_artifacts, run_id, "training_config.json"
        )
        with open(config_path) as f:
            return json.load(f)
    except Exception:
        return {}


async def process_run(client, run):
    """Process a single run: fetch its metadata, config, and metrics."""
    run_id = run.info.run_id
    run_name = run.info.run_name

    # Fetch config and metrics concurrently
    config, metrics_data = await asyncio.gather(
        fetch_run_config(client, run_id), fetch_run_metrics(client, run_id)
    )

    # Extract required config parameters
    weight_decay = config.get("optimizer", {}).get("weight_decay")
    init_scale = config.get("model", {}).get("initialization_scale")
    train_size = config.get("data", {}).get("train_size")


    p = config.get("data", {}).get("mask", {}).get("p", 1.0)
    batch_size = config.get("batch_size", 128)

    dataset_size = int(train_size * p) if train_size is not None else 1

    # Add metadata to each metric row
    for row in metrics_data:
        row.update(
            {
                "run_name": run_name,
                "weight_decay": weight_decay,
                "initialization_scale": init_scale,
                "train_size": train_size,
                "p": p,
                "dataset_size": dataset_size,
                "batch_size": batch_size,
            }
        )

    return metrics_data


async def main():
    parser = argparse.ArgumentParser(description="Gather run metrics and config asynchronously.")
    parser.add_argument("experiment_name", type=str, help="Name of the MLflow experiment")
    parser.add_argument(
        "--tag", type=str, help="Optional tag to filter runs (e.g. 'key:value')", default=None
    )
    parser.add_argument(
        "--out", type=str, default="run_metrics.parquet", help="Output parquet file path"
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(TRACKING_URI)
    experiment = client.get_experiment_by_name(args.experiment_name)
    if not experiment:
        print(f"Experiment '{args.experiment_name}' not found.")
        return

    # Construct filter_string if tag is provided
    filter_string = ""
    if args.tag:
        if ":" in args.tag:
            k, v = args.tag.split(":", 1)
            filter_string = f"tags.`{k}` = '{v}'"
        else:
            print("Tag format should be key:value")
            return

    print(f"Searching for runs in experiment '{args.experiment_name}'...")
    runs = client.search_runs(experiment.experiment_id, filter_string=filter_string)
    print(f"Found {len(runs)} runs. Fetching data asynchronously...")

    # Process all runs concurrently
    tasks = [process_run(client, run) for run in runs]
    results = await asyncio.gather(*tasks)

    # Flatten the results list of lists
    all_metrics_data = [item for sublist in results for item in sublist]

    if not all_metrics_data:
        print("No metrics found for the specified runs.")
        return

    # Convert to DataFrame
    df_long = pd.DataFrame(all_metrics_data)

    # We have a long format DataFrame.
    # Let's pivot it to a wide format so each metric is a column (easier to analyze).
    index_cols = [
        "run_id",
        "run_name",
        "weight_decay",
        "initialization_scale",
        "train_size",
        "p",
        "dataset_size",
        "batch_size",
        "step",
        "timestamp",
    ]
    df_wide = df_long.pivot_table(
        index=index_cols, columns="metric_name", values="value"
    ).reset_index()

    import numpy as np

    batches_per_epoch = np.ceil(df_wide["dataset_size"] / df_wide["batch_size"])
    if "epoch" not in df_wide.columns:
        df_wide["epoch"] = df_wide["step"] / batches_per_epoch
    else:
        df_wide["epoch"] = df_wide["epoch"].fillna(df_wide["step"] / batches_per_epoch)

    # Save to Parquet
    out_path = Path(args.out)
    df_wide.to_parquet(out_path, index=False)
    print(f"Successfully saved metrics to {out_path}")
    print(f"DataFrame shape: {df_wide.shape}")


if __name__ == "__main__":
    asyncio.run(main())
