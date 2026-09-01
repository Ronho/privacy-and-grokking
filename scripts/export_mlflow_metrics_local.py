import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import sys
import urllib.parse

import pandas as pd
from tqdm import tqdm
import yaml

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


def parse_tracking_dir(uri_or_path: str) -> str:
    """Extracts a local directory path from a URI or file path."""
    if uri_or_path.startswith("file://"):
        p = uri_or_path[7:]
        if sys.platform == "win32" and len(p) >= 3 and p[0] == "/" and p[2] == ":":
            p = p[1:]
        return os.path.normpath(p)
    return os.path.normpath(uri_or_path)


def parse_single_run_dir(r_path: str, history: bool):
    """Directly parses a single MLflow run directory from disk without MLflow middleware."""
    run_id = os.path.basename(r_path)
    run_name = run_id

    # 1. Read meta.yaml for run_name if available
    r_meta_file = os.path.join(r_path, "meta.yaml")
    if os.path.isfile(r_meta_file):
        try:
            with open(r_meta_file, "r", encoding="utf-8", errors="ignore") as f:
                r_meta = yaml.safe_load(f)
            if isinstance(r_meta, dict) and r_meta.get("run_name"):
                run_name = str(r_meta["run_name"])
        except Exception:
            pass

    # 2. Read tags (mlflow.runName takes precedence)
    tdir = os.path.join(r_path, "tags")
    if os.path.isdir(tdir):
        rn_file = os.path.join(tdir, "mlflow.runName")
        if os.path.isfile(rn_file):
            try:
                with open(rn_file, "r", encoding="utf-8", errors="ignore") as fp:
                    val = fp.read().strip()
                    if val:
                        run_name = val
            except Exception:
                pass

    # 3. Read params
    params = {}
    pdir = os.path.join(r_path, "params")
    if os.path.isdir(pdir):
        for pf in os.listdir(pdir):
            if pf.startswith("."):
                continue
            pf_path = os.path.join(pdir, pf)
            if os.path.isfile(pf_path):
                try:
                    with open(pf_path, "r", encoding="utf-8", errors="ignore") as fp:
                        clean_key = urllib.parse.unquote(pf)
                        params[f"params.{clean_key}"] = fp.read().strip()
                except Exception:
                    pass

    # 4. Read metrics directly from files
    records = []
    mdir = os.path.join(r_path, "metrics")
    if os.path.isdir(mdir):
        for root, _, files in os.walk(mdir):
            for fname in files:
                if fname.startswith("."):
                    continue
                fpath = os.path.join(root, fname)
                raw_rel = os.path.relpath(fpath, mdir).replace("\\", "/")
                metric_name = urllib.parse.unquote(raw_rel)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as fp:
                        lines = [l for l in fp.readlines() if l.strip()]
                    if not lines:
                        continue
                    if not history:
                        lines = [lines[-1]]  # only keep the latest step if not history

                    for line in lines:
                        parts = line.split()
                        if len(parts) >= 2:
                            ts = int(parts[0])
                            val = float(parts[1])
                            step = int(parts[2]) if len(parts) >= 3 else 0
                            records.append({
                                "run_id": run_id,
                                "metric_name": metric_name,
                                "value": val,
                                "step": step,
                                "timestamp": ts,
                                **params,
                                "run_name": run_name,
                            })
                except Exception:
                    pass

    return run_id, run_name, records


def export_experiment_direct(
    experiment_name: str, output_file: str, tracking_dir: str, history: bool, max_workers: int = 16
):
    """Direct, ultra-fast parallel filesystem extraction of MLflow runs, completely bypassing MLflow middleware."""
    print(f"Direct filesystem mode: scanning '{tracking_dir}' (bypassing MLflow middleware)...")
    if not os.path.isdir(tracking_dir):
        print(f"Error: Directory '{tracking_dir}' not found.")
        return

    # Find matching experiments by scanning meta.yaml in each subdirectory
    matched_exps = []
    all_found_exps = []
    for entry in os.listdir(tracking_dir):
        if entry.startswith(".") or entry == ".trash":
            continue
        exp_path = os.path.join(tracking_dir, entry)
        if not os.path.isdir(exp_path):
            continue
        meta_file = os.path.join(exp_path, "meta.yaml")
        if os.path.isfile(meta_file):
            try:
                with open(meta_file, "r", encoding="utf-8", errors="ignore") as f:
                    meta = yaml.safe_load(f)
                if isinstance(meta, dict):
                    name = str(meta.get("name", ""))
                    exp_id = str(meta.get("experiment_id", entry))
                    all_found_exps.append(f"'{name}' (ID: {exp_id})")
                    if name == experiment_name or exp_id == experiment_name:
                        matched_exps.append((exp_id, name, exp_path))
            except Exception:
                pass

    if not matched_exps:
        print(f"Error: Experiment '{experiment_name}' not found in '{tracking_dir}'.")
        if all_found_exps:
            print("Available experiments:")
            for e in all_found_exps:
                print(f"  - {e}")
        return

    exp_ids = [e[0] for e in matched_exps]
    print(f"Found experiment '{experiment_name}' (ID(s): {', '.join(exp_ids)}).")

    # Discover all active runs
    run_dirs = []
    for exp_id, name, exp_path in matched_exps:
        for r_entry in os.listdir(exp_path):
            if r_entry.startswith(".") or r_entry == ".trash":
                continue
            r_path = os.path.join(exp_path, r_entry)
            if not os.path.isdir(r_path):
                continue
            r_meta_file = os.path.join(r_path, "meta.yaml")
            if os.path.isfile(r_meta_file):
                try:
                    with open(r_meta_file, "r", encoding="utf-8", errors="ignore") as f:
                        r_meta = yaml.safe_load(f)
                    if isinstance(r_meta, dict) and r_meta.get("lifecycle_stage") == "deleted":
                        continue
                except Exception:
                    pass
                run_dirs.append(r_path)

    if not run_dirs:
        print("No runs found in the experiment.")
        return

    print(f"Found {len(run_dirs)} runs.")
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    # Check for existing parquet file to resume
    existing_df = pd.DataFrame()
    if os.path.exists(output_file):
        print(f"Loading existing data from {output_file}...")
        try:
            existing_df = pd.read_parquet(output_file)
        except Exception as e:
            print(f"Warning: Could not read existing parquet file ({e}). Starting fresh.")
            existing_df = pd.DataFrame()

    existing_run_ids = set()
    if not existing_df.empty and "run_id" in existing_df.columns:
        existing_run_ids = set(existing_df["run_id"].unique())

    runs_to_process = [r for r in run_dirs if os.path.basename(r) not in existing_run_ids]
    skipped_count = len(run_dirs) - len(runs_to_process)
    if skipped_count > 0:
        print(f"Skipping {skipped_count} runs already present in {output_file}.")

    if not runs_to_process:
        print("All runs are already exported.")
        return

    print(f"Extracting metrics from {len(runs_to_process)} runs in parallel ({max_workers} worker threads)...")
    current_df = existing_df.copy()

    run_pbar = tqdm(total=len(runs_to_process), desc="Exporting runs", unit="run")
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_run = {
                executor.submit(parse_single_run_dir, r, history): r for r in runs_to_process
            }
            for future in as_completed(future_to_run):
                run_id, run_name, records = future.result()
                disp_name = run_name if len(run_name) <= 25 else run_name[:22] + "..."
                run_pbar.set_description(f"Run: {disp_name}")

                if records:
                    run_df = pd.DataFrame(records)
                    if not current_df.empty:
                        current_df = pd.concat([current_df, run_df], ignore_index=True)
                    else:
                        current_df = run_df

                    # Atomic incremental save after each completed run
                    tmp_output = f"{output_file}.tmp"
                    current_df.to_parquet(tmp_output)
                    os.replace(tmp_output, output_file)

                run_pbar.update(1)
                run_pbar.set_postfix(points=len(current_df))
    except KeyboardInterrupt:
        print(f"\nExport interrupted by user. Intermediate progress saved: {len(current_df)} rows in {output_file}.")
        return

    if not current_df.empty:
        print(f"Successfully saved {len(current_df)} metric data points to {output_file}")
    else:
        print("No metric data found.")


def export_experiment_mlflow(
    experiment_name: str, output_file: str, tracking_uri: str, history: bool, max_workers: int = 16
):
    """Fallback export using the MLflow Client over HTTP/remote server."""
    import mlflow
    from mlflow.tracking import MlflowClient

    print(f"Connecting to MLflow tracking server at: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri)

    print(f"Searching for experiment '{experiment_name}'...")
    experiments = mlflow.search_experiments(filter_string=f"name = '{experiment_name}'")

    if not experiments:
        all_exps = mlflow.search_experiments()
        exp_names = [f"'{e.name}' (ID: {e.experiment_id})" for e in all_exps]
        print(f"Error: Experiment '{experiment_name}' not found.")
        if exp_names:
            print("Available experiments in tracking store:")
            for name in exp_names:
                print(f"  - {name}")
        return

    experiment_ids = [exp.experiment_id for exp in experiments]
    print(f"Found experiment '{experiment_name}' (ID(s): {', '.join(experiment_ids)})")

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
            existing_df = pd.DataFrame()

    existing_run_ids = set()
    if not existing_df.empty and "run_id" in existing_df.columns:
        existing_run_ids = set(existing_df["run_id"].unique())

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

    def fetch_metric(run_id, metric_key):
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
        except Exception:
            return []

    run_pbar = tqdm(runs_to_process, desc="Exporting runs", unit="run")
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for row in run_pbar:
                run_id = row["run_id"]
                run_name = str(row.get("tags.mlflow.runName") or run_id)
                disp_name = run_name if len(run_name) <= 25 else run_name[:22] + "..."
                run_pbar.set_description(f"Run: {disp_name}")

                run_metric_keys = [k for k in metric_keys if pd.notna(row.get(f"metrics.{k}"))]
                if not run_metric_keys:
                    run_metric_keys = metric_keys

                run_metrics = []
                metric_pbar = tqdm(total=len(run_metric_keys), desc="Metrics", leave=False, unit="metric")
                future_to_metric = {
                    executor.submit(fetch_metric, run_id, k): k for k in run_metric_keys
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


def export_experiment(
    experiment_name: str,
    output_file: str,
    tracking_uri: str,
    history: bool,
    max_workers: int = 16,
):
    tracking_uri = tracking_uri.strip()
    if tracking_uri.startswith("http://") or tracking_uri.startswith("https://"):
        export_experiment_mlflow(
            experiment_name, output_file, tracking_uri, history, max_workers
        )
    else:
        tracking_dir = parse_tracking_dir(tracking_uri)
        export_experiment_direct(
            experiment_name, output_file, tracking_dir, history, max_workers
        )


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
        help="MLflow tracking URI or directory (default: MLFLOW_TRACKING_URI, ./mlruns, or http://localhost:5051)",
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
        default=16,
        help="Number of parallel worker threads for reading runs/metrics (default: 16).",
    )

    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(CACHE_DIR, f"{args.experiment_name}_mlflow_export.parquet")

    export_experiment(args.experiment_name, args.output, args.uri, args.history, args.max_workers)
