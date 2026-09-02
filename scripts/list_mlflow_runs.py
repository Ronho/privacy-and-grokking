import argparse
import datetime
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

import mlflow
import pandas as pd
from mlflow.entities import ViewType

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def fetch_run_config(
    run_row, tracking_uri: str, client: mlflow.tracking.MlflowClient | None = None
) -> dict | None:
    """Attempts to find and load training_config.json for a given run row."""
    run_id = str(run_row.get("run_id", ""))
    artifact_uri = str(run_row.get("artifact_uri", ""))

    # 1. Direct artifact_uri path if file:// based
    if artifact_uri.startswith("file://"):
        local_path = artifact_uri[7:]
        if os.name == "nt" and len(local_path) >= 3 and local_path[0] == "/" and local_path[2] == ":":
            local_path = local_path[1:]
        cfg_file = os.path.join(local_path, "training_config.json")
        if os.path.isfile(cfg_file):
            try:
                with open(cfg_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

    # 2. Tracking URI directory if file:// based
    if tracking_uri.startswith("file://"):
        base_dir = tracking_uri[7:]
        if os.name == "nt" and len(base_dir) >= 3 and base_dir[0] == "/" and base_dir[2] == ":":
            base_dir = base_dir[1:]
        exp_id = str(run_row.get("experiment_id", ""))
        exp_name = str(run_row.get("experiment_name", ""))
        candidates = [
            os.path.join(base_dir, exp_name, run_id, "artifacts", "training_config.json"),
            os.path.join(base_dir, exp_id, run_id, "artifacts", "training_config.json"),
            os.path.join(base_dir, run_id, "artifacts", "training_config.json"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                try:
                    with open(c, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception:
                    pass

    # 3. Relative workspace directories (cache/mlruns or ./mlruns)
    for base in [
        os.path.join(SCRIPT_DIR, "..", "cache", "mlruns"),
        os.path.join(SCRIPT_DIR, "..", "mlruns"),
    ]:
        if os.path.isdir(base):
            exp_id = str(run_row.get("experiment_id", ""))
            exp_name = str(run_row.get("experiment_name", ""))
            candidates = [
                os.path.join(base, exp_name, run_id, "artifacts", "training_config.json"),
                os.path.join(base, exp_id, run_id, "artifacts", "training_config.json"),
                os.path.join(base, run_id, "artifacts", "training_config.json"),
            ]
            for c in candidates:
                if os.path.isfile(c):
                    try:
                        with open(c, "r", encoding="utf-8") as f:
                            return json.load(f)
                    except Exception:
                        pass
            # Also check all experiment subdirectories in base
            try:
                for sub in os.listdir(base):
                    sub_cfg = os.path.join(base, sub, run_id, "artifacts", "training_config.json")
                    if os.path.isfile(sub_cfg):
                        with open(sub_cfg, "r", encoding="utf-8") as f:
                            return json.load(f)
            except Exception:
                pass

    # 4. Fallback via MlflowClient download
    if client is not None:
        try:
            local_cfg = client.download_artifacts(run_id, "training_config.json")
            if os.path.isfile(local_cfg):
                with open(local_cfg, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass

    return None


def enrich_runs_with_seeds(runs_df: pd.DataFrame, tracking_uri: str) -> pd.DataFrame:
    """Ensures seeds (seed, data.seed, data.mask.seed, data.mask.model_index) are extracted into runs_df."""
    need_enrich = False
    for col in ["params.seed", "params.data.seed"]:
        if col not in runs_df.columns or runs_df[col].isna().any():
            need_enrich = True
            break

    if not need_enrich:
        return runs_df

    client = None
    try:
        client = mlflow.tracking.MlflowClient()
    except Exception:
        pass

    seed_map = {}
    data_seed_map = {}
    mask_seed_map = {}
    model_idx_map = {}

    def process_row(idx_row):
        idx, row = idx_row
        cfg = fetch_run_config(row, tracking_uri, client)
        if not cfg:
            return idx, None, None, None, None
        s = cfg.get("seed")
        d_cfg = cfg.get("data") if isinstance(cfg.get("data"), dict) else {}
        d_s = d_cfg.get("seed")
        m_cfg = d_cfg.get("mask") if isinstance(d_cfg.get("mask"), dict) else {}
        m_s = m_cfg.get("seed")
        m_idx = m_cfg.get("model_index")
        return idx, s, d_s, m_s, m_idx

    rows = list(runs_df.iterrows())
    with ThreadPoolExecutor(max_workers=min(32, max(1, len(rows)))) as executor:
        for idx, s, d_s, m_s, m_idx in executor.map(process_row, rows):
            if s is not None:
                seed_map[idx] = s
            if d_s is not None:
                data_seed_map[idx] = d_s
            if m_s is not None:
                mask_seed_map[idx] = m_s
            if m_idx is not None:
                model_idx_map[idx] = m_idx

    if "params.seed" not in runs_df.columns:
        runs_df["params.seed"] = pd.Series(seed_map, dtype="object")
    else:
        runs_df["params.seed"] = runs_df["params.seed"].fillna(pd.Series(seed_map))

    if "params.data.seed" not in runs_df.columns:
        runs_df["params.data.seed"] = pd.Series(data_seed_map, dtype="object")
    else:
        runs_df["params.data.seed"] = runs_df["params.data.seed"].fillna(pd.Series(data_seed_map))

    if "params.data.mask.seed" not in runs_df.columns:
        runs_df["params.data.mask.seed"] = pd.Series(mask_seed_map, dtype="object")
    else:
        runs_df["params.data.mask.seed"] = runs_df["params.data.mask.seed"].fillna(pd.Series(mask_seed_map))

    if "params.data.mask.model_index" not in runs_df.columns:
        runs_df["params.data.mask.model_index"] = pd.Series(model_idx_map, dtype="object")
    else:
        runs_df["params.data.mask.model_index"] = runs_df["params.data.mask.model_index"].fillna(pd.Series(model_idx_map))

    # Also map direct names for convenience
    for param_key in ["seed", "data.seed", "data.mask.seed", "data.mask.model_index"]:
        col = f"params.{param_key}"
        if col in runs_df.columns and param_key not in runs_df.columns:
            runs_df[param_key] = runs_df[col]

    return runs_df



def get_default_tracking_uri() -> str:
    """Returns the default tracking URI based on environment or local storage."""
    env_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if env_uri:
        return env_uri
    workspace_mlruns = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "mlruns"))
    if os.path.isdir(workspace_mlruns):
        return f"file:///{workspace_mlruns.replace(os.sep, '/')}"
    return "http://localhost:5051"


def format_duration(seconds: float | None) -> str:
    """Formats a duration in seconds into a human-readable string (e.g. 1h 23m 45s or 12.3s)."""
    if seconds is None or pd.isna(seconds) or seconds < 0:
        return "-"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    rem_seconds = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m {rem_seconds:02d}s"
    hours = int(minutes // 60)
    rem_minutes = int(minutes % 60)
    return f"{hours}h {rem_minutes:02d}m {rem_seconds:02d}s"


def format_time(ts) -> str:
    """Formats timestamp into YYYY-MM-DD HH:MM:SS."""
    if ts is None or pd.isna(ts):
        return "-"
    if isinstance(ts, pd.Timestamp):
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(ts, (int, float)):
        try:
            dt = datetime.datetime.fromtimestamp(ts / 1000.0)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return str(ts)
    return str(ts)


def list_runs(
    experiment_name: str | None,
    tracking_uri: str,
    status_filter: str | None = None,
    name_filter: str | None = None,
    param_filters: list[str] | None = None,
    output_file: str | None = None,
    view_type: str = "ACTIVE_ONLY",
    sort_by: str = "start_time",
    sort_descending: bool = True,
    limit: int | None = None,
    show_all_params: bool = False,
    selected_params: list[str] | None = None,
    show_metrics: bool = False,
):
    print(f"Connecting to MLflow tracking server at: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)

    # Resolve view type
    vtype = ViewType.ACTIVE_ONLY
    if view_type.upper() == "DELETED_ONLY":
        vtype = ViewType.DELETED_ONLY
    elif view_type.upper() == "ALL":
        vtype = ViewType.ALL

    # Search experiments
    all_experiments = mlflow.search_experiments(view_type=ViewType.ALL)
    exp_map = {exp.experiment_id: exp.name for exp in all_experiments}

    if not all_experiments:
        print("No experiments found in the MLflow tracking store.")
        return

    experiment_ids = []
    if experiment_name:
        matched_exps = [
            exp
            for exp in all_experiments
            if exp.name == experiment_name or exp.experiment_id == experiment_name
        ]
        if not matched_exps:
            matched_exps = [
                exp for exp in all_experiments if experiment_name.lower() in exp.name.lower()
            ]
        if not matched_exps:
            print(f"Error: Experiment '{experiment_name}' not found.")
            print("Available experiments:")
            for exp in all_experiments:
                print(f"  - {exp.name} (ID: {exp.experiment_id})")
            return
        experiment_ids = [exp.experiment_id for exp in matched_exps]
        exp_labels = [f"{e.name} (ID: {e.experiment_id})" for e in matched_exps]
        print(f"Targeting Experiment(s): {', '.join(exp_labels)}")
    else:
        experiment_ids = [exp.experiment_id for exp in all_experiments]
        print(f"Searching across all {len(experiment_ids)} experiments...")

    # Fetch runs
    try:
        runs_df = mlflow.search_runs(
            experiment_ids=experiment_ids,
            run_view_type=vtype,
            output_format="pandas",
        )
    except Exception as e:
        print(f"Error while querying runs from MLflow: {e}")
        return

    if runs_df.empty:
        print("No runs found matching the query criteria.")
        return

    # Add experiment_name column
    runs_df["experiment_name"] = (
        runs_df["experiment_id"].map(exp_map).fillna(runs_df["experiment_id"])
    )

    # Ensure run_name column exists
    if "tags.mlflow.runName" in runs_df.columns:
        runs_df["run_name"] = runs_df["tags.mlflow.runName"].fillna("-")
    else:
        runs_df["run_name"] = "-"

    # Enrich runs with seeds (seed, data.seed, data.mask.seed, data.mask.model_index)
    runs_df = enrich_runs_with_seeds(runs_df, tracking_uri)

    # Calculate duration
    tz_info = getattr(getattr(runs_df["start_time"], "dt", None), "tz", None)
    now = pd.Timestamp.now(tz=tz_info)

    def compute_duration(row):
        start = row.get("start_time")
        end = row.get("end_time")
        status = str(row.get("status", "")).upper()
        if pd.isna(start):
            return None
        if pd.notna(end):
            diff = (end - start).total_seconds()
            return diff if diff >= 0 else 0.0
        elif status == "RUNNING":
            diff = (now - start).total_seconds()
            return diff if diff >= 0 else 0.0
        return None

    runs_df["duration_seconds"] = runs_df.apply(compute_duration, axis=1)
    runs_df["duration_formatted"] = runs_df["duration_seconds"].apply(format_duration)

    # Filter by Status
    if status_filter and status_filter.upper() != "ALL":
        target_statuses = [s.strip().upper() for s in status_filter.split(",")]
        runs_df = runs_df[runs_df["status"].str.upper().isin(target_statuses)]
        if runs_df.empty:
            print(f"No runs found with status in {target_statuses}.")
            return

    # Filter by Name
    if name_filter:
        pattern = re.compile(name_filter, re.IGNORECASE)
        runs_df = runs_df[runs_df["run_name"].apply(lambda n: bool(pattern.search(str(n))))]
        if runs_df.empty:
            print(f"No runs found matching run name pattern '{name_filter}'.")
            return

    # Filter by Parameters (e.g. key=value)
    if param_filters:
        for pf in param_filters:
            if "=" not in pf:
                print(f"Warning: Invalid param filter '{pf}'. Format should be 'key=val'. Skip.")
                continue
            k, v = pf.split("=", 1)
            col_name = f"params.{k.strip()}"
            if col_name in runs_df.columns:
                runs_df = runs_df[runs_df[col_name].astype(str) == str(v.strip())]
            elif k.strip() in runs_df.columns:
                runs_df = runs_df[runs_df[k.strip()].astype(str) == str(v.strip())]
            else:
                print(f"Warning: Parameter column '{k}' not found in runs data.")
        if runs_df.empty:
            print(f"No runs found matching parameter filters {param_filters}.")
            return

    # Sorting
    sort_col = "start_time"
    if sort_by:
        if sort_by in runs_df.columns:
            sort_col = sort_by
        elif f"params.{sort_by}" in runs_df.columns:
            sort_col = f"params.{sort_by}"
        elif f"metrics.{sort_by}" in runs_df.columns:
            sort_col = f"metrics.{sort_by}"
        elif sort_by == "duration":
            sort_col = "duration_seconds"
        elif sort_by == "name":
            sort_col = "run_name"

    runs_df = runs_df.sort_values(by=sort_col, ascending=not sort_descending)

    if limit and limit > 0:
        runs_df = runs_df.head(limit)

    # Parameter columns
    all_param_cols = sorted([col for col in runs_df.columns if col.startswith("params.")])

    if selected_params:
        display_param_cols = [
            f"params.{p}" for p in selected_params if f"params.{p}" in all_param_cols
        ]
    elif show_all_params:
        display_param_cols = all_param_cols
    else:
        display_param_cols = []
        for col in all_param_cols:
            non_null = runs_df[col].dropna()
            if not non_null.empty:
                display_param_cols.append(col)

    # Print Summary & Table
    total_runs = len(runs_df)
    finished_count = (runs_df["status"] == "FINISHED").sum()
    failed_count = (runs_df["status"] == "FAILED").sum()
    running_count = (runs_df["status"] == "RUNNING").sum()
    killed_count = (runs_df["status"] == "KILLED").sum()

    total_duration_sec = runs_df["duration_seconds"].sum(skipna=True)
    avg_duration_sec = runs_df["duration_seconds"].mean(skipna=True) if total_runs > 0 else 0

    print("\n" + "=" * 120)
    print(f" MLFLOW RUNS OVERVIEW (Total: {total_runs})")
    status_summary = (
        f" Status: FINISHED={finished_count} | FAILED={failed_count} | "
        f"RUNNING={running_count} | KILLED={killed_count}"
    )
    print(status_summary)
    time_summary = (
        f" Runtime: Total={format_duration(total_duration_sec)} | "
        f"Average={format_duration(avg_duration_sec)}"
    )
    print(time_summary)
    print("=" * 120)

    rows = []
    for idx, (_, row) in enumerate(runs_df.iterrows(), 1):
        run_id = str(row["run_id"])
        run_name = str(row["run_name"]) if pd.notna(row["run_name"]) else "-"
        status = str(row["status"]) if pd.notna(row["status"]) else "-"
        dur = str(row["duration_formatted"])
        start = format_time(row.get("start_time"))
        exp_name = str(row.get("experiment_name", "-"))

        param_parts = []
        for col in display_param_cols:
            val = row.get(col)
            if pd.notna(val) and str(val) != "" and str(val) != "None":
                param_name = col.replace("params.", "")
                param_parts.append(f"{param_name}={val}")
        params_str = ", ".join(param_parts) if param_parts else "-"

        rows.append({
            "#": idx,
            "Run ID": run_id,
            "Run Name": run_name,
            "Experiment": exp_name,
            "Status": status,
            "Duration": dur,
            "Start Time": start,
            "Parameters": params_str,
        })

    header_line = (
        f"{'#':<4} | {'Run ID':<32} | {'Run Name':<25} | {'Status':<10} | "
        f"{'Duration':<12} | {'Start Time':<19} | Parameters"
    )
    print(header_line)
    print("-" * 120)

    for r in rows:
        r_name = (r["Run Name"][:22] + "...") if len(r["Run Name"]) > 25 else r["Run Name"]
        line = (
            f"{r['#']:<4} | {r['Run ID']:<32} | {r_name:<25} | {r['Status']:<10} | "
            f"{r['Duration']:<12} | {r['Start Time']:<19} | {r['Parameters']}"
        )
        print(line)

    print("=" * 120)

    # Optional export to file
    if output_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        ext = os.path.splitext(output_file)[1].lower()
        if ext == ".parquet":
            runs_df.to_parquet(output_file)
        elif ext == ".json":
            runs_df.to_json(output_file, orient="records", indent=2, date_format="iso")
        elif ext == ".md":
            table_df = pd.DataFrame(rows)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("# MLflow Runs Summary\n\n")
                f.write(f"- **Total Runs**: {total_runs}\n")
                f.write(f"- **Total Duration**: {format_duration(total_duration_sec)}\n")
                f.write(f"- **Average Duration**: {format_duration(avg_duration_sec)}\n\n")
                f.write(table_df.to_markdown(index=False))
        else:
            runs_df.to_csv(output_file, index=False)
        print(f"\nExported runs overview to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="List MLflow runs with Run ID, Run Name, Status, Duration, and Parameters.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python scripts/list_mlflow_runs.py find-grokking
  python scripts/list_mlflow_runs.py -e find-grokking --status FAILED
  python scripts/list_mlflow_runs.py -e canary-selection -p "seed=42" --sort-by duration
  python scripts/list_mlflow_runs.py --output runs_summary.csv
""",
    )
    parser.add_argument(
        "experiment",
        nargs="?",
        default=None,
        help="Name or ID of the MLflow experiment (searches all if omitted)",
    )
    parser.add_argument(
        "--experiment",
        "-e",
        dest="exp_opt",
        type=str,
        default=None,
        help="Explicit experiment name or ID flag",
    )
    parser.add_argument(
        "--status",
        "-s",
        type=str,
        default=None,
        help="Filter runs by status: FINISHED, FAILED, RUNNING, KILLED, ALL",
    )
    parser.add_argument(
        "--name-filter",
        "-n",
        type=str,
        default=None,
        help="Filter runs by run name regex/substring",
    )
    parser.add_argument(
        "--param-filter",
        "-p",
        action="append",
        default=[],
        help="Filter runs by parameter key=value (can be used multiple times, e.g. -p seed=42)",
    )
    parser.add_argument(
        "--params",
        type=str,
        default=None,
        help="Comma-separated list of specific parameters to display (e.g. 'seed,lr')",
    )
    parser.add_argument(
        "--all-params",
        action="store_true",
        help="Display all parameter columns instead of filtering empty ones",
    )
    parser.add_argument(
        "--show-metrics",
        action="store_true",
        help="Include final metrics in the output",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path to save runs (.csv, .json, .parquet, .md)",
    )
    parser.add_argument(
        "--view-type",
        type=str,
        default="ACTIVE_ONLY",
        choices=["ACTIVE_ONLY", "DELETED_ONLY", "ALL"],
        help="View type for runs (ACTIVE_ONLY, DELETED_ONLY, ALL)",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="start_time",
        help="Sort column: 'start_time', 'duration', 'name', 'status', or param name",
    )
    parser.add_argument(
        "--asc",
        action="store_true",
        help="Sort ascending instead of descending",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=None,
        help="Limit number of runs returned",
    )
    parser.add_argument(
        "--uri",
        "-u",
        type=str,
        default=get_default_tracking_uri(),
        help="MLflow tracking URI (default: MLFLOW_TRACKING_URI or http://localhost:5051)",
    )

    args = parser.parse_args()

    exp_name = args.exp_opt if args.exp_opt else args.experiment
    selected_params = [p.strip() for p in args.params.split(",")] if args.params else None

    list_runs(
        experiment_name=exp_name,
        tracking_uri=args.uri,
        status_filter=args.status,
        name_filter=args.name_filter,
        param_filters=args.param_filter,
        output_file=args.output,
        view_type=args.view_type,
        sort_by=args.sort_by,
        sort_descending=not args.asc,
        limit=args.limit,
        show_all_params=args.all_params,
        selected_params=selected_params,
        show_metrics=args.show_metrics,
    )


if __name__ == "__main__":
    main()
