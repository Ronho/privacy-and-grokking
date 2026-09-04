import argparse
import datetime
import os
import re
import sys

import mlflow
import pandas as pd
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_default_tracking_uri() -> str:
    """Returns default tracking URI (defaults to local MLflow server at http://localhost:5051)."""
    env_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if env_uri:
        return env_uri
    return "http://localhost:5051"


def format_duration(seconds: float | None) -> str:
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


def parse_time_delta(time_str: str) -> datetime.timedelta:
    """Parses a time string like '24h', '7d', '30m' into a timedelta."""
    match = re.match(r"^(\d+)([mhd])$", time_str.strip().lower())
    if not match:
        raise ValueError(
            f"Invalid time format '{time_str}'. Use format like '30m', '12h', or '7d'."
        )
    value, unit = int(match.group(1)), match.group(2)
    if unit == "m":
        return datetime.timedelta(minutes=value)
    elif unit == "h":
        return datetime.timedelta(hours=value)
    elif unit == "d":
        return datetime.timedelta(days=value)
    raise ValueError(f"Unknown time unit '{unit}'.")


def delete_runs(
    run_ids: list[str] | None,
    run_ids_file: str | None,
    experiment_name: str | None,
    status_filter: str | None,
    name_filter: str | None,
    param_filters: list[str] | None,
    filter_string: str | None,
    older_than: str | None,
    tracking_uri: str,
    dry_run: bool,
    force: bool,
    restore: bool,
):
    action_name = "Restore" if restore else "Delete"
    print(f"Connecting to MLflow server at: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri)

    # Collect explicitly specified run IDs
    target_explicit_ids = []
    if run_ids:
        target_explicit_ids.extend(run_ids)
    if run_ids_file and os.path.exists(run_ids_file):
        with open(run_ids_file, encoding="utf-8") as f:
            for line in f:
                cleaned = line.strip().split("#")[0].strip()
                if cleaned:
                    target_explicit_ids.append(cleaned)

    # Resolve Experiments
    all_experiments = mlflow.search_experiments(view_type=ViewType.ALL)
    exp_map = {exp.experiment_id: exp.name for exp in all_experiments}

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
    else:
        experiment_ids = [exp.experiment_id for exp in all_experiments]

    # Decide search view type: if restoring, look in DELETED_ONLY, otherwise ACTIVE_ONLY
    vtype = ViewType.DELETED_ONLY if restore else ViewType.ACTIVE_ONLY

    # If explicit run IDs were provided without other filters, look them up directly
    has_other_filters = bool(
        experiment_name or status_filter or name_filter or param_filters or filter_string
    )
    if target_explicit_ids and not has_other_filters:
        runs_list = []
        from tqdm import tqdm

        print(f"Fetching details for {len(target_explicit_ids)} runs...")
        for rid in tqdm(target_explicit_ids, desc="Fetching runs", unit="run"):
            try:
                r = client.get_run(rid)
                runs_list.append(r)
            except Exception as e:
                tqdm.write(f"Warning: Could not fetch run '{rid}': {e}")
        if not runs_list:
            print("No valid runs found for the provided Run IDs.")
            return
        runs_data = []
        for r in runs_list:
            run_dict = {
                "run_id": r.info.run_id,
                "experiment_id": r.info.experiment_id,
                "status": r.info.status,
                "start_time": (
                    pd.to_datetime(r.info.start_time, unit="ms") if r.info.start_time else None
                ),
                "end_time": (
                    pd.to_datetime(r.info.end_time, unit="ms") if r.info.end_time else None
                ),
                "tags.mlflow.runName": r.data.tags.get("mlflow.runName", "-"),
            }
            for k, v in r.data.params.items():
                run_dict[f"params.{k}"] = v
            runs_data.append(run_dict)
        runs_df = pd.DataFrame(runs_data)
    else:
        # Search runs via MLflow API
        try:
            runs_df = mlflow.search_runs(
                experiment_ids=experiment_ids,
                filter_string=filter_string or "",
                run_view_type=vtype,
                output_format="pandas",
            )
        except Exception as e:
            print(f"Error querying runs: {e}")
            return

        if runs_df.empty:
            print(f"No {'deleted' if restore else 'active'} runs found matching the query.")
            return

        if target_explicit_ids:
            runs_df = runs_df[runs_df["run_id"].isin(target_explicit_ids)]
            if runs_df.empty:
                print("None of the specified Run IDs matched the search criteria.")
                return

    # Add experiment name
    runs_df["experiment_name"] = (
        runs_df["experiment_id"].map(exp_map).fillna(runs_df["experiment_id"])
    )

    # Ensure run_name
    if "tags.mlflow.runName" in runs_df.columns:
        runs_df["run_name"] = runs_df["tags.mlflow.runName"].fillna("-")
    else:
        runs_df["run_name"] = "-"

    # Filter by Status
    if status_filter:
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

    # Filter by Parameters
    if param_filters:
        for pf in param_filters:
            if "=" not in pf:
                print(f"Warning: Invalid param filter '{pf}'. Format should be 'key=val'. Skip.")
                continue
            k, v = pf.split("=", 1)
            col_name = f"params.{k.strip()}"
            if col_name in runs_df.columns:
                runs_df = runs_df[runs_df[col_name].astype(str) == str(v.strip())]
            else:
                print(f"Warning: Parameter '{k}' not present in fetched runs.")
        if runs_df.empty:
            print(f"No runs found matching parameter filters {param_filters}.")
            return

    # Filter by Age (older_than)
    if older_than:
        delta = parse_time_delta(older_than)
        tz_info = getattr(getattr(runs_df["start_time"], "dt", None), "tz", None)
        cutoff_time = pd.Timestamp.now(tz=tz_info) - delta
        runs_df = runs_df[runs_df["start_time"] < cutoff_time]
        if runs_df.empty:
            print(f"No runs found older than {older_than}.")
            return

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

    total_matched = len(runs_df)
    param_cols = sorted([col for col in runs_df.columns if col.startswith("params.")])

    # Display preview table
    print("\n" + "=" * 120)
    print(f" TARGETED RUNS FOR {action_name.upper()} (Total matched: {total_matched})")
    print("=" * 120)

    header_line = (
        f"{'#':<4} | {'Run ID':<32} | {'Run Name':<25} | {'Status':<10} | "
        f"{'Duration':<12} | {'Start Time':<19} | Parameters"
    )
    print(header_line)
    print("-" * 120)

    for idx, (_, row) in enumerate(runs_df.iterrows(), 1):
        run_id = str(row["run_id"])
        run_name = str(row["run_name"])
        r_name = (run_name[:22] + "...") if len(run_name) > 25 else run_name
        status = str(row.get("status", "-"))
        dur = str(row.get("duration_formatted", "-"))
        start = format_time(row.get("start_time"))

        param_parts = []
        for col in param_cols:
            val = row.get(col)
            if pd.notna(val) and str(val) != "" and str(val) != "None":
                param_parts.append(f"{col.replace('params.', '')}={val}")
        params_str = ", ".join(param_parts[:4]) + ("..." if len(param_parts) > 4 else "")

        line = (
            f"{idx:<4} | {run_id:<32} | {r_name:<25} | {status:<10} | "
            f"{dur:<12} | {start:<19} | {params_str}"
        )
        print(line)

    print("=" * 120)

    # Dry-run check
    if dry_run:
        msg = (
            f"\n[DRY RUN] No changes made. Found {total_matched} run(s) "
            f"that would be {action_name.lower()}d."
        )
        print(msg)
        print(f"To perform actual {action_name.lower()}, run without --dry-run (or with -y).")
        return

    # User confirmation prompt
    if not force:
        prompt_text = (
            f"\nAre you sure you want to {action_name.upper()} "
            f"these {total_matched} run(s)? [y/N]: "
        )
        try:
            choice = input(prompt_text).strip().lower()
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            return

        if choice not in ["y", "yes"]:
            print("Operation aborted. No runs were modified.")
            return

    # Perform deletion / restoration
    from tqdm import tqdm

    print(f"\nProceeding to {action_name.lower()} {total_matched} run(s)...")
    success_count = 0
    fail_count = 0

    with tqdm(total=total_matched, desc=f"{action_name}ing runs", unit="run") as pbar:
        for _, row in runs_df.iterrows():
            rid = row["run_id"]
            try:
                if restore:
                    client.restore_run(rid)
                else:
                    client.delete_run(rid)
                success_count += 1
            except Exception as e:
                pbar.write(f"Error {action_name.lower()}ing run {rid}: {e}")
                fail_count += 1
            pbar.update(1)

    print(f"\nSuccessfully {action_name.lower()}d {success_count} run(s).", end="")
    if fail_count > 0:
        print(f" Failed: {fail_count} run(s).")
    else:
        print("")

    if not restore:
        print("\nNote: MLflow performs a soft delete (marked as deleted in UI and tracking).")
        print("To permanently reclaim disk space on HPC, submit the garbage collector job:")
        print("  sbatch commands/mlflow_gc.sbatch")


def main():
    parser = argparse.ArgumentParser(
        description="Targeted deletion (or restoration) of MLflow runs via local or remote server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Dry run: see which failed runs would be deleted in experiment 'find-grokking'
  python scripts/delete_mlflow_runs.py -e find-grokking --status FAILED --dry-run

  # Delete specific runs by ID (with confirmation prompt)
  python scripts/delete_mlflow_runs.py --run-ids 0123456789abcdef0123456789abcdef

  # Delete all FAILED or KILLED runs in an experiment without interactive prompt
  python scripts/delete_mlflow_runs.py -e find-grokking --status FAILED,KILLED -y

  # Delete runs matching a name pattern and older than 3 days
  python scripts/delete_mlflow_runs.py -e canary-selection -n "test_.*" --older-than 3d

  # Restore previously soft-deleted runs
  python scripts/delete_mlflow_runs.py -e find-grokking --restore
""",
    )
    parser.add_argument(
        "--run-ids",
        "-r",
        nargs="+",
        default=None,
        help="One or more specific Run IDs to delete",
    )
    parser.add_argument(
        "--run-ids-file",
        "-f",
        type=str,
        default=None,
        help="Path to a text file containing Run IDs to delete (one ID per line)",
    )
    parser.add_argument(
        "--experiment",
        "-e",
        type=str,
        default=None,
        help="Target a specific experiment by name or ID",
    )
    parser.add_argument(
        "--status",
        "-s",
        type=str,
        default=None,
        help="Filter by run status (e.g. FAILED, KILLED, RUNNING, FINISHED)",
    )
    parser.add_argument(
        "--name-pattern",
        "-n",
        type=str,
        default=None,
        help="Filter by run name (regex or substring match)",
    )
    parser.add_argument(
        "--param-filter",
        "-p",
        action="append",
        default=[],
        help="Filter by parameter key=value (e.g. -p seed=42 -p lr=0.01)",
    )
    parser.add_argument(
        "--filter-string",
        type=str,
        default=None,
        help="Raw MLflow search filter string (e.g. 'params.batch_size = \\'64\\'')",
    )
    parser.add_argument(
        "--older-than",
        type=str,
        default=None,
        help="Filter runs started older than duration (e.g. '30m', '12h', '7d')",
    )
    parser.add_argument(
        "--uri",
        "-u",
        type=str,
        default=get_default_tracking_uri(),
        help="MLflow tracking URI (default: MLFLOW_TRACKING_URI or http://localhost:5051)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which runs would be deleted without actually deleting them",
    )
    parser.add_argument(
        "--yes",
        "-y",
        "--force",
        dest="force",
        action="store_true",
        help="Skip interactive confirmation prompt",
    )
    parser.add_argument(
        "--restore",
        action="store_true",
        help="Restore previously soft-deleted runs instead of deleting active runs",
    )

    args = parser.parse_args()

    # Safety check: Require at least one selection criterion
    if not (
        args.run_ids
        or args.run_ids_file
        or args.experiment
        or args.status
        or args.name_pattern
        or args.param_filter
        or args.filter_string
        or args.older_than
    ):
        print("Error: No selection criteria provided.")
        print("To prevent accidental deletion of everything, provide at least one filter:")
        print("  -e <exp>, -s <status>, -n <pattern>, -r <run_ids>, -p <param=val>, etc.")
        print("Use --help for usage details.")
        sys.exit(1)

    delete_runs(
        run_ids=args.run_ids,
        run_ids_file=args.run_ids_file,
        experiment_name=args.experiment,
        status_filter=args.status,
        name_filter=args.name_pattern,
        param_filters=args.param_filter,
        filter_string=args.filter_string,
        older_than=args.older_than,
        tracking_uri=args.uri,
        dry_run=args.dry_run,
        force=args.force,
        restore=args.restore,
    )


if __name__ == "__main__":
    main()
