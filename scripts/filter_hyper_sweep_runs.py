import argparse
import os
import re
import sys
from pathlib import Path
import pandas as pd


def parse_command(line: str) -> dict | None:
    """Parses relevant identifiers from a pag train command line."""
    line_s = line.strip()
    if not line_s or line_s.startswith("#"):
        return None

    rn_match = re.search(r"--run-name\s+(\S+)", line_s)
    s_match = re.search(r"-o\s+seed=(\d+)", line_s)
    ds_match = re.search(r"-o\s+data\.seed=(\d+)", line_s)
    dms_match = re.search(r"-o\s+data\.mask\.seed=(\d+)", line_s)
    dmi_match = re.search(r"-o\s+data\.mask\.model_index=(\d+)", line_s)

    if not (rn_match and s_match and dmi_match):
        return None

    return {
        "run_name": rn_match.group(1),
        "seed": int(s_match.group(1)),
        "data_seed": int(ds_match.group(1)) if ds_match else None,
        "data_mask_seed": int(dms_match.group(1)) if dms_match else None,
        "model_index": int(dmi_match.group(1)),
        "command": line_s,
    }


def find_parquet_file(candidate: str | None) -> Path:
    if candidate and Path(candidate).exists():
        return Path(candidate)

    # Search common locations
    priorities = [
        Path("hyper-sweeps_runs.parquet"),
        Path("cache/hyper-sweeps_runs.parquet"),
        Path("cache/hyper-sweep_runs.parquet"),
        Path("hyper-sweep_runs.parquet"),
    ]
    for p in priorities:
        if p.exists():
            return p

    raise FileNotFoundError("Could not locate hyper-sweep runs parquet file.")


def main():
    parser = argparse.ArgumentParser(
        description="Filter finished runs from hyper_sweep_<idx>.txt files and export running runs."
    )
    parser.add_argument(
        "--parquet",
        type=str,
        default=None,
        help="Path to the runs parquet file (defaults to auto-detect hyper-sweeps_runs.parquet or cache/hyper-sweep_runs.parquet).",
    )
    parser.add_argument(
        "--commands-dir",
        type=str,
        default="commands",
        help="Directory containing hyper_sweep_<idx>.txt files (default: commands).",
    )
    parser.add_argument(
        "--output-running",
        type=str,
        default="running_runs.txt",
        help="File to write currently running runs to (default: running_runs.txt).",
    )
    parser.add_argument(
        "--field",
        type=str,
        choices=["run_name", "run_id", "command"],
        default="run_name",
        help="Field to write for each running item (default: run_name).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate removal without modifying any hyper_sweep_<idx>.txt files.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create .bak backup copies of command files before modifying.",
    )
    args = parser.parse_args()

    parquet_path = find_parquet_file(args.parquet)
    print(f"Loading parquet: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    # Verification: check if seeds and model_index are present
    has_model_idx = "data.mask.model_index" in df.columns or "params.data.mask.model_index" in df.columns
    has_seed = "seed" in df.columns or "params.seed" in df.columns

    if not (has_model_idx and has_seed):
        print("\n" + "=" * 70)
        print("WARNING: CANNOT 100% IDENTIFY CORRECT RUNS WITH THIS PARQUET FILE!")
        print("=" * 70)
        print(f"The file '{parquet_path}' lacks 'data.mask.model_index' and 'seed' columns.")
        print("Because all 6 repetition files share the exact same --run-name,")
        print("it is impossible to know which of the 6 hyper_sweep_<idx>.txt files")
        print("each finished run belongs to.")
        print("\nNote: 'hyper-sweeps_runs.parquet' in the repository root has all")
        print("required seed and model_index columns to identify runs 100% accurately.")
        print("=" * 70 + "\n")
        sys.exit(1)

    model_idx_col = "data.mask.model_index" if "data.mask.model_index" in df.columns else "params.data.mask.model_index"
    seed_col = "seed" if "seed" in df.columns else "params.seed"
    data_seed_col = "data.seed" if "data.seed" in df.columns else "params.data.seed"
    data_mask_seed_col = "data.mask.seed" if "data.mask.seed" in df.columns else "params.data.mask.seed"

    # Separate running and finished runs
    running_df = df[df["status"] == "RUNNING"].copy()
    finished_df = df[df["status"] == "FINISHED"].copy()

    print(f"Total runs in parquet: {len(df)}")
    print(f"  FINISHED: {len(finished_df)}")
    print(f"  RUNNING:  {len(running_df)}")

    # 1. Produce running runs file
    commands_dir = Path(args.commands_dir)
    running_lines = []

    # If user selected command, we build a lookup map from the command files
    cmd_lookup = {}
    for f in sorted(commands_dir.glob("hyper_sweep_*.txt")):
        if f.name.endswith(".bak"):
            continue
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                parsed = parse_command(line)
                if parsed:
                    key = (parsed["run_name"], parsed["model_index"], parsed["seed"])
                    cmd_lookup[key] = parsed["command"]

    for _, row in running_df.iterrows():
        rn = str(row["run_name"])
        rid = str(row["run_id"])
        mid = int(row[model_idx_col])
        s = int(row[seed_col])

        if args.field == "run_id":
            running_lines.append(rid)
        elif args.field == "run_name":
            running_lines.append(rn)
        elif args.field == "command":
            cmd = cmd_lookup.get((rn, mid, s), f"# Command not found for {rn} (seed={s})")
            running_lines.append(cmd)

    output_running_path = Path(args.output_running)
    output_running_path.write_text("\n".join(running_lines) + "\n", encoding="utf-8")
    print(f"\nWrote {len(running_lines)} running runs ({args.field}) to: {output_running_path}")

    # Also create running_run_ids.txt and running_run_names.txt if different
    if args.field != "run_id":
        run_ids_path = output_running_path.parent / "running_run_ids.txt"
        run_ids_path.write_text("\n".join(running_df["run_id"].astype(str).tolist()) + "\n", encoding="utf-8")
        print(f"Also wrote running run IDs to: {run_ids_path}")

    # 2. Filter finished runs from hyper_sweep_<idx>.txt
    print("\nFiltering finished runs from hyper_sweep_<idx>.txt files...")

    # Build lookup set of finished run keys: (run_name, model_index, seed)
    finished_keys = set()
    for _, row in finished_df.iterrows():
        rn = str(row["run_name"])
        mid = int(row[model_idx_col])
        s = int(row[seed_col])
        finished_keys.add((rn, mid, s))

    print(f"Total unique finished run keys to remove: {len(finished_keys)}")

    total_removed = 0
    total_remaining = 0

    txt_files = sorted(commands_dir.glob("hyper_sweep_[0-9].txt"))
    if not txt_files:
        print(f"No hyper_sweep_[0-9].txt files found in {commands_dir}")
        return

    for txt_file in txt_files:
        m = re.search(r"hyper_sweep_(\d+)\.txt", txt_file.name)
        if not m:
            continue
        file_idx = int(m.group(1))

        with open(txt_file, "r", encoding="utf-8") as fh:
            lines = fh.readlines()

        kept_lines = []
        removed_count = 0

        for line in lines:
            parsed = parse_command(line)
            if not parsed:
                kept_lines.append(line)
                continue

            key = (parsed["run_name"], parsed["model_index"], parsed["seed"])
            if key in finished_keys:
                removed_count += 1
            else:
                kept_lines.append(line)

        print(
            f"  {txt_file.name}: {len(lines)} original -> removed {removed_count} finished -> {len(kept_lines)} remaining"
        )
        total_removed += removed_count
        total_remaining += len(kept_lines)

        if not args.dry_run:
            if not args.no_backup:
                backup_path = txt_file.with_suffix(".txt.bak")
                txt_file.replace(backup_path) if backup_path.exists() else None
                backup_path.write_text("".join(lines), encoding="utf-8")

            with open(txt_file, "w", encoding="utf-8") as fh:
                fh.writelines(kept_lines)

    print("\nSummary:")
    print(f"  Total finished runs removed: {total_removed} / {len(finished_keys)}")
    print(f"  Total remaining commands across all files: {total_remaining}")
    if args.dry_run:
        print("  (Dry run: no files were changed)")
    else:
        print("  Command files successfully updated.")


if __name__ == "__main__":
    main()
