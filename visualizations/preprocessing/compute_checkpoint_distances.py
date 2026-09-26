#!/usr/bin/env python3
"""compute_checkpoint_distances.py.

Computes pairwise Euclidean distance matrices between checkpoint parameter vectors
of training runs. Supports single pairs or full matrices across a list of runs.

Required Inputs:
  - Checkpoint model weights (`checkpoints/{step}/model.pth`) from the local cache
    (`cache/runs/{run_id}/...`) or fetched on demand from the MLflow tracking server.
  - Training configuration (`training_config.json`) for run metadata.

Generated Outputs:
  - cache/runs/{run_id_1}/distances/{run_id_2}_distance.json: Serialized JSON artifact containing:
      - `run_id_1`, `run_id_2`: Identifiers of both runs.
      - `run_name_1`, `run_name_2`: Human-readable run names.
      - `steps_1`, `steps_2`: Sorted list of checkpoint step numbers for row and column axes.
      - `matrix`: 2D nested list of shape (len(steps_1), len(steps_2)) with Euclidean distances.
      - `metric`: Distance metric used ("euclidean").
"""

from __future__ import annotations

from typing import Annotated

import typer

from privacy_and_grokking.artifacts.cache import DEFAULT_TRACKING_URI
from privacy_and_grokking.metrics.checkpoint_distance import (
    compute_multiple_runs_distances,
    compute_run_pair_distance,
)

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def main(
    run_ids: Annotated[
        list[str] | None,
        typer.Option(
            "--run-ids",
            "-r",
            help="List of MLflow run IDs to compute all pairwise distance matrices for.",
        ),
    ] = None,
    run_id_1: Annotated[
        str | None,
        typer.Option(
            "--run-id-1",
            "-r1",
            help="First MLflow run ID for single pair computation.",
        ),
    ] = None,
    run_id_2: Annotated[
        str | None,
        typer.Option(
            "--run-id-2",
            "-r2",
            help="Second MLflow run ID for single pair computation. Defaults to run-id-1.",
        ),
    ] = None,
    tracking_uri: Annotated[
        str,
        typer.Option("--tracking-uri", "-u", help="MLflow tracking URI."),
    ] = DEFAULT_TRACKING_URI,
    max_workers: Annotated[
        int,
        typer.Option("--max-workers", "-w", help="Max parallel worker threads for fetching."),
    ] = 8,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Force recomputation even if cached."),
    ] = False,
) -> None:
    """CLI entrypoint for computing checkpoint distance matrices."""
    # Parse run IDs if provided as a list or comma-separated string
    target_run_ids: list[str] = []
    if run_ids:
        for item in run_ids:
            target_run_ids.extend([x.strip() for x in item.split(",") if x.strip()])

    if target_run_ids:
        print(f"Processing {len(target_run_ids)} runs in batch mode...")
        compute_multiple_runs_distances(
            run_ids=target_run_ids,
            tracking_uri=tracking_uri,
            max_workers=max_workers,
            force_recompute=force,
        )
    elif run_id_1:
        target_run_2 = run_id_2 or run_id_1
        compute_run_pair_distance(
            run_id_1=run_id_1,
            run_id_2=target_run_2,
            tracking_uri=tracking_uri,
            max_workers=max_workers,
            force_recompute=force,
        )
    else:
        raise typer.BadParameter("Please provide either --run-ids / -r or --run-id-1 / -r1.")


if __name__ == "__main__":
    app()
