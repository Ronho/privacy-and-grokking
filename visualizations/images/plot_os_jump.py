#!/usr/bin/env python3
"""plot_os_jump.py.

Generates jump plane plots for the exact same canonical default runs
as the optimizer dynamics (PCA) plots.

Outputs:
  - cache/plots/jump_planes/jump_planes_2d_{short_descriptor}_{run_id}.pdf
  - cache/plots/jump_planes/jump_planes_2d_{short_descriptor}_{run_id}.png
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Annotated

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import typer
from tqdm import tqdm

from privacy_and_grokking.artifacts.cache import DEFAULT_TRACKING_URI
from privacy_and_grokking.utils import Logger
from visualizations.images.plot_jump_planes import plot_jump_planes
from visualizations.preprocessing.extract_1d_interpolation import (
    get_available_checkpoints,
    get_checkpoint_weight,
)

app = typer.Typer(pretty_exceptions_enable=False)

# Mapping of model tag to canonical default PCA run IDs
DEFAULT_RUN_IDS = {
    "GROK_MNIST_CE_MLP": "23d2c3bbfd5d4c3eb373ec98564ba77f",
    "GROK_MNIST_MSE_MLP": "580fc01a67d04a6697626730a88e7bb7",
    "NO_GROK_MNIST_CE_MLP": "51c177903c1d4f29a8f4dae05c35587f",
    "GROK_MNIST_CE_VIT": "a46642282384408481241143a6b4bdc0",
    "NO_GROK_MNIST_CE_VIT": "0ad0d9f8dbc64e6fa407479a613726d9",
}


def download_checkpoints_for_run(run_id: str, tracking_uri: str, num_workers: int = 4) -> None:
    """Downloads all checkpoints for a run concurrently."""
    try:
        steps = get_available_checkpoints(run_id, tracking_uri)
    except Exception as e:
        print(f"Failed to fetch steps for run {run_id}: {e}")
        return

    print(f"Found {len(steps)} checkpoints for run '{run_id}'. Downloading concurrently...")
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(get_checkpoint_weight, run_id, step, tracking_uri): step 
            for step in steps
        }
        
        with tqdm(total=len(futures), desc=f"Downloading {run_id[:8]}") as pbar:
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    step = futures[future]
                    print(f"Error downloading step {step}: {e}")
                pbar.update(1)


@app.command()
def main(
    tracking_uri: Annotated[
        str,
        typer.Option("--tracking-uri", "-u", help="MLflow server URI."),
    ] = DEFAULT_TRACKING_URI,
    force_extract: Annotated[
        bool,
        typer.Option("--force-extract", help="Force re-extraction of jump planes data."),
    ] = False,
    download_only: Annotated[
        bool,
        typer.Option("--download-only", help="Only download the model checkpoints, don't compute planes."),
    ] = False,
    num_workers: Annotated[
        int,
        typer.Option("--num-workers", help="Number of concurrent download threads."),
    ] = 4,
) -> None:
    """Entrypoint: Generates jump plane plots for all canonical PCA runs from the optimizer sweep."""
    Logger().setup()

    if download_only:
        print(f"Executing DOWNLOAD-ONLY mode for {len(DEFAULT_RUN_IDS)} canonical runs...")
        for tag, run_id in DEFAULT_RUN_IDS.items():
            print(f"\n{'=' * 70}\n[Downloading Checkpoints] {tag} (Run ID: {run_id})\n{'=' * 70}")
            download_checkpoints_for_run(run_id, tracking_uri, num_workers)
        print("\n[Done] Completed checkpoint downloads!")
        return

    print(f"Generating jump plane plots for {len(DEFAULT_RUN_IDS)} canonical runs...")
    for tag, run_id in DEFAULT_RUN_IDS.items():
        print(f"\n{'=' * 70}\n[Plotting Jump Planes] {tag} (Run ID: {run_id})\n{'=' * 70}")
        try:
            plot_jump_planes(
                run_id=run_id,
                tracking_uri=tracking_uri,
                force_extract=force_extract,
                short_descriptor=tag,
            )
        except Exception as e:
            print(f"Error generating jump plane plot for {tag} ({run_id}): {e}")

    print("\n[Done] Successfully generated all jump plane plots for optimizer sweep runs!")


if __name__ == "__main__":
    app()
