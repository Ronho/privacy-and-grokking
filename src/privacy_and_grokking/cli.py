import os
from multiprocessing import Pool
from pathlib import Path
from typing import Annotated

import typer
from typer import Typer

from privacy_and_grokking.config import TrainConfig
from privacy_and_grokking.training import RestartConfig
from privacy_and_grokking.training import train as training
from privacy_and_grokking.utils import Logger

app = Typer(name="Privacy and Grokking CLI", pretty_exceptions_enable=False)

CONFIG_DIR = Path(__file__).parent.parent.parent / "configs"


@app.command()
def list_models(verbose: bool = False):
    with Logger() as logger:
        for model in CONFIG_DIR.iterdir():
            cfg = TrainConfig.model_validate_json((CONFIG_DIR / model).read_bytes())
            extra = {"config": cfg.model_dump()} if verbose else {}
            logger.info(model.name, **extra)


@app.command()
def train(
    exp_name: str,
    model: str,
    total_steps: int,
    mask_index: int,
    seed: int | None = None,
    run_name: str | None = None,
):
    cfg = TrainConfig.model_validate_json((CONFIG_DIR / model).read_bytes())
    if seed is not None:
        cfg.seed = seed
    cfg.dataset_mask_idx = mask_index
    training(exp_name=exp_name, total_steps=total_steps, cfg=cfg, run_name=run_name)


@app.command()
def restart(exp_name: str, run_id: str, checkpoint: int, total_steps: int):
    cfg = RestartConfig(run_id=run_id, checkpoint=checkpoint)
    training(exp_name=exp_name, total_steps=total_steps, cfg=cfg, run_name="")


@app.command()
def extract(
    exp_name: str,
    run_id: str,
):
    from privacy_and_grokking.extraction import extraction_handler

    with Logger() as logger:
        logger.info("Starting extraction handler.", extra={"run_id": run_id})
        extraction_handler(exp_name, run_id)
        logger.info("Extraction handler completed.")


@app.command()
def visualize_single(
    exp_name: str,
    run_id: str,
    include: Annotated[
        list[str] | None,
        typer.Option("--include", help="Visualization names to include (default: all)."),
    ] = None,
    exclude: Annotated[
        list[str] | None,
        typer.Option("--exclude", help="Visualization names to exclude."),
    ] = None,
):
    from privacy_and_grokking.visualize import visualization_single_handler

    with Logger() as logger:
        logger.info("Starting single-run visualization handler.", extra={"run_id": run_id})
        visualization_single_handler(exp_name, run_id, include=include, exclude=exclude)
        logger.info("Single-run visualization handler completed.")


@app.command()
def visualize_multi(
    exp_name: str,
    run_ids: list[str],
    include: Annotated[
        list[str] | None,
        typer.Option("--include", help="Visualization names to include (default: all)."),
    ] = None,
    exclude: Annotated[
        list[str] | None,
        typer.Option("--exclude", help="Visualization names to exclude."),
    ] = None,
):
    from privacy_and_grokking.visualize import visualization_multi_handler

    with Logger() as logger:
        logger.info("Starting multi-run visualization handler.", extra={"run_ids": run_ids})
        visualization_multi_handler(exp_name, run_ids, include=include, exclude=exclude)
        logger.info("Multi-run visualization handler completed.")


PipelineSteps = Literal["train", "extract", "visualize"]


@app.command()
def pipeline(
    exp_name: str,
    model: str,
    total_steps: int,
    mask_index: int,
    seed: int | None = None,
    run_name: str | None = None,
    all_activations: bool = False,
    run_id: str | None = None,
    checkpoint: int | None = None,
    include: list[PipelineSteps] | None = None,
    exclude: list[PipelineSteps] | None = None,
):
    active_steps: set[str] = set(include) if include else set(VALID_PIPELINE_STEPS)
    if exclude:
        active_steps -= set(exclude)
    invalid = active_steps - VALID_PIPELINE_STEPS
    if invalid:
        raise typer.BadParameter(
            f"Unknown steps: {sorted(invalid)!r}. Valid: {sorted(VALID_PIPELINE_STEPS)!r}"
        )
    if checkpoint is not None and run_id is None:
        raise typer.BadParameter("--run-id is required when --checkpoint is provided.")

    with Logger() as logger:
        logger.info("Starting pipeline.", extra={"active_steps": sorted(active_steps)})

        current_run_id = run_id

        if "train" in active_steps:
            if checkpoint is not None:
                cfg: TrainConfig | RestartConfig = RestartConfig(
                    run_id=run_id, checkpoint=checkpoint
                )
                logger.info(
                    "Running train step (restart).",
                    extra={"run_id": run_id, "checkpoint": checkpoint},
                )
            else:
                cfg = TrainConfig.model_validate_json((CONFIG_DIR / model).read_bytes())
                if seed is not None:
                    cfg.seed = seed
                cfg.dataset_mask_idx = mask_index
                logger.info("Running train step.")
            current_run_id = training(
                exp_name=exp_name, total_steps=total_steps, cfg=cfg, run_name=run_name
            )
            logger.info("Train step complete.", extra={"run_id": current_run_id})
        elif current_run_id is None:
            raise typer.BadParameter(
                "--run-id is required when the train step is excluded from the pipeline."
            )

        if "extract" in active_steps:
            from privacy_and_grokking.extraction import extraction_handler

            logger.info("Running extract step.", extra={"run_id": current_run_id})
            extraction_handler(exp_name, current_run_id, save_all_activations=all_activations)
            logger.info("Extract step complete.")

        if "visualize" in active_steps:
            from privacy_and_grokking.visualize import visualization_single_handler

            logger.info("Running visualize step.", extra={"run_id": current_run_id})
            visualization_single_handler(exp_name, current_run_id)
            logger.info("Visualize step complete.")

        logger.info("Pipeline complete.", extra={"run_id": current_run_id})


def _handle(line):
    line = line.strip()
    if line:
        with Logger() as logger:
            logger.info("Processing command.", extra={"command": line})
            os.system(line)


@app.command()
def process(path: str, num_workers: int):
    with open(path) as f, Pool(num_workers) as pool:
        pool.map(_handle, f)


if __name__ == "__main__":
    app()
