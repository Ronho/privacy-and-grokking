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
def list_runs(exp_name: str, output: Path | None = None):
    import mlflow

    from privacy_and_grokking.utils import setup_mlflow

    setup_mlflow(exp_name)
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(exp_name)
    if experiment is None:
        raise typer.BadParameter(f"Experiment '{exp_name}' not found.")
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    lines = [
        f"{run.info.run_id} {run.data.tags.get('mlflow.runName', '<no name>')}"
        for run in runs
    ]
    for line in lines:
        typer.echo(line)
    if output is not None:
        output.write_text("\n".join(lines) + "\n")
        typer.echo(f"Written to {output}")


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
    from privacy_and_grokking.visualize import SINGLE_VIZ_NAMES, visualization_single_handler

    effective_include: list[str] | None = include
    if exclude:
        all_names = include if include is not None else list(SINGLE_VIZ_NAMES)
        effective_include = [n for n in all_names if n not in exclude]

    with Logger() as logger:
        logger.info("Starting single-run visualization handler.", extra={"run_id": run_id})
        visualization_single_handler(exp_name, run_id, include=effective_include)
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
    from privacy_and_grokking.visualize import MULTI_VIZ_NAMES, visualization_multi_handler

    effective_include: list[str] | None = include
    if exclude:
        all_names = include if include is not None else list(MULTI_VIZ_NAMES)
        effective_include = [n for n in all_names if n not in exclude]

    with Logger() as logger:
        logger.info("Starting multi-run visualization handler.", extra={"run_ids": run_ids})
        visualization_multi_handler(exp_name, run_ids, include=effective_include)
        logger.info("Multi-run visualization handler completed.")


@app.command()
def pipeline(
    exp_name: str,
    model: str,
    total_steps: int,
    mask_index: int,
    seed: int | None = None,
    run_name: str | None = None,
    run_id: str | None = None,
    checkpoint: int | None = None,
):
    if checkpoint is not None and run_id is None:
        raise typer.BadParameter("--run-id is required when --checkpoint is provided.")

    with Logger() as logger:
        logger.info("Starting pipeline.")

        # Train
        if checkpoint is not None:
            assert run_id is not None  # guaranteed by the check above
            cfg: TrainConfig | RestartConfig = RestartConfig(run_id=run_id, checkpoint=checkpoint)
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

        # Extract
        from privacy_and_grokking.extraction import extraction_handler

        logger.info("Running extract step.", extra={"run_id": current_run_id})
        extraction_handler(exp_name, current_run_id)
        logger.info("Extract step complete.")

        # Visualize
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
