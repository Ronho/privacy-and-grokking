import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import typer
from typer import Typer

from privacy_and_grokking.config import TrainConfig
from privacy_and_grokking.training import RestartConfig
from privacy_and_grokking.training import train as training
from privacy_and_grokking.training.train import LOG_FREQUENCY
from privacy_and_grokking.utils import Logger
from privacy_and_grokking.visualize import visualization_multi_handler, visualization_single_handler

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
        f"{run.info.run_id} {run.data.tags.get('mlflow.runName', '<no name>')}" for run in runs
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
    checkpoint_frequency: int = LOG_FREQUENCY,
):
    cfg = TrainConfig.model_validate_json((CONFIG_DIR / model).read_bytes())
    if seed is not None:
        cfg.seed = seed
    cfg.data.mask.model_index = mask_index
    training(
        exp_name=exp_name,
        total_steps=total_steps,
        cfg=cfg,
        run_name=run_name,
        checkpoint_frequency=checkpoint_frequency,
    )


@app.command()
def restart(
    exp_name: str,
    run_id: str,
    checkpoint: int,
    total_steps: int,
    checkpoint_frequency: int = LOG_FREQUENCY,
):
    cfg = RestartConfig(run_id=run_id, checkpoint=checkpoint)
    training(
        exp_name=exp_name,
        total_steps=total_steps,
        cfg=cfg,
        run_name="",
        checkpoint_frequency=checkpoint_frequency,
    )


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
    include: list[str] | None = None,
    exclude: list[str] | None = None,
):
    with Logger() as logger:
        logger.info("Starting single-run visualization handler.", extra={"run_id": run_id})
        visualization_single_handler(exp_name, run_id, include=include, exclude=exclude)
        logger.info("Single-run visualization handler completed.")


@app.command()
def visualize_multi(
    exp_name: str,
    tag: str | None = None,
    run_ids: list[str] | None = None,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    group: bool = False,
    aggregate: bool = False,
    postfix: str | None = None,
):
    with Logger() as logger:
        logger.info("Starting multi-run visualization handler.", extra={"run_ids": run_ids})
        visualization_multi_handler(
            exp_name,
            run_ids,
            tag,
            include=include,
            exclude=exclude,
            postfix=postfix,
            group=group,
            aggregate=aggregate,
        )
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
    checkpoint_frequency: int = LOG_FREQUENCY,
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
            cfg.data.mask.model_index = mask_index
            logger.info("Running train step.")
        current_run_id = training(
            exp_name=exp_name,
            total_steps=total_steps,
            cfg=cfg,
            run_name=run_name,
            checkpoint_frequency=checkpoint_frequency,
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
    with open(path) as f, ThreadPoolExecutor(max_workers=num_workers) as executor:
        executor.map(_handle, f)


@app.command()
def search():
    from privacy_and_grokking.search import generate_search_configs

    with Logger() as logger:
        configs = generate_search_configs()
        logger.info(f"Generated {len(configs)} search configs.")


@app.command()
def command():
    from pathlib import Path

    configs = Path("./configs")
    commands = Path("./commands")
    command = commands / "train.txt"
    num_samples = 5
    lines = []
    for config in configs.iterdir():
        for i in range(num_samples):
            lines.append(
                f"CUDA_VISIBLE_DEVICES=1 uv run pag train v5.0.0 {config.name} 10000000 {i} --seed {i} --run-name {config.stem} --checkpoint-frequency 25000"
            )
    command.parent.mkdir(parents=True, exist_ok=True)
    command.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    app()
