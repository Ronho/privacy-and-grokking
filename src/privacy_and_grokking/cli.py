from datetime import datetime, timezone
from typer import Typer
from .logger import register_logger
from .path_keeper import get_path_keeper
from .config import TrainingRegistry
from .training import train as training, RestartConfig


app = Typer(name="Privacy and Grokking CLI", pretty_exceptions_enable=False)

def _init(id: str):
    paths = get_path_keeper()
    paths.set_params({"run_id": id, "log_id": datetime.now(timezone.utc).isoformat()})
    logger = register_logger(
        "default",
        log_file=paths.LOG,
        overwrite=True,
        log_level="DEBUG",
        channel="all",
        run_id=id
    )
    return logger


@app.command()
def train(id: str, models: list[str] | None = None):
    logger = _init(id)
    logger.info(f"Starting training for run {id}.")

    TrainingRegistry.load_defaults()
    model_list = TrainingRegistry.list()
    if models is None:
        models = model_list

    for model in models:
        if model in model_list:
            logger.info(f"Training model '{model}'.", extra={"model": model})

            training(TrainingRegistry.get(model))
        else:
            logger.error(f"Unknown model '{model}' specified for training. Ignoring.", extra={"model": model})

    logger.info(f"Training completed for run {id}.")

@app.command()
def restart(id: str, model: str, checkpoint: int):
    logger = _init(id)
    logger.info(f"Restarting training for run {id}, model '{model}' from checkpoint {checkpoint}.", extra={"model": model, "checkpoint": checkpoint})

    config = RestartConfig(name=model, checkpoint=checkpoint)
    training(config)

if __name__ == "__main__":
    app()
