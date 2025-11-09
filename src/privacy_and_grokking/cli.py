from typer import Typer

from .logger import register_logger
from .path_keeper import get_path_keeper
from .training import TrainingRegistry

app = Typer(name="Privacy and Grokking CLI", pretty_exceptions_enable=False)

@app.command()
def train(id: str, models: list[str] | None = None):
    paths = get_path_keeper()
    paths.set_params({"run_id": id})
    logger = register_logger(
        "default",
        log_file=paths.LOG,
        overwrite=True,
        log_level="DEBUG",
        channel="all",
        run_id=id
    )
    logger.info(f"Starting training for run {id}.")

    if models is None:
        models = TrainingRegistry.list_models()

    for model in models:
        if model in TrainingRegistry.list_models():
            logger.info(f"Training model '{model}'.", extra={"model": model})

            TrainingRegistry.get(model)()
        else:
            logger.error(f"Unknown model '{model}' specified for training. Ignoring.", extra={"model": model})

    logger.info(f"Training completed for run {id}.")

if __name__ == "__main__":
    app()
