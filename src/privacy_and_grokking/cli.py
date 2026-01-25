from datetime import datetime, timezone
from typer import Typer
from typing import Literal

from .attacks import mia_threshold, mia_rmia
from .config import TrainingRegistry, TrainConfig
from .logger import register_logger, get_logger
from .path_keeper import get_path_keeper
from .training import train as training, RestartConfig
from .visualize import visualize_data, visualize_training, visualize_mia


app = Typer(name="Privacy and Grokking CLI", pretty_exceptions_enable=False)

def _init(id: str):
    pk = get_path_keeper()
    pk.set_params({"run_id": id, "log_id": datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")})
    logger = register_logger(
        "default",
        log_file=pk.LOG,
        overwrite=True,
        log_level="DEBUG",
        channel="all",
        run_id=id
    )
    return logger

def _models(models: list[str] | None, existing: Literal["log", "raise", "ignore"] = "log") -> list[TrainConfig]:
    TrainingRegistry.load_defaults()
    model_list = TrainingRegistry.list()

    if models is None:
        models = model_list

    configs = []
    for model in models:
        if model not in model_list:
            raise ValueError(f"Unknown model '{model}' specified.")
        if existing != "ignore":
            pk = get_path_keeper()
            pk.set_params({"model": model})
            if not pk.TRAIN_CONFIG.exists():
                if existing == "log":
                    logger = get_logger()
                    logger.warning("Model was not trained yet and will be skipped.", extra={"model": model})
                    continue
                else:
                    raise ValueError(f"Model '{model}' has not been trained yet.")
        configs.append(TrainingRegistry.get(model))

    return configs


@app.command()
def train(id: str, models: list[str] | None = None):
    logger = _init(id)
    logger.info("Starting training run.", extra={"run": id, "models": models})

    configs = _models(models, existing="ignore")
    for config in configs:
        logger.info("Starting training.", extra={"model": config.name})
        training(config)

    logger.info("Training run completed.", extra={"run": id, "models": models})

@app.command()
def restart(id: str, model: str, checkpoint: int):
    logger = _init(id)
    logger.info(f"Restarting training for run {id}, model '{model}' from checkpoint {checkpoint}.", extra={"model": model, "checkpoint": checkpoint})

    config = RestartConfig(name=model, checkpoint=checkpoint)
    training(config)

@app.command()
def attack(attack_name: str, id: str, models: list[str] | None = None):
    logger = _init(id)
    logger.info("Starting attack run.", extra={"attack": attack_name, "run": id, "model": models})

    func = None
    match attack_name:
        case "mia_threshold":
            func = mia_threshold
        case "mia_rmia":
            func = mia_rmia
        case _:
            raise ValueError(f"Unknown attack '{attack_name}' specified.")

    configs = _models(models, existing="raise")
    for config in configs:
        logger.info("Starting attack.", extra={"attack": attack_name, "model": config.name})
        func(cfg=config)

    logger.info("Attack run completed.", extra={"attack": attack_name, "run": id, "model": models})

@app.command()
def evaluate(id: str, models: list[str] | None = None):
    logger = _init(id)
    logger.info("Starting evaluation run.", extra={"run": id})

    visualize_data()

    configs = _models(models, existing="log")
    for config in configs:
        logger.info("Starting evaluation.", extra={"model": config.name})
        pk = get_path_keeper()
        pk.set_params({"model": config.name})
        visualize_training(cfg=config)
    visualize_mia(cfgs=configs)
    
    logger.info("Evaluation run completed.", extra={"run": id, "models": models})


if __name__ == "__main__":
    app()
