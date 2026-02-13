import os
from datetime import UTC, datetime
from multiprocessing import Pool
from typing import Literal

from typer import Typer

from .attacks import mia_simple
from .config import TrainConfig, TrainingRegistry
from .logger import get_logger, register_logger
from .path_keeper import get_path_keeper
from .training import RestartConfig
from .training import train as training
from .visualize import visualize_data

app = Typer(name="Privacy and Grokking CLI", pretty_exceptions_enable=False)


def _init(id: str):
    pk = get_path_keeper()
    pk.set_params({"run_id": id, "log_id": datetime.now(UTC).strftime("%Y-%m-%d-%H-%M-%S")})
    logger = register_logger(
        "default", log_file=pk.LOG, overwrite=True, log_level="DEBUG", channel="all", run_id=id
    )
    return logger


def _models(model: str, mask_index: int, existing: Literal["log", "raise", "ignore"] = "log") -> TrainConfig:
    TrainingRegistry.load_defaults()
    model_list = TrainingRegistry.list()

    if model not in model_list:
        raise ValueError(f"Unknown model '{model}' specified.")

    if existing != "ignore":
        pk = get_path_keeper()
        pk.set_params({"model": f"{model}_{mask_index}"})
        if not pk.TRAIN_CONFIG.exists():
            if existing == "log":
                logger = get_logger()
                logger.warning(
                    "Model was not trained yet and will be skipped.", extra={"model": model}
                )
            else:
                raise ValueError(f"Model '{model}' has not been trained yet.")

    return TrainingRegistry.get(model)


@app.command()
def train(id: str, model: str, mask_index: int):
    logger = _init(id)
    logger.info("Starting training run.", extra={"run": id, "model": model, "mask_index": mask_index})
    config = _models(model, mask_index, existing="ignore")
    training(cfg=config, mask_index=mask_index)
    logger.info("Training run completed.", extra={"run": id, "model": model, "mask_index": mask_index})


@app.command()
def restart(id: str, model: str, checkpoint: int, mask_index: int):
    logger = _init(id)
    logger.info(
        f"Restarting training for run {id}, model '{model}' from checkpoint {checkpoint}.",
        extra={"model": model, "checkpoint": checkpoint},
    )

    config = RestartConfig(name=model, checkpoint=checkpoint)
    training(cfg=config, mask_index=mask_index)


@app.command()
def attack(id: str, attack: str, model: str, mask_index: int):
    logger = _init(id)
    logger.info("Starting attack run.", extra={"run": id, "attack": attack, "model": model, "mask_index": mask_index})

    available_attacks = {
        "mia_simple": mia_simple
    }

    if attack not in available_attacks:
        raise ValueError(f"Unknown attack '{attack}' specified.")

    config = _models(model, mask_index, existing="log")
    func = available_attacks[attack]
    logger.info("Starting attack.", extra={"attack": attack, "model": config.name, "mask_index": mask_index})
    func(cfg=config, mask_index=mask_index)
    logger.info("Attack run completed.", extra={"run": id, "attack": attack, "model": model, "mask_index": mask_index})


@app.command()
def evaluate(id: str, models: list[str] | None = None):
    logger = _init(id)
    logger.info("Starting evaluation run.", extra={"run": id})

    visualize_data()

    # configs = _models(models, existing="log")
    # for config in configs:
    #     logger.info("Starting evaluation.", extra={"model": config.name})
    #     pk = get_path_keeper()
    #     pk.set_params({"model": config.name})
    #     visualize_training(cfg=config)
    # visualize_mia(cfgs=configs)

    # logger.info("Evaluation run completed.", extra={"run": id, "models": models})

def _handle(line):
    line = line.strip()
    if line:
        logger = get_logger()
        logger.info("Processing command.", extra={"command": line})
        os.system(line)

@app.command()
def process(path: str, num_workers: int):
    logger = _init("processing")
    logger.info("Starting processing run.", extra={"run": path})

    with open(path) as f, Pool(num_workers) as pool:
        pool.map(_handle, f)
    logger.info("Processing run completed.", extra={"run": path})

if __name__ == "__main__":
    app()
