import os
from datetime import UTC, datetime
from multiprocessing import Pool
from typing import Literal

from typer import Typer

from privacy_and_grokking.attacks import mia_merlin_morgan, mia_rmia, mia_simple
from privacy_and_grokking.config import TrainConfig, TrainingRegistry
from privacy_and_grokking.logger import get_logger, register_logger
from privacy_and_grokking.path_keeper import get_path_keeper
from privacy_and_grokking.training import RestartConfig
from privacy_and_grokking.training import train as training
from privacy_and_grokking.utils import Logger
from privacy_and_grokking.visualize import visualize_data, visualize_evaluation

app = Typer(name="Privacy and Grokking CLI", pretty_exceptions_enable=False)


def _init(id: str):
    pk = get_path_keeper()
    pk.set_params({"run_id": id, "log_id": datetime.now(UTC).strftime("%Y-%m-%d-%H-%M-%S")})
    logger = register_logger(
        "default", log_file=pk.LOG, overwrite=True, log_level="DEBUG", channel="all", run_id=id
    )
    return logger


def _models(
    model: str, mask_index: int, existing: Literal["log", "raise", "ignore"] = "log"
) -> TrainConfig:
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

    config = TrainingRegistry.get(model)
    config.dataset_mask_idx = mask_index

    return config


@app.command()
def list_models():
    with Logger() as logger:
        TrainingRegistry.load_defaults()
        model_list = TrainingRegistry.list()
        print("Available models:")
        for model in model_list:
            logger.info(f"- {model}")


@app.command()
def train(exp_name: str, model: str, total_steps: int, mask_index: int, seed: int | None = None, run_name: str | None = None):
    cfg = _models(model, mask_index, existing="ignore")
    if seed is not None:
        cfg.seed = seed
    training(exp_name=exp_name, total_steps=total_steps, cfg=cfg, run_name=run_name)

@app.command()
def restart(exp_name: str, run_id: str, checkpoint: int, total_steps: int):
    cfg = RestartConfig(run_id=run_id, checkpoint=checkpoint)
    training(exp_name=exp_name, total_steps=total_steps, cfg=cfg, run_name="")


@app.command()
def attack(id: str, attack: str, model: str, mask_index: int):
    logger = _init(id)
    logger.info(
        "Starting attack run.",
        extra={"run": id, "attack": attack, "model": model, "mask_index": mask_index},
    )

    available_attacks = {
        "mia_simple": mia_simple,
        "mia_rmia": mia_rmia,
        "mia_merlin_morgan": mia_merlin_morgan,
    }

    if attack not in available_attacks:
        raise ValueError(f"Unknown attack '{attack}' specified.")

    config = _models(model, mask_index, existing="log")
    func = available_attacks[attack]
    logger.info(
        "Starting attack.", extra={"attack": attack, "model": config.name, "mask_index": mask_index}
    )
    func(cfg=config)
    logger.info(
        "Attack run completed.",
        extra={"run": id, "attack": attack, "model": model, "mask_index": mask_index},
    )


@app.command()
def evaluate(id: str, overwrite: bool = False):
    logger = _init(id)
    logger.info("Starting evaluation run.", extra={"run": id})

    visualize_data(overwrite=overwrite)

    pk = get_path_keeper()
    configs = []
    for model_path in pk.MODEL_FOLDER.iterdir():
        pk.set_params({"model": model_path.name})
        if pk.TRAIN_CONFIG.exists():
            try:
                config = TrainConfig.model_validate_json(pk.TRAIN_CONFIG.read_text())
                configs.append(config)
            except Exception as e:
                logger.error(f"Failed to load config from {pk.TRAIN_CONFIG}: {e}")

    if configs:
        logger.info(f"Found {len(configs)} models. Starting visualization.")
        visualize_evaluation(cfgs=configs, overwrite=overwrite)

    logger.info("Evaluation run completed.", extra={"run": id})


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
