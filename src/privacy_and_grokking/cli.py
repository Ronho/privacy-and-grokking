import os
from multiprocessing import Pool

from typer import Typer

from privacy_and_grokking.config import TrainingRegistry
from privacy_and_grokking.logger import get_logger
from privacy_and_grokking.training import RestartConfig
from privacy_and_grokking.training import train as training
from privacy_and_grokking.utils import Logger

app = Typer(name="Privacy and Grokking CLI", pretty_exceptions_enable=False)


@app.command()
def list_models():
    with Logger() as logger:
        TrainingRegistry.load_defaults()
        model_list = TrainingRegistry.list()
        logger.info(f"Available models ({len(model_list)}): {', '.join(model_list)}")


@app.command()
def train(
    exp_name: str,
    model: str,
    total_steps: int,
    mask_index: int,
    seed: int | None = None,
    run_name: str | None = None,
):
    TrainingRegistry.load_defaults()
    model_list = TrainingRegistry.list()
    if model not in model_list:
        raise ValueError(f"Unknown model '{model}' specified.")
    cfg = TrainingRegistry.get(model)
    if seed is not None:
        cfg.seed = seed
    if mask_index is not None:
        cfg.dataset_mask_idx = mask_index
    training(exp_name=exp_name, total_steps=total_steps, cfg=cfg, run_name=run_name)


@app.command()
def restart(exp_name: str, run_id: str, checkpoint: int, total_steps: int):
    cfg = RestartConfig(run_id=run_id, checkpoint=checkpoint)
    training(exp_name=exp_name, total_steps=total_steps, cfg=cfg, run_name="")


@app.command()
def extract(exp_name: str, run_id: str):
    from privacy_and_grokking.extraction import extraction_handler

    extraction_handler(exp_name, run_id)


@app.command()
def visualize(exp_name: str, run_ids: list[str]):
    from privacy_and_grokking.visualize import visualization_handler

    visualization_handler(exp_name, run_ids)


def _handle(line):
    line = line.strip()
    if line:
        logger = get_logger()
        logger.info("Processing command.", extra={"command": line})
        os.system(line)


@app.command()
def process(path: str, num_workers: int):
    with open(path) as f, Pool(num_workers) as pool:
        pool.map(_handle, f)


if __name__ == "__main__":
    app()
