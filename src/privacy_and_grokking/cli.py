import os
from multiprocessing import Pool
from pathlib import Path

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
    if mask_index is not None:
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
    all_activations: bool = False,
):
    from privacy_and_grokking.extraction import extraction_handler

    with Logger() as logger:
        logger.info("Starting extraction handler.", extra={"run_id": run_id})
        extraction_handler(exp_name, run_id, save_all_activations=all_activations)
        logger.info("Extraction handler completed.")


@app.command()
def visualize(
    exp_name: str,
    run_ids: list[str],
    tsne_video: bool = False,
):
    from privacy_and_grokking.visualize import visualization_handler

    with Logger() as logger:
        logger.info("Starting visualization handler.", extra={"run_ids": run_ids})
        visualization_handler(exp_name, run_ids, tsne_video=tsne_video)
        logger.info("Visualization handler completed.")


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
