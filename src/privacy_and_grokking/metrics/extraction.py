import tempfile
from collections import defaultdict
from pathlib import Path

import mlflow
import torch
from mlflow.tracking import MlflowClient
from torch.utils.data import Subset
from tqdm import tqdm

from privacy_and_grokking.config import TrainConfig
from privacy_and_grokking.metrics.evaluate import evaluate
from privacy_and_grokking.utils import Logger, get_device, setup_mlflow


def _list_checkpoint_steps(run_id: str) -> list[int]:
    client = MlflowClient()
    artifacts = client.list_artifacts(run_id, path="checkpoints")
    steps = []
    for artifact in artifacts:
        parts = artifact.path.split("/")
        # Expected patterns: "checkpoints/123" or "checkpoints/123/model.pth"
        if len(parts) >= 2 and parts[0] == "checkpoints":
            candidate = parts[1]
            if candidate.isdigit():
                steps.append(int(candidate))
    return sorted(set(steps))


def _stratified_indices(dataset, n: int) -> list[int]:
    class_indices: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        class_indices[int(label)].append(idx)

    num_classes = len(class_indices)
    per_class = n // num_classes

    generator = torch.Generator().manual_seed(4711)
    selected: list[int] = []
    # Note: this may select fewer than n samples if some classes have fewer than per_class samples.
    for indices in class_indices.values():
        t = torch.tensor(indices)
        perm = torch.randperm(len(t), generator=generator)
        chosen = t[perm[: min(per_class, len(t))]]
        selected.extend(chosen.tolist())

    return selected


def _get_datasets(cfg: TrainConfig):
    container = cfg.data()
    subsample_size = min(len(container.train), len(container.test))  # type: ignore
    train_sub = Subset(
        container.train,
        _stratified_indices(container.train, subsample_size),
    )
    test_sub = Subset(
        container.test,
        _stratified_indices(container.test, subsample_size),
    )
    pin_memory = torch.cuda.is_available()

    train_loader = torch.utils.data.DataLoader(
        train_sub, batch_size=cfg.batch_size, shuffle=False, pin_memory=pin_memory
    )
    test_loader = torch.utils.data.DataLoader(
        test_sub,
        batch_size=cfg.batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader, container.input_shape, container.num_classes


def _step_wise(run_id: str) -> None:
    logger = Logger.get()
    device = get_device()

    steps = _list_checkpoint_steps(run_id)
    if not steps:
        logger.warning("No checkpoints found for run.", run_id=run_id)
        return

    cfg = TrainConfig.model_validate(
        mlflow.artifacts.load_dict(f"runs:/{run_id}/training_config.json")
    )

    train, test, input_shape, num_classes = _get_datasets(cfg)
    loss_fn = cfg.loss()

    for step in tqdm(steps, desc="Extracting Data", unit="ckpt"):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_uri = f"runs:/{run_id}/checkpoints/{step}/model.pth"
            mlflow.artifacts.download_artifacts(
                artifact_uri=artifact_uri,
                dst_path=tmpdir,
            )
            model_path = Path(tmpdir) / "model.pth"
            state_dict = torch.load(
                model_path,
                map_location=device,
                weights_only=True,
            )
        model = cfg.model(
            input_dim=input_shape,
            num_classes=num_classes,
        )
        model.to(device)
        model.load_state_dict(state_dict)
        model.eval()

        evaluate(
            model,
            step,
            None,
            loss_fn,
            "extraction",
            train,
            test,
            True,
            last_step=step == steps[-1],
        )


def extraction_handler(exp_name: str, run_id: str) -> None:
    """Re-evaluate all saved checkpoints for a given run and log metrics.

    This is the post-training extraction entry point that loads each
    checkpoint, runs the full evaluation suite, and stores results in MLflow.
    """
    setup_mlflow(exp_name)
    with (
        Logger() as logger,
        mlflow.start_run(run_id=run_id),
    ):
        logger.info("Starting data extraction for run.", run_id=run_id)
        _step_wise(run_id)
        logger.info(
            "Completed data extraction for run.",
            run_id=run_id,
        )
