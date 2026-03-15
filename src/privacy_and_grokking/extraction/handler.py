import tempfile
from collections import defaultdict
from pathlib import Path

import mlflow
import torch
import torch.nn.functional as F
from mlflow.tracking import MlflowClient
from torch import nn
from torch.utils.data import Subset
from tqdm import tqdm

from privacy_and_grokking.config import TrainConfig
from privacy_and_grokking.datasets import (
    create_masking,
    generate_datasets,
    mask_dataset,
)
from privacy_and_grokking.extraction.distribution_overlap import compute_distribution_overlap
from privacy_and_grokking.extraction.roc import (
    compute_roc_metrics_single_step,
)
from privacy_and_grokking.models import create_model
from privacy_and_grokking.utils import Logger, get_device, setup_mlflow

MERLIN_MORGAN_NOISY_SAMPLES = 100
MERLIN_MORGAN_NOISE_SCALE = 0.01


def _list_checkpoint_steps(run_id: str) -> list[int]:
    client = MlflowClient()
    artifacts = client.list_artifacts(run_id, path="checkpoints")
    steps = []
    for artifact in artifacts:
        name = artifact.path.split("/")[-1]
        if name.isdigit():
            steps.append(int(name))
    return sorted(steps)


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
    train, test = generate_datasets(cfg.dataset)
    masking = create_masking(
        config=cfg.dataset_mask,
        num_samples=len(train),
        num_classes=train.num_classes,
    )
    train_subset = mask_dataset(
        masking,
        train,
        cfg.dataset_mask_idx,
    )
    subsample_size = min(len(train_subset), len(test))
    train_sub = Subset(
        train_subset,
        _stratified_indices(train_subset, subsample_size),
    )
    test_sub = Subset(
        test,
        _stratified_indices(test, subsample_size),
    )
    train_loader = torch.utils.data.DataLoader(train_sub, batch_size=cfg.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_sub, batch_size=cfg.batch_size, shuffle=False)

    return train_loader, test_loader, train.input_shape, train.num_classes


def _iterate_dataloader(dataloader, device, model, last_step: bool):
    correct_probs: list[torch.Tensor] = []
    correct_logits: list[torch.Tensor] = []
    ce_losses: list[torch.Tensor] = []
    mse_losses: list[torch.Tensor] = []
    correctness_list: list[torch.Tensor] = []
    mm_ce_votes: list[float] = []
    mm_mse_votes: list[float] = []

    ce_criterion = nn.CrossEntropyLoss(reduction="none")
    mse_criterion = nn.MSELoss(reduction="none")

    buffers: dict[str, list[torch.Tensor]] = {}
    label_list: list[torch.Tensor] = []
    handles: list = []
    if last_step:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                key = name
                buffers[key] = []

                def _make_hook(k: str):
                    def _hook(_module: nn.Module, _inp: tuple, output: torch.Tensor) -> None:
                        buffers[k].append(output.detach().cpu())

                    return _hook

                handles.append(module.register_forward_hook(_make_hook(key)))
    try:
        with torch.no_grad():
            for x, y in dataloader:
                if last_step:
                    label_list.append(y.cpu())
                x, y = x.to(device), y.to(device)
                _logits = model(x)
                _probs = F.softmax(_logits, dim=1)
                _correct_probs = _probs.gather(1, y.view(-1, 1))
                _correct_logits = _logits.gather(1, y.view(-1, 1))
                _ce_losses = ce_criterion(_logits, y)
                _mse_losses = mse_criterion(
                    _logits,
                    F.one_hot(
                        y,
                        num_classes=_logits.size(1),
                    ).float(),
                ).gather(1, y.view(-1, 1))
                _correctly_classified = (_logits.argmax(dim=1) == y).float()

                for i in range(x.size(0)):
                    img = x[i]
                    label = y[i]
                    ce_loss = _ce_losses[i]
                    mse_loss = _mse_losses[i]
                    label_oh = F.one_hot(label, num_classes=_logits.size(1)).float()

                    noise = (
                        torch.randn(
                            (MERLIN_MORGAN_NOISY_SAMPLES, *img.shape),
                            device=device,
                        )
                        * MERLIN_MORGAN_NOISE_SCALE
                    )
                    noisy_imgs = img.unsqueeze(0) + noise
                    noisy_output = model(noisy_imgs)

                    noisy_ce = ce_criterion(
                        noisy_output,
                        label.repeat(MERLIN_MORGAN_NOISY_SAMPLES),
                    )
                    noisy_mse = mse_criterion(
                        noisy_output,
                        label_oh.repeat(MERLIN_MORGAN_NOISY_SAMPLES, 1),
                    ).sum(dim=1)

                    mm_ce_votes.append((noisy_ce > ce_loss).float().mean())
                    mm_mse_votes.append((noisy_mse > mse_loss).float().mean())

                correct_logits.append(_correct_logits)
                correct_probs.append(_correct_probs)
                ce_losses.append(_ce_losses)
                mse_losses.append(_mse_losses)
                correctness_list.append(_correctly_classified)
    finally:
        if last_step:
            for h in handles:
                h.remove()
            buffers = {k: torch.cat(v, dim=0) for k, v in buffers.items()}
            label_list = torch.cat(label_list, dim=0)

    cat_correct_probs = torch.cat(correct_probs, dim=0).squeeze()
    cat_correct_logits = torch.cat(correct_logits, dim=0).squeeze()
    cat_ce_losses = torch.cat(ce_losses, dim=0).squeeze()
    cat_mse_losses = torch.cat(mse_losses, dim=0).squeeze()
    cat_correctness_list = torch.cat(correctness_list, dim=0).squeeze()
    cat_mm_ce_votes = torch.tensor(mm_ce_votes)
    cat_mm_mse_votes = torch.tensor(mm_mse_votes)

    return (
        cat_correct_probs,
        cat_correct_logits,
        cat_ce_losses,
        cat_mse_losses,
        cat_correctness_list,
        cat_mm_ce_votes,
        cat_mm_mse_votes,
        buffers,
        label_list,
    )


def _extract_weight_norm(state_dict, step: int):
    norms: dict[str, float] = {}
    all_params = []

    for name, param in state_dict.items():
        norms[f"weight_norm/{name}"] = torch.linalg.norm(param.float()).item()
        all_params.append(param.float().flatten())

    if all_params:
        norms["weight_norm/total"] = torch.linalg.norm(torch.cat(all_params)).item()
    else:
        norms["weight_norm/total"] = 0.0
    mlflow.log_metrics(norms, step=step)


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
        model = create_model(
            name=cfg.model,
            input_dim=input_shape,
            num_classes=num_classes,
            initialization_scale=None,
        )
        model.to(device)
        model.load_state_dict(state_dict)
        model.eval()

        _extract_weight_norm(state_dict, step)

        last_step = step == steps[-1]
        tr_cp, tr_cl, tr_ce, tr_mse, tr_corr, tr_mm_ce, tr_mm_mse, tr_acts, tr_labels = (
            _iterate_dataloader(train, device, model, last_step)
        )
        te_cp, te_cl, te_ce, te_mse, te_corr, te_mm_ce, te_mm_mse, te_acts, te_labels = (
            _iterate_dataloader(test, device, model, last_step)
        )

        attacks = [
            ("mia_prob", tr_cp, te_cp),
            ("mia_logit", tr_cl, te_cl),
            ("mia_ce_loss", -tr_ce, -te_ce),
            ("mia_mse_loss", -tr_mse, -te_mse),
            ("mia_correctness", tr_corr, te_corr),
            ("mia_merlin_morgan_ce", tr_mm_ce, te_mm_ce),
            ("mia_merlin_morgan_mse", tr_mm_mse, te_mm_mse),
        ]

        roc_metrics: dict[str, float] = {}
        for prefix, tr_sig, te_sig in attacks:
            m = compute_roc_metrics_single_step(tr_sig, te_sig)
            for key, value in m.items():
                roc_metrics[f"{prefix}/{key}"] = value

        mlflow.log_metrics(roc_metrics, step=step)

        tr_ce_flat = tr_ce.squeeze().float()
        te_ce_flat = te_ce.squeeze().float()
        loss_dist_metrics: dict[str, float] = {
            "extraction.train.loss.mean": float(tr_ce_flat.mean().item()),
            "extraction.train.loss.std": float(tr_ce_flat.std().item()),
            "extraction.test.loss.mean": float(te_ce_flat.mean().item()),
            "extraction.test.loss.std": float(te_ce_flat.std().item()),
            "extraction.loss.overlap": compute_distribution_overlap(tr_ce_flat, te_ce_flat),
        }
        mlflow.log_metrics(loss_dist_metrics, step=step)

        if last_step:
            payload = {
                "train_activations": tr_acts,
                "test_activations": te_acts,
                "train_labels": tr_labels,
                "test_labels": te_labels,
                "step": step,
            }
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / f"{step}.pt"
                torch.save(payload, path)
                mlflow.log_artifact(str(path), artifact_path="activations")


def extraction_handler(exp_name: str, run_id: str) -> None:
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
