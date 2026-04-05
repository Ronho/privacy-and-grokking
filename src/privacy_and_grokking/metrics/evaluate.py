import tempfile
from collections import defaultdict
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F

from privacy_and_grokking.metrics.curvature import curvature
from privacy_and_grokking.metrics.distribution_overlap import compute_distribution_overlap
from privacy_and_grokking.metrics.norms import compute_gradient_norms, compute_weight_norms
from privacy_and_grokking.metrics.optimizer_params import get_optimizer_internals
from privacy_and_grokking.metrics.roc import compute_roc_metrics_single_step
from privacy_and_grokking.utils import eval_mode, get_device

MERLIN_MORGAN_NOISY_SAMPLES = 100
MERLIN_MORGAN_NOISE_SCALE = 0.01

def _process_loader(model: nn.Module, loader: torch.utils.data.DataLoader, compute_mm: bool, last_step: bool):
    device = get_device()
    ce_criterion = nn.CrossEntropyLoss(reduction="none")
    mse_criterion = nn.MSELoss(reduction="none")

    buffers_accum: dict[str, list[torch.Tensor]] = {}
    label_list_accum: list[torch.Tensor] = []
    handles: list = []
    if last_step:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                key = name
                buffers_accum[key] = []

                def _make_hook(k: str):
                    def _hook(_module: nn.Module, _inp: tuple, output: torch.Tensor) -> None:
                        buffers_accum[k].append(output.detach().cpu())

                    return _hook

                handles.append(module.register_forward_hook(_make_hook(key)))

    result = defaultdict(list)

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logit = model(x)
        prob = F.softmax(logit, dim=1)
        result["true_class_logit"].append(logit.gather(1, y.view(-1, 1)))
        result["max_logit"].append(logit.max(dim=1, keepdim=True).values)
        result["min_logit"].append(logit.min(dim=1, keepdim=True).values)
        result["true_class_prob"].append(prob.gather(1, y.view(-1, 1)))
        result["max_prob"].append(prob.max(dim=1, keepdim=True).values)
        result["min_prob"].append(prob.min(dim=1, keepdim=True).values)
        result["ce_loss"].append(ce_criterion(logit, y))
        result["mse_loss"].append(mse_criterion(
            logit,
            F.one_hot(y, num_classes=logit.size(1)).float(),
        ).gather(1, y.view(-1, 1)))
        result["correctness"].append((logit.argmax(dim=1) == y).float())

        if compute_mm:
            mm_ce_votes = []
            mm_mse_votes = []

            for i in range(x.size(0)):
                img = x[i]
                label = y[i]
                ce_loss_i = result["ce_loss"][i]
                mse_loss_i = result["mse_loss"][i]
                label_oh = F.one_hot(label, num_classes=logit.size(1)).float()

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

                mm_ce_votes.append((noisy_ce > ce_loss_i).float().mean())
                mm_mse_votes.append((noisy_mse > mse_loss_i).float().mean())

            result["mm_ce"].append(torch.stack(mm_ce_votes))
            result["mm_mse"].append(torch.stack(mm_mse_votes))

    if last_step:
        for h in handles:
            h.remove()
        # Convert list of tensors to single tensor per layer
        buffers = {k: torch.cat(v, dim=0) for k, v in buffers_accum.items()}
        label_list = torch.cat(label_list_accum, dim=0)
    else:
        buffers = {}
        label_list = torch.tensor([])

    result = {k: torch.cat(v, dim=0) for k, v in result.items()}

    return result, buffers, label_list

def evaluate(
    model: nn.Module,
    step: int,
    optimizer,
    loss_fn,
    key_prefix: str,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    compute_heavy_metrics: bool,
    last_step: bool,
) -> dict[str, float]:
    metrics = {}
    with eval_mode(model):
        metrics.update(compute_weight_norms(model))
        metrics.update(compute_gradient_norms(model))
        metrics.update(get_optimizer_internals(optimizer))
        metrics.update(curvature(model, loss_fn, train_loader))

        train_results, train_activations, train_labels = _process_loader(model, train_loader, compute_mm=compute_heavy_metrics, last_step=last_step)
        test_results, test_activations, test_labels = _process_loader(model, test_loader, compute_heavy_metrics, last_step=last_step)
        metrics["train/loss/mse/mean"] = train_results["mse_loss"].mean()
        metrics["train/loss/mse/std"] = train_results["mse_loss"].std()
        metrics["test/loss/mse/mean"] = test_results["mse_loss"].mean()
        metrics["test/loss/mse/std"] = test_results["mse_loss"].std()

        metrics["train/loss/ce/mean"] = train_results["ce_loss"].mean()
        metrics["train/loss/ce/std"] = train_results["ce_loss"].std()
        metrics["test/loss/ce/mean"] = test_results["ce_loss"].mean()
        metrics["test/loss/ce/std"] = test_results["ce_loss"].std()

        metrics["loss/mse/overlap"] = compute_distribution_overlap(train_results["mse_loss"], test_results["mse_loss"])
        metrics["loss/ce/overlap"] = compute_distribution_overlap(train_results["ce_loss"], test_results["ce_loss"])

        metrics["train/accuracy"] = train_results["correctness"].sum() / train_results["correctness"].count()
        metrics["test/accuracy"] = test_results["correctness"].sum() / test_results["correctness"].count()

        attacks = [
            ("prob", train_results["prob"], test_results["prob"]),
            ("logit", train_results["logit"], test_results["logit"]),
            ("ce_loss", train_results["ce_loss"], test_results["ce_loss"]),
            ("mse_loss", train_results["mse_loss"], test_results["mse_loss"]),
            ("correctness", train_results["correctness"], test_results["correctness"]),
            ("logit", train_results["prob"], test_results["prob"]),
            ("logit", train_results["prob"], test_results["prob"]),
            ("logit", train_results["prob"], test_results["prob"]),
            ("logit", train_results["prob"], test_results["prob"]),
        ]

        if compute_heavy_metrics:
            attacks.extend([
                ("mm_ce", train_results["mm_ce"], test_results["mm_ce"]),
                ("mm_mse", train_results["mm_mse"], test_results["mm_mse"]),
            ])

        for prefix, train_sig, test_sig in attacks:
            m = compute_roc_metrics_single_step(train_sig, test_sig)
            for key, value in m.items():
                metrics[f"attack/{prefix}/{key}"] = value

    for key, value in metrics.items():
        metrics[f"{key_prefix}/{key}"] = value
        del metrics[key]

    # For safety
    optimizer.zero_grad()

    mlflow.log_metrics(metrics, step=step)
    if last_step:
        payload = {
            "train_activations": train_activations,
            "test_activations": test_activations,
            "train_labels": train_labels,
            "test_labels": test_labels,
            "step": step,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / f"{step}.pt"
            torch.save(payload, path)
            mlflow.log_artifact(str(path), artifact_path="activations")

    return metrics
