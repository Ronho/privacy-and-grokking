from collections import defaultdict
from privacy_and_grokking.metrics.norms import compute_weight_norms, compute_gradient_norms
from privacy_and_grokking.utils import eval_mode, get_device
import torch
import torch.nn as nn
import torch.nn.functional as F

from privacy_and_grokking.metrics.distribution_overlap import compute_distribution_overlap
from privacy_and_grokking.metrics.roc import compute_roc_metrics_single_step


def evaluate(model: nn.Module, key_prefix: str):
    metrics = {}
    with eval_mode(model):
        metrics.update(compute_weight_norms(model))
        metrics.update(compute_gradient_norms(model))
        
        process_loader()

    for key, value in metrics.items():
        metrics[f"{key_prefix}/{key}"] = value
        del metrics[key]
    return metrics

def process_loader(model: nn.Module, loader: torch.utils.data.DataLoader, compute_mm: bool):
    device = get_device()
    ce_criterion = nn.CrossEntropyLoss(reduction="none")
    mse_criterion = nn.MSELoss(reduction="none")

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

class MetricComputer:
    MERLIN_MORGAN_NOISY_SAMPLES = 100
    MERLIN_MORGAN_NOISE_SCALE = 0.01

    @staticmethod
    def _process_batch(
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        device: torch.device,
        compute_mm: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Process a single batch, returning per-sample signals.

        Returns dict with keys: prob, logit, ce_loss, mse_loss, correctness.
        If compute_mm=True, also returns mm_ce and mm_mse.
        """
        ce_criterion = nn.CrossEntropyLoss(reduction="none")
        mse_criterion = nn.MSELoss(reduction="none")

        with torch.no_grad():
            x, y = x.to(device), y.to(device)
            logit = model(x)
            prob = F.softmax(logit, dim=1)
            true_class_prob = prob.gather(1, y.view(-1, 1))
            true_class_logit = logit.gather(1, y.view(-1, 1))
            ce_loss = ce_criterion(logit, y)
            mse_loss = mse_criterion(
                logit,
                F.one_hot(y, num_classes=logit.size(1)).float(),
            ).gather(1, y.view(-1, 1))
            correct = (logit.argmax(dim=1) == y).float()

            result = {
                "prob": true_class_prob,
                "logit": true_class_logit,
                "ce_loss": ce_loss,
                "mse_loss": mse_loss,
                "correctness": correct,
            }

            if compute_mm:
                mm_ce_votes = []
                mm_mse_votes = []
                for i in range(x.size(0)):
                    img = x[i]
                    label = y[i]
                    ce_loss_i = ce_loss[i]
                    mse_loss_i = mse_loss[i]
                    label_oh = F.one_hot(label, num_classes=logit.size(1)).float()

                    noise = (
                        torch.randn(
                            (MetricComputer.MERLIN_MORGAN_NOISY_SAMPLES, *img.shape),
                            device=device,
                        )
                        * MetricComputer.MERLIN_MORGAN_NOISE_SCALE
                    )
                    noisy_imgs = img.unsqueeze(0) + noise
                    noisy_output = model(noisy_imgs)

                    noisy_ce = ce_criterion(
                        noisy_output,
                        label.repeat(MetricComputer.MERLIN_MORGAN_NOISY_SAMPLES),
                    )
                    noisy_mse = mse_criterion(
                        noisy_output,
                        label_oh.repeat(MetricComputer.MERLIN_MORGAN_NOISY_SAMPLES, 1),
                    ).sum(dim=1)

                    mm_ce_votes.append((noisy_ce > ce_loss_i).float().mean())
                    mm_mse_votes.append((noisy_mse > mse_loss_i).float().mean())

                result["mm_ce"] = torch.stack(mm_ce_votes)
                result["mm_mse"] = torch.stack(mm_mse_votes)

            return result

    @staticmethod
    def compute_attack_signals(
        model: nn.Module,
        loss_fn,
        loader: torch.utils.data.DataLoader,
        device: torch.device,
        compute_mm: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Compute per-sample signals for attack evaluation.

        Returns a dict with keys:
          - "prob": true class probability
          - "logit": true class logit
          - "ce_loss": cross-entropy loss
          - "mse_loss": MSE loss between logits and one-hot target
          - "correctness": 1 if correctly classified else 0
          - "mm_ce": Merlin-Morgan CE votes (if compute_mm=True)
          - "mm_mse": Merlin-Morgan MSE votes (if compute_mm=True)
        """
        model.eval()
        accumulated = {k: [] for k in ["prob", "logit", "ce_loss", "mse_loss", "correctness"]}
        if compute_mm:
            accumulated["mm_ce"] = []
            accumulated["mm_mse"] = []

        with torch.no_grad():
            for x, y in loader:
                batch_result = MetricComputer._process_batch(
                    model, x, y, device, compute_mm=compute_mm
                )
                for key in ["prob", "logit", "ce_loss", "mse_loss", "correctness"]:
                    accumulated[key].append(batch_result[key])
                if compute_mm:
                    accumulated["mm_ce"].append(batch_result["mm_ce"])
                    accumulated["mm_mse"].append(batch_result["mm_mse"])

        result = {}
        for key, tensors in accumulated.items():
            if tensors:
                if key in ["mm_ce", "mm_mse"]:
                    # tensors are lists of tensors, each tensor is 1D (per-sample votes)
                    # need to concatenate along first dimension
                    result[key] = torch.cat(tensors, dim=0).squeeze().cpu()
                else:
                    result[key] = torch.cat(tensors, dim=0).squeeze().cpu()
        return result

    @staticmethod
    def compute_attack_auc_metrics(
        train_signals: dict[str, torch.Tensor],
        test_signals: dict[str, torch.Tensor],
        include_mm: bool = False,
    ) -> dict[str, float]:
        """Compute AUC and TPR@FPR metrics for each attack signal.

        Returns a dict with keys like "attack/{signal}/auc", "attack/{signal}/tpr-at-{pct}-fpr".
        """
        metrics = {}
        attacks = [
            ("prob", train_signals["prob"], test_signals["prob"]),
            ("logit", train_signals["logit"], test_signals["logit"]),
            ("ce_loss", -train_signals["ce_loss"], -test_signals["ce_loss"]),
            ("mse_loss", -train_signals["mse_loss"], -test_signals["mse_loss"]),
            ("correctness", train_signals["correctness"], test_signals["correctness"]),
        ]
        if include_mm and "mm_ce" in train_signals:
            attacks.extend(
                [
                    ("mm_ce", train_signals["mm_ce"], test_signals["mm_ce"]),
                    ("mm_mse", train_signals["mm_mse"], test_signals["mm_mse"]),
                ]
            )

        for prefix, tr_sig, te_sig in attacks:
            m = compute_roc_metrics_single_step(tr_sig, te_sig)
            for key, value in m.items():
                metrics[f"attack/{prefix}/{key}"] = value
        return metrics

    @staticmethod
    def compute_basic_metrics(
        model: nn.Module,
        loss_fn,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device,
    ) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
        """Compute basic loss and accuracy metrics.

        Returns:
            metrics_dict: dict with keys like "train/loss_mean", "test/accuracy", etc.
            train_losses: tensor of per-sample train losses
            test_losses: tensor of per-sample test losses
        """
        model.eval()
        train_losses = []
        test_losses = []
        train_correct = 0
        train_total = 0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            # Train loader
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                losses = loss_fn(logits, y)
                train_losses.append(losses)
                preds = torch.argmax(logits, dim=1)
                train_correct += torch.sum(preds == y).item()
                train_total += x.size(0)

            # Test loader
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                losses = loss_fn(logits, y)
                test_losses.append(losses)
                preds = torch.argmax(logits, dim=1)
                test_correct += torch.sum(preds == y).item()
                test_total += x.size(0)

        train_losses_t = torch.cat(train_losses).cpu()
        test_losses_t = torch.cat(test_losses).cpu()

        metrics = {
            "train/loss_mean": train_losses_t.mean().item(),
            "train/loss_std": train_losses_t.std().item(),
            "train/accuracy": train_correct / train_total if train_total > 0 else 0.0,
            "test/loss_mean": test_losses_t.mean().item(),
            "test/loss_std": test_losses_t.std().item(),
            "test/accuracy": test_correct / test_total if test_total > 0 else 0.0,
            "loss/overlap": compute_distribution_overlap(train_losses_t, test_losses_t),
        }

        return metrics, train_losses_t, test_losses_t
