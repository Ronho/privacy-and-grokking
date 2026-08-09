from __future__ import annotations

import tempfile
from collections import defaultdict
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F

from privacy_and_grokking.metrics.config import MetricsConfig
from privacy_and_grokking.metrics.curvature import curvature
from privacy_and_grokking.metrics.distribution_overlap import (
    compute_distribution_overlap,
    compute_distribution_overlap_adaptive,
    compute_distribution_overlap_kde,
    compute_js_distance,
    compute_js_distance_adaptive,
    compute_js_distance_kde,
    compute_kl_divergence,
    compute_kl_divergence_adaptive,
    compute_kl_divergence_kde,
    compute_mmd,
    soft_distribution_overlap,
)
from privacy_and_grokking.metrics.neural_collapse import (
    compute_all_nc_metrics,
    compute_rnc1,
    compute_rnc1_train_mean,
)
from privacy_and_grokking.metrics.norms import compute_gradient_norms, compute_weight_norms
from privacy_and_grokking.metrics.optimizer_params import get_optimizer_internals
from privacy_and_grokking.metrics.roc import compute_roc_metrics_single_step
from privacy_and_grokking.utils import eval_mode, get_device

MERLIN_MORGAN_NOISY_SAMPLES = 100
MERLIN_MORGAN_NOISE_SCALE = 0.01


def _process_loader(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    compute_mm: bool,
    last_step: bool,
    collect_features: bool = False,
    collect_inputs: bool = False,
    normalization = None,
):
    device = get_device()
    ce_criterion = nn.CrossEntropyLoss(reduction="none")
    mse_criterion = nn.MSELoss(reduction="none")

    buffers_accum: dict[str, list[torch.Tensor]] = {}
    label_list_accum: list[torch.Tensor] = []
    input_list_accum: list[torch.Tensor] = []
    handles: list = []
    capture_state = {"enabled": True}
    _collect = last_step or collect_features
    if _collect:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                key = name
                buffers_accum[key] = []
                buffers_accum[f"{key}.input"] = []

                def _make_hook(k: str):
                    def _hook(_module: nn.Module, _inp: tuple, output: torch.Tensor) -> None:
                        if capture_state["enabled"]:
                            buffers_accum[k].append(output.detach())
                            buffers_accum[f"{k}.input"].append(_inp[0].detach())

                    return _hook

                handles.append(module.register_forward_hook(_make_hook(key)))

    result = defaultdict(list)

    if normalization is not None:
        norm_mean = torch.tensor(normalization.mean, device=device).view(-1, 1, 1)
        norm_std = torch.tensor(normalization.std, device=device).view(-1, 1, 1)

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if _collect:
            label_list_accum.append(y)
        if collect_inputs:
            input_list_accum.append(x.detach().reshape(x.size(0), -1))
        if normalization is not None:
            x = (x - norm_mean) / norm_std
        logit = model(x)
        prob = F.softmax(logit, dim=1)
        result["true_class_logit"].append(logit.gather(1, y.view(-1, 1)))
        result["max_logit"].append(logit.max(dim=1, keepdim=True).values)
        result["min_logit"].append(logit.min(dim=1, keepdim=True).values)
        result["true_class_prob"].append(prob.gather(1, y.view(-1, 1)))
        result["max_prob"].append(prob.max(dim=1, keepdim=True).values)
        result["min_prob"].append(prob.min(dim=1, keepdim=True).values)
        result["ce_loss"].append(ce_criterion(logit, y))
        result["mse_loss"].append(
            mse_criterion(
                logit,
                F.one_hot(y, num_classes=logit.size(1)).float(),
            ).gather(1, y.view(-1, 1))
        )
        result["correctness"].append((logit.argmax(dim=1) == y).float())

        if compute_mm:
            mm_ce_votes = []
            mm_mse_votes = []

            for i in range(x.size(0)):
                img = x[i]
                label = y[i]
                ce_loss_i = result["ce_loss"][-1][i].to(device)
                mse_loss_i = result["mse_loss"][-1][i].to(device)
                label_oh = F.one_hot(label, num_classes=logit.size(1)).float()

                noise = (
                    torch.randn(
                        (MERLIN_MORGAN_NOISY_SAMPLES, *img.shape),
                        device=device,
                    )
                    * MERLIN_MORGAN_NOISE_SCALE
                )
                noisy_imgs = img.unsqueeze(0) + noise
                if _collect:
                    capture_state["enabled"] = False
                try:
                    noisy_output = model(noisy_imgs)
                finally:
                    if _collect:
                        capture_state["enabled"] = True

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

    if _collect:
        for h in handles:
            h.remove()
        # Convert list of tensors to single tensor per layer
        buffers = {k: torch.cat(v, dim=0).cpu() for k, v in buffers_accum.items() if len(v) > 0}
        label_list = torch.cat(label_list_accum, dim=0).cpu()
    else:
        buffers = {}
        label_list = torch.tensor([])

    if collect_inputs and input_list_accum:
        input_list = torch.cat(input_list_accum, dim=0).cpu()
    else:
        input_list = torch.tensor([])

    result = {k: torch.cat(v, dim=0).cpu() for k, v in result.items()}

    return result, buffers, label_list, input_list


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
    metrics_config: MetricsConfig | None = None,
    in_canary_indices: list[int] | None = None,
    out_canary_loader: torch.utils.data.DataLoader | None = None,
    normalization = None,
) -> dict[str, float]:
    if metrics_config is None:
        metrics_config = MetricsConfig()

    metrics = {}
    with eval_mode(model):
        if metrics_config.weight_norms:
            metrics.update(compute_weight_norms(model))
        if metrics_config.gradient_norms:
            metrics.update(compute_gradient_norms(model))
        if metrics_config.optimizer_internals:
            metrics.update(get_optimizer_internals(optimizer))

        compute_mm = compute_heavy_metrics and metrics_config.merlin_morgan
        collect_features = compute_heavy_metrics and (
            metrics_config.neural_collapse
            or metrics_config.rnc1
            or metrics_config.rnc1_train_mean
            or metrics_config.nhsic
            or metrics_config.attack_distance_to_class_mean
            or metrics_config.attack_margin_distance_lf
        )
        collect_inputs = metrics_config.nhsic
        train_results, train_activations, train_labels, train_inputs = _process_loader(
            model, train_loader, compute_mm=compute_mm, last_step=last_step,
            collect_features=collect_features, collect_inputs=collect_inputs,
            normalization=normalization,
        )
        test_results, test_activations, test_labels, test_inputs = _process_loader(
            model, test_loader, compute_mm=compute_mm, last_step=last_step,
            collect_features=collect_features, collect_inputs=collect_inputs,
            normalization=normalization,
        )

        if metrics_config.loss_stats:
            metrics["train/loss/mse/mean"] = train_results["mse_loss"].mean()
            metrics["train/loss/mse/std"] = train_results["mse_loss"].std()
            metrics["test/loss/mse/mean"] = test_results["mse_loss"].mean()
            metrics["test/loss/mse/std"] = test_results["mse_loss"].std()

            metrics["train/loss/cross_entropy/mean"] = train_results["ce_loss"].mean()
            metrics["train/loss/cross_entropy/std"] = train_results["ce_loss"].std()
            metrics["test/loss/cross_entropy/mean"] = test_results["ce_loss"].mean()
            metrics["test/loss/cross_entropy/std"] = test_results["ce_loss"].std()

        if metrics_config.any_distribution_metric:
            for loss_key in ("mse", "ce"):
                t = train_results[f"{loss_key}_loss"]
                v = test_results[f"{loss_key}_loss"]
                if metrics_config.distribution_overlap:
                    metrics[f"loss/{loss_key}/overlap"] = compute_distribution_overlap(t, v)
                if metrics_config.distribution_overlap_adaptive:
                    metrics[f"loss/{loss_key}/overlap_adaptive"] = (
                        compute_distribution_overlap_adaptive(t, v)
                    )
                if metrics_config.distribution_overlap_kde:
                    metrics[f"loss/{loss_key}/overlap_kde"] = compute_distribution_overlap_kde(t, v)
                if metrics_config.soft_overlap:
                    metrics[f"loss/{loss_key}/soft_overlap"] = soft_distribution_overlap(
                        t, v
                    ).item()
                if metrics_config.kl_divergence:
                    metrics[f"loss/{loss_key}/kl_divergence"] = compute_kl_divergence(t, v)
                if metrics_config.kl_divergence_adaptive:
                    metrics[f"loss/{loss_key}/kl_divergence_adaptive"] = (
                        compute_kl_divergence_adaptive(t, v)
                    )
                if metrics_config.kl_divergence_kde:
                    metrics[f"loss/{loss_key}/kl_divergence_kde"] = compute_kl_divergence_kde(t, v)
                if metrics_config.js_distance:
                    metrics[f"loss/{loss_key}/js_distance"] = compute_js_distance(t, v)
                if metrics_config.js_distance_adaptive:
                    metrics[f"loss/{loss_key}/js_distance_adaptive"] = compute_js_distance_adaptive(
                        t, v
                    )
                if metrics_config.js_distance_kde:
                    metrics[f"loss/{loss_key}/js_distance_kde"] = compute_js_distance_kde(t, v)
                if metrics_config.mmd:
                    metrics[f"loss/{loss_key}/mmd"] = compute_mmd(t, v)

        if metrics_config.accuracy:
            train_accuracy = train_results["correctness"].sum() / len(
                train_results["correctness"]
            )
            test_accuracy = test_results["correctness"].sum() / len(test_results["correctness"])
            metrics["train/accuracy"] = train_accuracy
            metrics["test/accuracy"] = test_accuracy
            metrics["generalization_gap"] = train_accuracy - test_accuracy

        if metrics_config.any_attack_metric:
            attacks = []
            if metrics_config.attack_true_class_prob:
                attacks.append(
                    (
                        "true_class_prob",
                        train_results["true_class_prob"],
                        test_results["true_class_prob"],
                    )
                )
            if metrics_config.attack_true_class_logit:
                attacks.append(
                    (
                        "true_class_logit",
                        train_results["true_class_logit"],
                        test_results["true_class_logit"],
                    )
                )
            if metrics_config.attack_ce_loss:
                attacks.append(
                    (
                        "ce_loss",
                        -train_results["ce_loss"],
                        -test_results["ce_loss"],
                    )
                )
            if metrics_config.attack_mse_loss:
                attacks.append(
                    (
                        "mse_loss",
                        -train_results["mse_loss"],
                        -test_results["mse_loss"],
                    )
                )
            if metrics_config.attack_correctness:
                attacks.append(
                    (
                        "correctness",
                        train_results["correctness"],
                        test_results["correctness"],
                    )
                )

            if compute_heavy_metrics and metrics_config.merlin_morgan:
                attacks.extend(
                    [
                        ("mm_ce", train_results["mm_ce"], test_results["mm_ce"]),
                        ("mm_mse", train_results["mm_mse"], test_results["mm_mse"]),
                    ]
                )

            for prefix, train_sig, test_sig in attacks:
                m = compute_roc_metrics_single_step(train_sig, test_sig)
                for key, value in m.items():
                    metrics[f"attack/{prefix}/{key}"] = value

        if metrics_config.one_run_audit and in_canary_indices and out_canary_loader:
            from privacy_and_grokking.metrics.one_run_audit import compute_empirical_epsilon
            in_canary_losses = train_results["ce_loss"][in_canary_indices]
            out_results, _, _, _ = _process_loader(
                model, out_canary_loader, compute_mm=False, last_step=False, collect_features=False,
                normalization=normalization,
            )
            out_canary_losses = out_results["ce_loss"]
            audit_metrics = compute_empirical_epsilon(in_canary_losses, out_canary_losses, step)
            for k, v in audit_metrics.items():
                metrics[f"audit/{k}"] = v

    if compute_heavy_metrics and metrics_config.curvature:
        metrics.update(curvature(model, loss_fn, train_loader))

    if train_activations and ((compute_heavy_metrics and metrics_config.neural_collapse) or metrics_config.rnc1 or metrics_config.rnc1_train_mean):
        # Find the last linear layer's weight (classifier head).
        last_linear_name = None
        last_linear_weight = None
        last_linear_bias = None
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                last_linear_name = name
                last_linear_weight = module.weight.detach().cpu()
                last_linear_bias = module.bias.detach().cpu() if module.bias is not None else None

        # Determine the penultimate layer (input to the classifier).
        # We prefer the exact input to the last linear layer if available.
        penultimate = None
        train_feats = None
        if last_linear_name and f"{last_linear_name}.input" in train_activations:
            train_feats = train_activations[f"{last_linear_name}.input"]
        else:
            layer_names = [k for k in list(train_activations.keys()) if not str(k).endswith(".input")]
            if last_linear_name and last_linear_name in layer_names and len(layer_names) >= 2:
                penultimate_idx = layer_names.index(last_linear_name) - 1
                if penultimate_idx >= 0:
                    penultimate = layer_names[penultimate_idx]
                else:
                    penultimate = layer_names[0]
            elif len(layer_names) >= 2:
                penultimate = layer_names[-2]
            else:
                penultimate = layer_names[0] if layer_names else None

            if penultimate:
                train_feats = train_activations[penultimate]

        if train_feats is not None:
            if train_feats.ndim > 2:
                train_feats = train_feats.reshape(train_feats.size(0), -1)

            if compute_heavy_metrics and metrics_config.neural_collapse:
                nc = compute_all_nc_metrics(
                    train_feats, train_labels.long(), last_linear_weight, last_linear_bias
                )
                metrics["nc/nc0/train"] = nc.nc0
                metrics["nc/rnc1/train"] = nc.rnc1
                metrics["nc/nc1/train"] = nc.nc1
                metrics["nc/nc2/train"] = nc.nc2
                metrics["nc/nc2_equinorm/train"] = nc.nc2_equinorm
                metrics["nc/nc2_equiangular/train"] = nc.nc2_equiangular
                metrics["nc/nc3/train"] = nc.nc3
                metrics["nc/nc3_papyan/train"] = nc.nc3_papyan
                metrics["nc/nc4/train"] = nc.nc4
                metrics["nc/between_class_variance/train"] = nc.between_class_variance
                metrics["nc/within_class_variance/train"] = nc.within_class_variance
            elif metrics_config.rnc1:
                metrics["nc/rnc1/train"] = compute_rnc1(train_feats, train_labels.long())

            test_feats = None
            if test_activations and len(test_labels) > 0:
                if last_linear_name and f"{last_linear_name}.input" in test_activations:
                    test_feats = test_activations[f"{last_linear_name}.input"]
                elif penultimate and penultimate in test_activations:
                    test_feats = test_activations[penultimate]

                if test_feats is not None and test_feats.ndim > 2:
                    test_feats = test_feats.reshape(test_feats.size(0), -1)

            if test_feats is not None:
                if compute_heavy_metrics and metrics_config.neural_collapse:
                    nc_test = compute_all_nc_metrics(
                        test_feats, test_labels.long(), last_linear_weight, last_linear_bias
                    )
                    metrics["nc/nc0/test"] = nc_test.nc0
                    metrics["nc/rnc1/test"] = nc_test.rnc1
                    metrics["nc/nc1/test"] = nc_test.nc1
                    metrics["nc/nc2/test"] = nc_test.nc2
                    metrics["nc/nc2_equinorm/test"] = nc_test.nc2_equinorm
                    metrics["nc/nc2_equiangular/test"] = nc_test.nc2_equiangular
                    metrics["nc/nc3/test"] = nc_test.nc3
                    metrics["nc/nc3_papyan/test"] = nc_test.nc3_papyan
                    metrics["nc/nc4/test"] = nc_test.nc4
                    metrics["nc/between_class_variance/test"] = nc_test.between_class_variance
                    metrics["nc/within_class_variance/test"] = nc_test.within_class_variance
                elif metrics_config.rnc1:
                    metrics["nc/rnc1/test"] = compute_rnc1(test_feats, test_labels.long())

                if metrics_config.rnc1_train_mean:
                    metrics["nc/rnc1_train_mean/test"] = compute_rnc1_train_mean(
                        test_feats, test_labels.long(), train_feats, train_labels.long()
                    )

            # --- nHSIC metrics ---
            if metrics_config.nhsic:
                from privacy_and_grokking.metrics.nhsic import (
                    nhsic_features_vs_inputs,
                    nhsic_features_vs_labels,
                )

                max_samples = metrics_config.nhsic_max_samples

                # Train split
                nhsic_train = nhsic_features_vs_labels(
                    train_feats, train_labels.long(), max_samples=max_samples,
                )
                for k, v in nhsic_train.items():
                    metrics[f"nhsic/{k}/train"] = v

                if len(train_inputs) > 0:
                    nhsic_train_x = nhsic_features_vs_inputs(
                        train_feats, train_inputs, max_samples=max_samples,
                    )
                    for k, v in nhsic_train_x.items():
                        metrics[f"nhsic/{k}/train"] = v

                # Test split
                if test_feats is not None and len(test_labels) > 0:
                    nhsic_test = nhsic_features_vs_labels(
                        test_feats, test_labels.long(), max_samples=max_samples,
                    )
                    for k, v in nhsic_test.items():
                        metrics[f"nhsic/{k}/test"] = v

                    if len(test_inputs) > 0:
                        nhsic_test_x = nhsic_features_vs_inputs(
                            test_feats, test_inputs, max_samples=max_samples,
                        )
                        for k, v in nhsic_test_x.items():
                            metrics[f"nhsic/{k}/test"] = v

            if metrics_config.attack_distance_to_class_mean and test_feats is not None:
                from privacy_and_grokking.metrics.mia import distances_to_class_mean

                num_classes = int(train_labels.max().item() + 1)
                class_means = torch.zeros(num_classes, train_feats.shape[1], device=train_feats.device)
                for c in range(num_classes):
                    mask = train_labels == c
                    if mask.sum() > 0:
                        class_means[c] = train_feats[mask].float().mean(dim=0)

                train_dists = distances_to_class_mean(train_feats, train_labels, class_means)
                test_dists = distances_to_class_mean(test_feats, test_labels, class_means)

                all_train_flat = torch.cat(list(train_dists.values())) if train_dists else torch.tensor([])
                all_test_flat = torch.cat(list(test_dists.values())) if test_dists else torch.tensor([])

                if len(all_train_flat) > 0 and len(all_test_flat) > 0:
                    m = compute_roc_metrics_single_step(-all_train_flat, -all_test_flat)
                    for key, value in m.items():
                        metrics[f"attack/distance_to_class_mean/global/{key}"] = value

                for c in range(num_classes):
                    if c in train_dists and c in test_dists:
                        train_c = train_dists[c]
                        test_c = test_dists[c]
                        if len(train_c) > 0 and len(test_c) > 0:
                            m = compute_roc_metrics_single_step(-train_c, -test_c)
                            for key, value in m.items():
                                metrics[f"attack/distance_to_class_mean/class_{c}/{key}"] = value

            if metrics_config.attack_margin_distance_lf and test_feats is not None and last_linear_weight is not None:
                from privacy_and_grokking.metrics.mia import margin_distance_lf

                num_classes = int(train_labels.max().item() + 1)
                pool_features = torch.cat([train_feats.float(), test_feats.float()])
                pool_mean_norm = pool_features.norm(dim=1).mean().item()

                train_dists_lf = margin_distance_lf(
                    train_feats, train_labels, last_linear_weight, last_linear_bias, pool_mean_norm
                )
                test_dists_lf = margin_distance_lf(
                    test_feats, test_labels, last_linear_weight, last_linear_bias, pool_mean_norm
                )

                all_train_flat_lf = torch.cat(list(train_dists_lf.values())) if train_dists_lf else torch.tensor([])
                all_test_flat_lf = torch.cat(list(test_dists_lf.values())) if test_dists_lf else torch.tensor([])

                if len(all_train_flat_lf) > 0 and len(all_test_flat_lf) > 0:
                    m = compute_roc_metrics_single_step(-all_train_flat_lf, -all_test_flat_lf)
                    for key, value in m.items():
                        metrics[f"attack/margin_distance_lf/global/{key}"] = value
                    
                    train_median = all_train_flat_lf.median()
                    m_centered = compute_roc_metrics_single_step(
                        -torch.abs(all_train_flat_lf - train_median),
                        -torch.abs(all_test_flat_lf - train_median)
                    )
                    for key, value in m_centered.items():
                        metrics[f"attack/margin_distance_lf_centered/global/{key}"] = value

                for c in range(num_classes):
                    if c in train_dists_lf and c in test_dists_lf:
                        train_c_lf = train_dists_lf[c]
                        test_c_lf = test_dists_lf[c]
                        if len(train_c_lf) > 0 and len(test_c_lf) > 0:
                            m = compute_roc_metrics_single_step(-train_c_lf, -test_c_lf)
                            for key, value in m.items():
                                metrics[f"attack/margin_distance_lf/class_{c}/{key}"] = value

                            train_median_c = train_c_lf.median()
                            m_centered_c = compute_roc_metrics_single_step(
                                -torch.abs(train_c_lf - train_median_c),
                                -torch.abs(test_c_lf - train_median_c)
                            )
                            for key, value in m_centered_c.items():
                                metrics[f"attack/margin_distance_lf_centered/class_{c}/{key}"] = value

    keys = list(metrics.keys())
    for key in keys:
        metrics[f"{key_prefix}/{key}"] = metrics[key]
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
