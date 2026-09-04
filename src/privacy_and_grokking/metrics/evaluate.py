from collections import defaultdict

import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F

from privacy_and_grokking.datasets.sets.base import Normalization
from privacy_and_grokking.metrics.config import MetricsConfig
from privacy_and_grokking.metrics.curvature import curvature
from privacy_and_grokking.metrics.distribution_overlap import (
    compute_distribution_overlap,
    compute_median_heuristic,
    compute_mmd,
    subsample_tensor,
)
from privacy_and_grokking.metrics.neural_collapse import compute_all_nc_metrics
from privacy_and_grokking.metrics.norms import compute_gradient_norms, compute_weight_norms
from privacy_and_grokking.metrics.roc import compute_roc_metrics_single_step
from privacy_and_grokking.models.base import ModelBase
from privacy_and_grokking.utils import eval_mode, get_device


def _process_loader(
    model: ModelBase,
    loader: torch.utils.data.DataLoader,
    collect_penultimate_layer_features: bool,
    normalization: Normalization | None,
):
    device = get_device()
    ce_criterion = nn.CrossEntropyLoss(reduction="none")
    mse_criterion = nn.MSELoss(reduction="none")

    if normalization is not None:
        norm_mean = torch.tensor(normalization.mean, device=device).view(-1, 1, 1)
        norm_std = torch.tensor(normalization.std, device=device).view(-1, 1, 1)
    else:
        norm_mean = None
        norm_std = None

    label_list_accum: list[torch.Tensor] = []
    feature_list_accum: list[torch.Tensor] = []
    logits_accum: list[torch.Tensor] = []
    result = defaultdict(list)

    expected_keys = [
        "true_class_logit",
        "max_logit",
        "min_logit",
        "true_class_prob",
        "max_prob",
        "min_prob",
        "ce_loss",
        "mse_loss",
        "correctness",
    ]
    for k in expected_keys:
        result[k] = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if normalization is not None:
            x = (x - norm_mean) / norm_std
        label_list_accum.append(y)
        logit, features = model(x, verbose=True)
        prob = F.softmax(logit, dim=1)
        logits_accum.append(logit)
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
        if collect_penultimate_layer_features:
            feature_list_accum.append(features)

    label_list = (
        torch.cat(label_list_accum, dim=0).cpu()
        if label_list_accum
        else torch.tensor([], dtype=torch.long)
    )
    if collect_penultimate_layer_features:
        feature = (
            torch.cat(feature_list_accum, dim=0).cpu() if feature_list_accum else torch.tensor([])
        )
    else:
        feature = torch.tensor([])

    logits = torch.cat(logits_accum, dim=0).cpu() if logits_accum else torch.tensor([])
    result = {k: torch.cat(v, dim=0).cpu() if v else torch.tensor([]) for k, v in result.items()}

    return result, feature, label_list, logits


def _evaluate_attacks(
    in_results: dict[str, torch.Tensor],
    out_results: dict[str, torch.Tensor],
    prefix_template: str,
    metrics_config: MetricsConfig,
) -> dict[str, float]:
    attacks = []
    if (
        metrics_config.attack_true_class_prob
        and "true_class_prob" in in_results
        and "true_class_prob" in out_results
    ):
        attacks.append(
            (
                prefix_template.format(metric="true_class_prob"),
                in_results["true_class_prob"],
                out_results["true_class_prob"],
            )
        )
    if (
        metrics_config.attack_true_class_logit
        and "true_class_logit" in in_results
        and "true_class_logit" in out_results
    ):
        attacks.append(
            (
                prefix_template.format(metric="true_class_logit"),
                in_results["true_class_logit"],
                out_results["true_class_logit"],
            )
        )
    if (
        metrics_config.attack_ce_loss
        and "ce_loss" in in_results
        and "ce_loss" in out_results
    ):
        attacks.append(
            (
                prefix_template.format(metric="ce_loss"),
                -in_results["ce_loss"],
                -out_results["ce_loss"],
            )
        )
    if (
        metrics_config.attack_mse_loss
        and "mse_loss" in in_results
        and "mse_loss" in out_results
    ):
        attacks.append(
            (
                prefix_template.format(metric="mse_loss"),
                -in_results["mse_loss"],
                -out_results["mse_loss"],
            )
        )
    if (
        metrics_config.attack_correctness
        and "correctness" in in_results
        and "correctness" in out_results
    ):
        attacks.append(
            (
                prefix_template.format(metric="correctness"),
                in_results["correctness"],
                out_results["correctness"],
            )
        )

    attack_metrics = {}
    for prefix, in_sig, out_sig in attacks:
        if len(in_sig) > 0 and len(out_sig) > 0:
            m = compute_roc_metrics_single_step(in_sig, out_sig)
            for key, value in m.items():
                attack_metrics[f"attack/{prefix}/{key}"] = value
    return attack_metrics


def _evaluate_distribution_overlap(
    in_results: dict[str, torch.Tensor],
    out_results: dict[str, torch.Tensor],
    prefix_template: str,
) -> dict[str, float]:
    overlap_metrics = {}
    for loss_key in ("mse", "ce"):
        k = f"{loss_key}_loss"
        if k in in_results and k in out_results:
            in_loss = in_results[k]
            out_loss = out_results[k]
            if len(in_loss) > 0 and len(out_loss) > 0:
                overlap_metrics[f"loss/{prefix_template.format(loss_key=loss_key)}/overlap"] = (
                    compute_distribution_overlap(in_loss, out_loss)
                )
    return overlap_metrics


def evaluate(
    model: nn.Module,
    step: int,
    optimizer,
    loss_fn,
    key_prefix: str,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    compute_heavy_metrics: bool,
    num_classes: int,
    metrics_config: MetricsConfig | None = None,
    train_canary_loader: torch.utils.data.DataLoader | None = None,
    test_canary_loader: torch.utils.data.DataLoader | None = None,
    normalization: Normalization | None = None,
) -> dict[str, float]:
    if metrics_config is None:
        metrics_config = MetricsConfig()

    metrics = {}
    with eval_mode(model):
        if metrics_config.weight_norms:
            metrics.update(compute_weight_norms(model))
        if metrics_config.gradient_norms:
            metrics.update(compute_gradient_norms(model))

        collect_features = compute_heavy_metrics and (
            metrics_config.neural_collapse
            or metrics_config.attack_distance_to_class_mean
            or metrics_config.attack_margin_distance_lf
        )

        train_results, train_activations, train_labels, train_logits = _process_loader(
            model,
            train_loader,
            collect_penultimate_layer_features=collect_features,
            normalization=normalization,
        )
        test_results, test_activations, test_labels, test_logits = _process_loader(
            model,
            test_loader,
            collect_penultimate_layer_features=collect_features,
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
                train_loss = train_results[f"{loss_key}_loss"]
                test_loss = test_results[f"{loss_key}_loss"]
                if metrics_config.distribution_overlap:
                    metrics[f"loss/{loss_key}/overlap"] = compute_distribution_overlap(
                        train_loss, test_loss
                    )
                if metrics_config.mmd:
                    subsampled_train_loss = subsample_tensor(train_loss, max_samples=1000)
                    subsampled_test_loss = subsample_tensor(test_loss, max_samples=1000)
                    bandwidth = compute_median_heuristic(
                        subsampled_train_loss, subsampled_test_loss
                    )
                    metrics[f"loss/{loss_key}/mmd/bw"] = bandwidth
                    metrics[f"loss/{loss_key}/mmd/bw_0.5"] = compute_mmd(
                        subsampled_train_loss, subsampled_test_loss, 0.5 * bandwidth
                    )
                    metrics[f"loss/{loss_key}/mmd/bw_1.0"] = compute_mmd(
                        subsampled_train_loss, subsampled_test_loss, bandwidth
                    )
                    metrics[f"loss/{loss_key}/mmd/bw_2.0"] = compute_mmd(
                        subsampled_train_loss, subsampled_test_loss, 2.0 * bandwidth
                    )

        if metrics_config.accuracy:
            train_accuracy = train_results["correctness"].sum() / len(train_results["correctness"])
            test_accuracy = test_results["correctness"].sum() / len(test_results["correctness"])
            metrics["train/accuracy"] = train_accuracy
            metrics["test/accuracy"] = test_accuracy
            metrics["generalization_gap"] = train_accuracy - test_accuracy

        if metrics_config.any_attack_metric:
            metrics.update(
                _evaluate_attacks(
                    train_results,
                    test_results,
                    "{metric}",
                    metrics_config,
                )
            )

        train_canary_results = None
        if train_canary_loader is not None:
            train_canary_results, _, _, _ = _process_loader(
                model,
                train_canary_loader,
                collect_penultimate_layer_features=False,
                normalization=normalization,
            )
            in_canary_correctness = train_canary_results["correctness"]
            if len(in_canary_correctness) > 0:
                metrics["train/canary_accuracy"] = in_canary_correctness.float().mean().item()
            if "mse_loss" in train_canary_results and len(train_canary_results["mse_loss"]) > 0:
                metrics["train/canary_loss/mse/mean"] = (
                    train_canary_results["mse_loss"].mean().item()
                )
                metrics["train/canary_loss/mse/std"] = train_canary_results["mse_loss"].std().item()
            if "ce_loss" in train_canary_results and len(train_canary_results["ce_loss"]) > 0:
                metrics["train/canary_loss/cross_entropy/mean"] = (
                    train_canary_results["ce_loss"].mean().item()
                )
                metrics["train/canary_loss/cross_entropy/std"] = (
                    train_canary_results["ce_loss"].std().item()
                )

        test_canary_results = None
        if test_canary_loader is not None:
            test_canary_results, _, _, _ = _process_loader(
                model,
                test_canary_loader,
                collect_penultimate_layer_features=False,
                normalization=normalization,
            )
            out_canary_correctness = test_canary_results["correctness"]
            if len(out_canary_correctness) > 0:
                metrics["test/canary_accuracy"] = out_canary_correctness.float().mean().item()
            if "mse_loss" in test_canary_results and len(test_canary_results["mse_loss"]) > 0:
                metrics["test/canary_loss/mse/mean"] = test_canary_results["mse_loss"].mean().item()
                metrics["test/canary_loss/mse/std"] = test_canary_results["mse_loss"].std().item()
            if "ce_loss" in test_canary_results and len(test_canary_results["ce_loss"]) > 0:
                metrics["test/canary_loss/cross_entropy/mean"] = (
                    test_canary_results["ce_loss"].mean().item()
                )
                metrics["test/canary_loss/cross_entropy/std"] = (
                    test_canary_results["ce_loss"].std().item()
                )

        if train_canary_results is not None:
            train_plus_canary_results = {
                k: torch.cat([train_results[k], train_canary_results[k]], dim=0)
                for k in train_results
                if k in train_canary_results
            }

            if metrics_config.any_attack_metric:
                metrics.update(
                    _evaluate_attacks(
                        train_canary_results,
                        test_results,
                        "train_canary_vs_test/{metric}",
                        metrics_config,
                    )
                )
                metrics.update(
                    _evaluate_attacks(
                        train_plus_canary_results,
                        test_results,
                        "train_plus_canary_vs_test/{metric}",
                        metrics_config,
                    )
                )

            if metrics_config.distribution_overlap:
                metrics.update(
                    _evaluate_distribution_overlap(
                        train_canary_results,
                        test_results,
                        "train_canary_vs_test/{loss_key}",
                    )
                )
                metrics.update(
                    _evaluate_distribution_overlap(
                        train_plus_canary_results,
                        test_results,
                        "train_plus_canary_vs_test/{loss_key}",
                    )
                )

        if train_canary_results is not None and test_canary_results is not None:
            if metrics_config.any_attack_metric:
                metrics.update(
                    _evaluate_attacks(
                        train_canary_results,
                        test_canary_results,
                        "canary_{metric}",
                        metrics_config,
                    )
                )

            if metrics_config.distribution_overlap:
                metrics.update(
                    _evaluate_distribution_overlap(
                        train_canary_results,
                        test_canary_results,
                        "canary_{loss_key}",
                    )
                )

    if compute_heavy_metrics and metrics_config.curvature:
        metrics.update(curvature(model, loss_fn, train_loader))

    if (
        train_activations.numel() > 0
        and test_activations.numel() > 0
        and compute_heavy_metrics
        and metrics_config.neural_collapse
    ):
        train_predictions = (
            train_logits.argmax(dim=1) if train_logits.numel() > 0 else torch.tensor([])
        )
        nc = compute_all_nc_metrics(
            train_activations,
            train_labels,
            test_activations,
            test_labels,
            train_predictions,
            model.classifier().weight,
        )
        metrics["nc/rnc1/train"] = nc.rnc1_train
        metrics["nc/rnc1/test"] = nc.rnc1_test
        metrics["nc/rnc1/train_mean_test_variance"] = nc.rnc1_train_mean_test_variance
        metrics["nc/rnc1/train_impl"] = nc.rnc1_train_impl
        metrics["nc/rnc1/test_impl"] = nc.rnc1_test_impl
        metrics["nc/rnc1/train_mean_test_variance_impl"] = nc.rnc1_train_mean_test_variance_impl
        metrics["nc/nc1"] = nc.nc1
        metrics["nc/nc2_equinorm"] = nc.nc2_equinorm
        metrics["nc/nc2_equinorm_weights"] = nc.nc2_equinorm_weights
        metrics["nc/nc2_equiangularity"] = nc.nc2_equiangularity
        metrics["nc/nc2_equiangularity_weights"] = nc.nc2_equiangularity_weights
        metrics["nc/nc2_maximal_angle_equiangularity"] = nc.nc2_maximal_angle_equiangularity
        metrics["nc/nc2_maximal_angle_equiangularity_weights"] = (
            nc.nc2_maximal_angle_equiangularity_weights
        )
        metrics["nc/nc3"] = nc.nc3
        metrics["nc/nc4"] = nc.nc4

    # TODO: check
    # if train_activations.numel() > 0 and test_activations.numel() > 0 and compute_heavy_metrics and metrics_config.attack_distance_to_class_mean:
    #     from privacy_and_grokking.metrics.mia import distances_to_class_mean

    #     num_classes = int(train_labels.max().item() + 1)
    #     class_means = torch.zeros(num_classes, train_feats.shape[1], device=train_feats.device)
    #     for c in range(num_classes):
    #         mask = train_labels == c
    #         if mask.sum() > 0:
    #             class_means[c] = train_feats[mask].float().mean(dim=0)

    #     train_dists = distances_to_class_mean(train_feats, train_labels, class_means)
    #     test_dists = distances_to_class_mean(test_feats, test_labels, class_means)

    #     all_train_flat = torch.cat(list(train_dists.values())) if train_dists else torch.tensor([])
    #     all_test_flat = torch.cat(list(test_dists.values())) if test_dists else torch.tensor([])

    #     if len(all_train_flat) > 0 and len(all_test_flat) > 0:
    #         m = compute_roc_metrics_single_step(-all_train_flat, -all_test_flat)
    #         for key, value in m.items():
    #             metrics[f"attack/distance_to_class_mean/global/{key}"] = value

    #     for c in range(num_classes):
    #         if c in train_dists and c in test_dists:
    #             train_c = train_dists[c]
    #             test_c = test_dists[c]
    #             if len(train_c) > 0 and len(test_c) > 0:
    #                 m = compute_roc_metrics_single_step(-train_c, -test_c)
    #                 for key, value in m.items():
    #                     metrics[f"attack/distance_to_class_mean/class_{c}/{key}"] = value

    # if train_activations.numel() > 0 and test_activations.numel() > 0 and compute_heavy_metrics and metrics_config.attack_margin_distance_lf:
    #     from privacy_and_grokking.metrics.mia import margin_distance_lf

    #     num_classes = int(train_labels.max().item() + 1)
    #     pool_features = torch.cat([train_feats.float(), test_feats.float()])
    #     pool_mean_norm = pool_features.norm(dim=1).mean().item()

    #     train_dists_lf = margin_distance_lf(
    #         train_feats, train_labels, last_linear_weight, last_linear_bias, pool_mean_norm
    #     )
    #     test_dists_lf = margin_distance_lf(
    #         test_feats, test_labels, last_linear_weight, last_linear_bias, pool_mean_norm
    #     )

    #     all_train_flat_lf = torch.cat(list(train_dists_lf.values())) if train_dists_lf else torch.tensor([])
    #     all_test_flat_lf = torch.cat(list(test_dists_lf.values())) if test_dists_lf else torch.tensor([])

    #     if len(all_train_flat_lf) > 0 and len(all_test_flat_lf) > 0:
    #         m = compute_roc_metrics_single_step(-all_train_flat_lf, -all_test_flat_lf)
    #         for key, value in m.items():
    #             metrics[f"attack/margin_distance_lf/global/{key}"] = value

    #         train_median = all_train_flat_lf.median()
    #         m_centered = compute_roc_metrics_single_step(
    #             -torch.abs(all_train_flat_lf - train_median),
    #             -torch.abs(all_test_flat_lf - train_median)
    #         )
    #         for key, value in m_centered.items():
    #             metrics[f"attack/margin_distance_lf_centered/global/{key}"] = value

    #     for c in range(num_classes):
    #         if c in train_dists_lf and c in test_dists_lf:
    #             train_c_lf = train_dists_lf[c]
    #             test_c_lf = test_dists_lf[c]
    #             if len(train_c_lf) > 0 and len(test_c_lf) > 0:
    #                 m = compute_roc_metrics_single_step(-train_c_lf, -test_c_lf)
    #                 for key, value in m.items():
    #                     metrics[f"attack/margin_distance_lf/class_{c}/{key}"] = value

    #                 train_median_c = train_c_lf.median()
    #                 m_centered_c = compute_roc_metrics_single_step(
    #                     -torch.abs(train_c_lf - train_median_c),
    #                     -torch.abs(test_c_lf - train_median_c)
    #                 )
    #                 for key, value in m_centered_c.items():
    #                     metrics[f"attack/margin_distance_lf_centered/class_{c}/{key}"] = value

    keys = list(metrics.keys())
    for key in keys:
        metrics[f"{key_prefix}/{key}"] = metrics[key]
        del metrics[key]

    optimizer.zero_grad(set_to_none=True)
    mlflow.log_metrics(metrics, step=step)
    return metrics
