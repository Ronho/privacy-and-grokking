"""Visualisierung der Logit-, Loss- und InfoRMIA-Verteilungen sowie AUC-Vergleich
für NO_GROK_MADD_CE_TRANSFORMER und NO_GROK_MADD_MSE_TRANSFORMER.

Verbesserungen:
- Cross-Entropy Loss wird logarithmisch dargestellt: log10(CE-Loss), sodass Verteilungen
  über 7 Größenordnungen (10^-7 bis 10^1) als vollwertige Glockenkurven sichtbar werden.
- InfoRMIA Score für reguläre Daten wird auf den robusten Hauptbereich fokussiert
  (mit Ausreißer-Annotation für die wenigen Fehlklassifikationen).
- InfoRMIA Score für Canaries verwendet den regulären Test-Split als Populations-Erwartung
  und bei extremen Clustern (CE Canaries) eine geteilte Achse (Broken Axis).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path
from typing import Literal

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / "scratch" / ".matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
from sklearn.metrics import auc, roc_curve
from torch.utils.data import DataLoader, Dataset

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from privacy_and_grokking.config import TrainConfig
from privacy_and_grokking.utils.logger import Logger

DEFAULT_TRACKING_URI = "http://localhost:5051"
DEFAULT_PARQUET = PROJECT_ROOT / "cache" / "canary-selection-v1_mlflow_export.parquet"
DEFAULT_STEP = 20000

# 6 Runs für NO_GROK_MADD_CE_TRANSFORMER (Experiment: canary-selection-1, Seed: 696484)
CE_RUNS_MADD = [
    "0a341d15fecf4c3fa471fb6a35fb5e26",  # model_index: 0 (Target)
    "d0c6dd6ca6694ebfb978fa6575f77163",  # model_index: 1 (Validation)
    "175346abb6054cdcbcd9c7b1c8fa7bb8",  # model_index: 2 (Ref 0)
    "4166ce0921d142b49bb629eef2e4a493",  # model_index: 3 (Ref 1)
    "4071bcece5604426bedb1521bcfb40ad",  # model_index: 4 (Ref 2)
    "3978800e6aa542068b75fb8f599b96d5",  # model_index: 5 (Ref 3)
]

# 6 Runs für NO_GROK_MADD_MSE_TRANSFORMER (Experiment: canary-selection-1, Seed: 725792)
MSE_RUNS_MADD = [
    "94558a882eca40b9b4a15972b2ee6f20",  # model_index: 0 (Target)
    "bb77cb85eb3742a5843a7f6d445dc71f",  # model_index: 1 (Validation)
    "f6be708ddce6408892f15adc948826a7",  # model_index: 2 (Ref 0)
    "41d3bfe6a9f44aac8f733be73e761e0e",  # model_index: 3 (Ref 1)
    "bbbda71e139b4a51ba3496c942167e74",  # model_index: 4 (Ref 2)
    "f9001c82258f42bbac54f7727ca78d5d",  # model_index: 5 (Ref 3)
]

# 6 Runs für GROK_MNIST_CE_MLP (canary-selection-v1)
CE_RUNS_MNIST = [
    "424b983a0b1949ef860708d7c81bd58f",  # Target
    "24eda2cbcd0b47e2a417f07ab1c852ba",  # Validation
    "35eeb04439984209be3289cf1717f5b9",  # Ref 0
    "3d1e5759deef4af99248ebc6c581cc5a",  # Ref 1
    "a451633d20bd4523b66fba5cf6ecdfa2",  # Ref 2
    "0317a20721584e08afe789a612e2db41",  # Ref 3
]

# 6 Runs für GROK_MNIST_MSE_MLP (canary-selection-v1)
MSE_RUNS_MNIST = [
    "8db2fdccab07431b88271c095c8a3340",  # Target
    "1c1f36ceade64664a2a13a1c968481ed",  # Validation
    "4e4e974897a3432e92b547a2535174a9",  # Ref 0
    "5ef3b7e613a44a97b5820436eed0c7cb",  # Ref 1
    "78dfb6381fef43f1bb66946ee467961a",  # Ref 2
    "f32c7e6585834823937b801942d05e1d",  # Ref 3
]

CE_RUNS = CE_RUNS_MADD
MSE_RUNS = MSE_RUNS_MADD

def download_artifact_bytes(
    run_id: str,
    artifact_rel_path: str,
    tracking_uri: str = DEFAULT_TRACKING_URI,
) -> bytes:
    """Lädt ein Artefakt vom MLflow-Server herunter und verwendet einen lokalen Cache."""
    cache_path = PROJECT_ROOT / "cache" / "downloaded_artifacts" / run_id / artifact_rel_path
    if cache_path.is_file():
        return cache_path.read_bytes()

    base_uri = tracking_uri.rstrip("/")
    if not base_uri.startswith("http://") and not base_uri.startswith("https://"):
        base_uri = f"http://{base_uri}"

    url = f"{base_uri}/get-artifact?path={artifact_rel_path}&run_uuid={run_id}"
    resp = requests.get(url, stream=True, timeout=60)
    if resp.status_code != 200:
        raise FileNotFoundError(
            f"Konnte Artefakt '{artifact_rel_path}' für Run '{run_id}' nicht von {url} laden (Status: {resp.status_code})."
        )
    content = resp.content

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(content)
    except Exception:
        pass

    return content


def load_model_and_data(
    run_id: str,
    step: int,
    device: str = "cpu",
    tracking_uri: str = DEFAULT_TRACKING_URI,
):
    """Lädt Konfiguration, Datensatz und Modell-Checkpoint."""
    cfg_bytes = download_artifact_bytes(run_id, "training_config.json", tracking_uri)
    cfg_dict = json.loads(cfg_bytes.decode("utf-8"))

    train_cfg = TrainConfig.model_validate(cfg_dict)
    data_container = train_cfg.data()

    ckpt_path = f"checkpoints/{step}/model.pth"
    ckpt_bytes = download_artifact_bytes(run_id, ckpt_path, tracking_uri)

    model = train_cfg.model(
        input_dim=data_container.input_shape,
        num_classes=data_container.num_classes,
    )
    buffer = io.BytesIO(ckpt_bytes)
    sd = torch.load(buffer, map_location=device, weights_only=True)
    model.load_state_dict(sd)
    model.to(device)
    model.eval()

    return model, data_container, train_cfg


@torch.no_grad()
def evaluate_model_signals(
    model: torch.nn.Module,
    dataset: Dataset,
    device: str = "cpu",
    norm_mean: list[float] | None = None,
    norm_std: list[float] | None = None,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Berechnet für einen Datensatz:
    - correct_logits: z_y
    - wrong_logits: alle z_j (j != y)
    - ce_losses: -log(p_y)
    - mse_losses: (1/C) sum (z_c - y_onehot_c)^2
    - true_probs: p_y
    """
    if len(dataset) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    mean_t = (
        torch.tensor(norm_mean, device=device).view(1, -1, 1, 1)
        if norm_mean
        else None
    )
    std_t = (
        torch.tensor(norm_std, device=device).view(1, -1, 1, 1) if norm_std else None
    )

    corr_list = []
    wrong_list = []
    ce_list = []
    mse_list = []
    prob_list = []

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        if mean_t is not None and std_t is not None:
            batch_x = (batch_x - mean_t) / std_t

        logits = model(batch_x, verbose=False)
        corr_val = logits.gather(1, batch_y.view(-1, 1)).squeeze(1)

        mask_wrong = torch.ones_like(logits, dtype=torch.bool)
        mask_wrong.scatter_(1, batch_y.view(-1, 1), False)
        wrong_val = logits[mask_wrong]

        probs = F.softmax(logits, dim=1)
        true_p = probs.gather(1, batch_y.view(-1, 1)).squeeze(1)
        ce = -torch.log(true_p.clamp(min=1e-12))

        num_classes = logits.size(1)
        one_hot = F.one_hot(batch_y, num_classes=num_classes).float()
        mse = ((logits - one_hot) ** 2).mean(dim=-1)

        corr_list.append(corr_val.cpu().numpy())
        wrong_list.append(wrong_val.cpu().numpy())
        ce_list.append(ce.cpu().numpy())
        mse_list.append(mse.cpu().numpy())
        prob_list.append(true_p.cpu().numpy())

    return (
        np.concatenate(corr_list),
        np.concatenate(wrong_list),
        np.concatenate(ce_list),
        np.concatenate(mse_list),
        np.concatenate(prob_list),
    )


@torch.no_grad()
def evaluate_probs_only(
    model: torch.nn.Module,
    dataset: Dataset,
    device: str = "cpu",
    norm_mean: list[float] | None = None,
    norm_std: list[float] | None = None,
    batch_size: int = 256,
) -> np.ndarray:
    """Berechnet nur die True-Class-Wahrscheinlichkeiten p_y."""
    if len(dataset) == 0:
        return np.array([])

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    mean_t = (
        torch.tensor(norm_mean, device=device).view(1, -1, 1, 1)
        if norm_mean
        else None
    )
    std_t = (
        torch.tensor(norm_std, device=device).view(1, -1, 1, 1) if norm_std else None
    )

    prob_list = []
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        if mean_t is not None and std_t is not None:
            batch_x = (batch_x - mean_t) / std_t
        logits = model(batch_x, verbose=False)
        probs = F.softmax(logits, dim=1)
        true_p = probs.gather(1, batch_y.view(-1, 1)).squeeze(1)
        prob_list.append(true_p.cpu().numpy())

    return np.concatenate(prob_list) if prob_list else np.array([])


def compute_informia_scores(
    target_probs: np.ndarray,
    ref_probs_matrix: np.ndarray,
    ref_memberships: np.ndarray,
    pop_target_probs: np.ndarray,
    pop_ref_probs: np.ndarray,
    offline_a: float = 0.5,
) -> np.ndarray:
    """Berechnet InfoRMIA Scores gemäß der Definition in info_rmia.py."""
    if len(target_probs) == 0:
        return np.array([])

    # 1. P_out: Durchschnitt über Referenzmodelle, für die das Sample OUT war
    non_members = (ref_memberships == 0).astype(float)
    counts = np.clip(non_members.sum(axis=0), a_min=1.0, a_max=None)
    p_out = (ref_probs_matrix * non_members).sum(axis=0) / counts

    # 2. Offline-Abschätzung von P(x)
    mean_x = ((1.0 + offline_a) / 2.0) * p_out + ((1.0 - offline_a) / 2.0)
    mean_x = np.clip(mean_x, a_min=1e-12, a_max=None)
    log_ratio_x = np.log(np.clip(target_probs / mean_x, a_min=1e-12, a_max=None))

    # 3. Erwartungswert über Population (reguläres Test-Set)
    if len(pop_target_probs) > 0 and len(pop_ref_probs) > 0:
        pop_p_out = pop_ref_probs.mean(axis=0)
        pop_mean_z = ((1.0 + offline_a) / 2.0) * pop_p_out + ((1.0 - offline_a) / 2.0)
        pop_mean_z = np.clip(pop_mean_z, a_min=1e-12, a_max=None)
        prob_ratio_z = np.clip(pop_target_probs / pop_mean_z, a_min=1e-12, a_max=None)
        expectation = np.sum(pop_mean_z * np.log(prob_ratio_z)) / np.sum(pop_mean_z)
    else:
        expectation = 0.0

    # 4. Test-Statistik (InfoRMIA Score)
    scores = log_ratio_x - expectation
    return scores


def tune_optimal_a(
    val_p_in: np.ndarray,
    val_p_out: np.ndarray,
    ref_p_val_in: np.ndarray,
    ref_p_val_out: np.ndarray,
    mem_val_in: np.ndarray,
    mem_val_out: np.ndarray,
    val_pop_target_p: np.ndarray,
    val_pop_ref_p: np.ndarray,
    split_name: str = "regular",
) -> tuple[float, float, dict[float, float]]:
    """Tuned offline_a auf dem Validation-Modell (model_index 1) in 0.1 Schritten (0.0 bis 1.0).
    
    Wählt dasjenige a, welches die ROC-AUC auf dem Validierungsdatensatz maximiert.
    Gibt (optimal_a, bester_val_auc, sweep_dict) zurück.
    """
    best_a = 0.5
    best_auc = -1.0
    sweep_results = {}

    for a in np.arange(0.0, 1.05, 0.1):
        a_val = round(float(a), 1)
        in_scores = compute_informia_scores(
            target_probs=val_p_in,
            ref_probs_matrix=ref_p_val_in,
            ref_memberships=mem_val_in,
            pop_target_probs=val_pop_target_p,
            pop_ref_probs=val_pop_ref_p,
            offline_a=a_val,
        )
        out_scores = compute_informia_scores(
            target_probs=val_p_out,
            ref_probs_matrix=ref_p_val_out,
            ref_memberships=mem_val_out,
            pop_target_probs=val_pop_target_p,
            pop_ref_probs=val_pop_ref_p,
            offline_a=a_val,
        )
        auc_val, _, _ = compute_roc_and_auc(in_scores, out_scores)
        sweep_results[a_val] = auc_val
        if auc_val > best_auc:
            best_auc = auc_val
            best_a = a_val

    return best_a, best_auc, sweep_results


def plot_loss_overlay(
    ax: plt.Axes,
    in_vals: np.ndarray,
    out_vals: np.ndarray,
    in_name: str,
    out_name: str,
    loss_type: Literal["ce", "mse", "mse_tc"],
    color_in: str,
    color_out: str,
    num_bins: int = 35,
    auc: float | None = None,
):
    """Plottet Realen, Anderen oder Klassen-MSE Loss mit Dichte-Histogrammen und AUC-Badge."""
    if len(in_vals) == 0 and len(out_vals) == 0:
        ax.text(0.5, 0.5, "Keine Daten", ha="center", va="center", color="gray")
        return

    if loss_type == "ce":
        # Log10-Transformation für Cross-Entropy (überwindet 7 Größenordnungen)
        in_p = np.log10(np.clip(in_vals, 1e-12, None))
        out_p = np.log10(np.clip(out_vals, 1e-12, None))

        val_min = min(in_p.min(), out_p.min()) - 0.2
        val_max = max(in_p.max(), out_p.max()) + 0.2
        bins = np.linspace(val_min, val_max, num_bins + 1)

        ax.hist(
            in_p,
            bins=bins,
            density=True,
            alpha=0.55,
            color=color_in,
            edgecolor=color_in,
            linewidth=0.8,
            label=f"{in_name} (N={len(in_vals)}, $\\mu_{{\\log10}}={np.mean(in_p):.2f}$)",
        )
        ax.hist(
            out_p,
            bins=bins,
            density=True,
            alpha=0.45,
            color=color_out,
            edgecolor=color_out,
            linestyle="--",
            linewidth=0.8,
            label=f"{out_name} (N={len(out_vals)}, $\\mu_{{\\log10}}={np.mean(out_p):.2f}$)",
        )
        ax.set_xlabel("CE-Loss: $\\log_{10}(\\text{CE})$", fontsize=10)
    elif loss_type == "mse_tc":
        # Klassen-MSE (nur wahre Klasse): (z_y - 1)^2
        val_min = min(in_vals.min(), out_vals.min())
        val_max = max(in_vals.max(), out_vals.max())
        if np.isclose(val_min, val_max):
            val_max = val_min + 1.0
        bins = np.linspace(val_min, val_max, num_bins + 1)

        ax.hist(
            in_vals,
            bins=bins,
            density=True,
            alpha=0.55,
            color=color_in,
            edgecolor=color_in,
            linewidth=0.8,
            label=f"{in_name} (N={len(in_vals)}, $\\mu={np.mean(in_vals):.3g}$)",
        )
        ax.hist(
            out_vals,
            bins=bins,
            density=True,
            alpha=0.45,
            color=color_out,
            edgecolor=color_out,
            linestyle="--",
            linewidth=0.8,
            label=f"{out_name} (N={len(out_vals)}, $\\mu={np.mean(out_vals):.3g}$)",
        )
        ax.set_xlabel("Klassen-MSE: $(z_y - 1)^2$", fontsize=10)
    else:
        # Standard lineare Skalierung für One-Hot MSE über alle Klassen
        val_min = min(in_vals.min(), out_vals.min())
        val_max = max(in_vals.max(), out_vals.max())
        if np.isclose(val_min, val_max):
            val_max = val_min + 1.0
        bins = np.linspace(val_min, val_max, num_bins + 1)

        ax.hist(
            in_vals,
            bins=bins,
            density=True,
            alpha=0.55,
            color=color_in,
            edgecolor=color_in,
            linewidth=0.8,
            label=f"{in_name} (N={len(in_vals)}, $\\mu={np.mean(in_vals):.3g}$)",
        )
        ax.hist(
            out_vals,
            bins=bins,
            density=True,
            alpha=0.45,
            color=color_out,
            edgecolor=color_out,
            linestyle="--",
            linewidth=0.8,
            label=f"{out_name} (N={len(out_vals)}, $\\mu={np.mean(out_vals):.3g}$)",
        )
        ax.set_xlabel("MSE-Loss (One-Hot): $\\frac{1}{C}\\sum (z_c - y_c)^2$", fontsize=10)

    if auc is not None:
        ax.text(
            0.04,
            0.92,
            f"Attack AUC: {auc:.3f}",
            transform=ax.transAxes,
            fontsize=9.5,
            fontweight="bold",
            color="#0f172a",
            bbox=dict(boxstyle="round,pad=0.28", facecolor="#ffffff", edgecolor="#94a3b8", alpha=0.92, linewidth=1.0),
            zorder=10,
        )

    ax.set_ylabel("Dichte", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=8, loc="upper right")


def plot_informia_overlay(
    ax: plt.Axes,
    in_vals: np.ndarray,
    out_vals: np.ndarray,
    in_name: str,
    out_name: str,
    color_in: str = "#ea580c",
    color_out: str = "#06b6d4",
    num_bins: int = 35,
    auc: float | None = None,
):
    """Plottet InfoRMIA-Scores mit robustem Zoom bzw. Broken-Axis bei extremen Abständen."""
    if len(in_vals) == 0 and len(out_vals) == 0:
        ax.text(0.5, 0.5, "Keine Daten", ha="center", va="center", color="gray")
        return

    val_min = min(in_vals.min(), out_vals.min())
    val_max = max(in_vals.max(), out_vals.max())

    # Fall 1: Extrem getrennte Cluster (z. B. CE Canary bei Schritt 20000: OUT bei -27, IN bei +1.4)
    if in_vals.min() > -5.0 and out_vals.max() < -8.0:
        ax.axis("off")

        ax_l = ax.inset_axes([0.0, 0.08, 0.46, 0.88])
        ax_r = ax.inset_axes([0.54, 0.08, 0.46, 0.88])

        b_out = np.linspace(out_vals.min() - 0.3, out_vals.max() + 0.3, 16)
        b_in = np.linspace(in_vals.min() - 0.05, in_vals.max() + 0.05, 16)

        ax_l.hist(
            out_vals,
            bins=b_out,
            density=True,
            alpha=0.55,
            color=color_out,
            edgecolor=color_out,
            label=f"{out_name} ($\\mu={np.mean(out_vals):.1f}$)",
        )
        ax_r.hist(
            in_vals,
            bins=b_in,
            density=True,
            alpha=0.6,
            color=color_in,
            edgecolor=color_in,
            label=f"{in_name} ($\\mu={np.mean(in_vals):.2f}$)",
        )

        ax_l.set_ylabel("Dichte", fontsize=10)
        ax_l.set_xlabel("InfoRMIA", fontsize=9)
        ax_r.set_xlabel("InfoRMIA", fontsize=9)
        ax_l.grid(True, linestyle=":", alpha=0.6)
        ax_r.grid(True, linestyle=":", alpha=0.6)
        ax_l.legend(fontsize=7.5, loc="upper right")
        ax_r.legend(fontsize=7.5, loc="upper right")

        ax_l.spines["right"].set_visible(False)
        ax_r.spines["left"].set_visible(False)
        ax_r.yaxis.set_ticks([])

        if auc is not None:
            ax_l.text(
                0.06,
                0.90,
                f"Attack AUC: {auc:.3f}",
                transform=ax_l.transAxes,
                fontsize=9.0,
                fontweight="bold",
                color="#0f172a",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#ffffff", edgecolor="#94a3b8", alpha=0.92, linewidth=1.0),
                zorder=10,
            )

        d = 0.03
        kwargs = dict(transform=ax_l.transAxes, color="k", clip_on=False, linewidth=1.2)
        ax_l.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        ax_l.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

        kwargs.update(transform=ax_r.transAxes)
        ax_r.plot((-d, +d), (-d, +d), **kwargs)
        ax_r.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        return

    # Fall 2: Kontinuierliche Verteilung
    data_max = max(float(in_vals.max()), float(out_vals.max()))
    if data_max > -2.0:
        # Hauptmasse um 0 mit eventuellen extremen negativen Ausreißern
        p_low = min(float(np.percentile(in_vals, 0.1)), float(np.percentile(out_vals, 0.5)))
        robust_min = max(-0.6, p_low)
        robust_max = data_max + 0.05
        if robust_max <= robust_min:
            robust_max = robust_min + 1.0
        n_outliers = int(np.sum(out_vals < robust_min) + np.sum(in_vals < robust_min))
        bins = np.linspace(robust_min, robust_max, num_bins + 1)

        in_plot = in_vals[in_vals >= robust_min]
        out_plot = out_vals[out_vals >= robust_min]
    else:
        # Beide Gruppen liegen im stark negativen Bereich (z. B. unmemorisierte Canaries bei frühen Schritten)
        robust_min = min(float(in_vals.min()), float(out_vals.min())) - 0.2
        robust_max = data_max + 0.2
        if robust_max <= robust_min:
            robust_max = robust_min + 1.0
        n_outliers = 0
        bins = np.linspace(robust_min, robust_max, num_bins + 1)
        in_plot = in_vals
        out_plot = out_vals

    ax.hist(
        in_plot,
        bins=bins,
        density=True,
        alpha=0.55,
        color=color_in,
        edgecolor=color_in,
        linewidth=0.8,
        label=f"{in_name} (N={len(in_vals)}, $\\mu={np.mean(in_vals):.3g}$)",
    )
    ax.hist(
        out_plot,
        bins=bins,
        density=True,
        alpha=0.45,
        color=color_out,
        edgecolor=color_out,
        linestyle="--",
        linewidth=0.8,
        label=f"{out_name} (N={len(out_vals)}, $\\mu={np.mean(out_vals):.3g}$)",
    )

    if auc is not None:
        ax.text(
            0.04,
            0.92,
            f"Attack AUC: {auc:.3f}",
            transform=ax.transAxes,
            fontsize=9.5,
            fontweight="bold",
            color="#0f172a",
            bbox=dict(boxstyle="round,pad=0.28", facecolor="#ffffff", edgecolor="#94a3b8", alpha=0.92, linewidth=1.0),
            zorder=10,
        )

    if n_outliers > 0:
        y_pos = 0.81 if auc is not None else 0.88
        ax.text(
            0.04,
            y_pos,
            f"[{n_outliers} Ausreißer < {robust_min:.1f}]",
            transform=ax.transAxes,
            fontsize=8,
            fontweight="bold",
            color="#0891b2",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#ecfeff", edgecolor="#0891b2", alpha=0.8),
        )

    ax.set_xlim(robust_min - 0.05, robust_max + 0.05)
    ax.set_xlabel("InfoRMIA: $\\log(p_y / P(x)) - \\mathbb{E}$", fontsize=10)
    ax.set_ylabel("Dichte", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=8, loc="upper right")


def compute_tpr_at_fpr(fpr: np.ndarray, tpr: np.ndarray, target_fpr: float = 0.01) -> float:
    """Berechnet den TPR-Wert bei einer Ziel-FPR (Default: 0.01 = 1%)."""
    if len(fpr) == 0 or len(tpr) == 0:
        return 0.0
    idx = np.where(fpr <= target_fpr)[0]
    if len(idx) == 0:
        return float(tpr[0])
    return float(tpr[idx[-1]])


def load_parquet_metrics(parquet_path: Path, run_ids: list[str]) -> pd.DataFrame:
    """Lädt Metriken für die übergebenen run_ids aus dem Parquet-Cache."""
    if not parquet_path.is_file():
        return pd.DataFrame()
    columns = ["run_id", "run_name", "metric_name", "step", "value"]
    try:
        df = pd.read_parquet(parquet_path, columns=columns)
        df_filtered = df[df["run_id"].isin(run_ids)].copy()
        return df_filtered
    except Exception as e:
        print(f"  [WARNUNG] Fehler beim Laden der Parquet-Datei '{parquet_path}': {e}")
        return pd.DataFrame()


def plot_accuracy_and_overlap_panel(
    ax: plt.Axes,
    df_metrics: pd.DataFrame,
    target_run_id: str,
    all_run_ids: list[str],
    is_canary: bool,
    real_loss_type: Literal["ce", "mse"],
    selected_step: int,
):
    """Spalte 0: Zeichnet Genauigkeit (Train & Test) und Loss Overlap über die Zeit (Schritte)."""
    if df_metrics.empty:
        ax.text(0.5, 0.5, "Keine Parquet-Daten", ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.set_ylabel("Genauigkeit & Overlap", fontsize=9)
        ax.set_xlabel("Schritt ($t$)", fontsize=9)
        return

    acc_train_key = "eval/train/canary_accuracy" if is_canary else "eval/train/accuracy"
    acc_test_key = "eval/test/canary_accuracy" if is_canary else "eval/test/accuracy"

    if not is_canary:
        real_ovl_key = "eval/loss/ce/overlap" if real_loss_type == "ce" else "eval/loss/mse/overlap"
        other_ovl_key = "eval/loss/mse/overlap" if real_loss_type == "ce" else "eval/loss/ce/overlap"
    else:
        real_ovl_key = "eval/loss/canary_ce/overlap" if real_loss_type == "ce" else "eval/loss/canary_mse/overlap"
        other_ovl_key = "eval/loss/canary_mse/overlap" if real_loss_type == "ce" else "eval/loss/canary_ce/overlap"

    curve_configs = [
        (acc_train_key, "Train Acc", "#2563eb", "-", 2.0),
        (acc_test_key, "Test Acc", "#16a34a", "-", 2.0),
        (real_ovl_key, f"Realer OVL ({real_loss_type.upper()})", "#d97706", "--", 1.8),
        (other_ovl_key, f"Anderer OVL ({('mse' if real_loss_type == 'ce' else 'ce').upper()})", "#9333ea", ":", 1.8),
    ]

    for m_key, label_name, color, linestyle, lw in curve_configs:
        sub_df = df_metrics[df_metrics["metric_name"] == m_key]
        if sub_df.empty:
            continue

        target_sub = sub_df[sub_df["run_id"] == target_run_id].sort_values("step")
        if not target_sub.empty:
            t_clean = target_sub.drop_duplicates(subset=["step"])
            ax.plot(
                t_clean["step"],
                t_clean["value"],
                label=label_name,
                color=color,
                linestyle=linestyle,
                linewidth=lw,
                alpha=0.95,
                zorder=4,
            )
            val_at_step = t_clean[t_clean["step"] == selected_step]
            if not val_at_step.empty:
                val = float(val_at_step["value"].iloc[0])
                ax.scatter([selected_step], [val], color=color, s=35, zorder=6, edgecolors="#ffffff", linewidths=1.0)

    ax.axvline(
        x=selected_step,
        color="#475569",
        linestyle="--",
        linewidth=1.4,
        alpha=0.85,
        label=f"t={selected_step:,}",
        zorder=5,
    )

    ax.set_xlabel("Trainings-Schritt ($t$)", fontsize=9.5)
    ax.set_ylabel("Genauigkeit & Overlap", fontsize=9.5)
    ax.set_ylim(-0.02, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))
    ax.grid(True, linestyle=":", alpha=0.6)
    legend_loc = "lower right" if not is_canary else "center right"
    ax.legend(loc=legend_loc, fontsize=7.0, framealpha=0.88, labelspacing=0.25, handlelength=1.4, borderpad=0.35)


def plot_loss_over_time_panel(
    ax: plt.Axes,
    df_metrics: pd.DataFrame,
    target_run_id: str,
    all_run_ids: list[str],
    is_canary: bool,
    real_loss_type: Literal["ce", "mse"],
    selected_step: int,
):
    """Spalte 1: Zeichnet Train- und Test-Loss über die Zeit (Schritte) auf logarithmischer Skala."""
    if df_metrics.empty:
        ax.text(0.5, 0.5, "Keine Parquet-Daten", ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.set_ylabel("Loss", fontsize=9)
        ax.set_xlabel("Schritt ($t$)", fontsize=9)
        return

    loss_name = "CE" if real_loss_type == "ce" else "MSE"
    loss_field = "cross_entropy" if real_loss_type == "ce" else "mse"

    if not is_canary:
        train_key = f"eval/train/loss/{loss_field}/mean"
        test_key = f"eval/test/loss/{loss_field}/mean"
        label_train = f"Train ({loss_name})"
        label_test = f"Test ({loss_name})"
    else:
        train_key = f"eval/train/canary_loss/{loss_field}/mean"
        test_key = f"eval/test/canary_loss/{loss_field}/mean"
        label_train = f"Canary Train ({loss_name})"
        label_test = f"Canary Test ({loss_name})"

    curve_configs = [
        (train_key, label_train, "#2563eb", "-", 2.0),
        (test_key, label_test, "#16a34a", "--", 2.0),
    ]

    for m_key, label_name, color, linestyle, lw in curve_configs:
        sub_df = df_metrics[df_metrics["metric_name"] == m_key]
        if sub_df.empty:
            continue

        target_sub = sub_df[sub_df["run_id"] == target_run_id].sort_values("step")
        if not target_sub.empty:
            t_clean = target_sub.drop_duplicates(subset=["step"])
            t_valid = t_clean[t_clean["value"] > 0]
            if not t_valid.empty:
                ax.plot(
                    t_valid["step"],
                    t_valid["value"],
                    label=label_name,
                    color=color,
                    linestyle=linestyle,
                    linewidth=lw,
                    alpha=0.95,
                    zorder=4,
                )
                val_at_step = t_valid[t_valid["step"] == selected_step]
                if not val_at_step.empty:
                    val = float(val_at_step["value"].iloc[0])
                    ax.scatter([selected_step], [val], color=color, s=35, zorder=6, edgecolors="#ffffff", linewidths=1.0)

    ax.axvline(
        x=selected_step,
        color="#475569",
        linestyle="--",
        linewidth=1.4,
        alpha=0.85,
        label=f"t={selected_step:,}",
        zorder=5,
    )

    ax.set_yscale("log")
    ax.set_xlabel("Trainings-Schritt ($t$)", fontsize=9.5)
    ax.set_ylabel(f"Realer Loss ({loss_name}, log)", fontsize=9.5)
    ax.grid(True, linestyle=":", alpha=0.6, which="both")
    legend_loc = "upper right"
    ax.legend(loc=legend_loc, fontsize=7.0, framealpha=0.88, labelspacing=0.25, handlelength=1.4, borderpad=0.35)


def plot_norms_over_time_panel(
    ax: plt.Axes,
    df_metrics: pd.DataFrame,
    target_run_id: str,
    all_run_ids: list[str],
    is_canary: bool,
    real_loss_type: Literal["ce", "mse"],
    selected_step: int,
):
    """Spalte 2: Zeichnet Gradientennorm (log-scale) und Gewichtsnorm ||W|| über die Zeit mit Doppel-Y-Achse."""
    if df_metrics.empty:
        ax.text(0.5, 0.5, "Keine Parquet-Daten", ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.set_ylabel("Norm", fontsize=9)
        ax.set_xlabel("Schritt ($t$)", fontsize=9)
        return

    sub_g = df_metrics[(df_metrics["run_id"] == target_run_id) & (df_metrics["metric_name"] == "eval/grad_norm/total")].drop_duplicates("step").sort_values("step")
    sub_w = df_metrics[(df_metrics["run_id"] == target_run_id) & (df_metrics["metric_name"] == "eval/weight_norm/total")].drop_duplicates("step").sort_values("step")

    ax_w = ax.twinx()
    l1, l2 = None, None

    if not sub_g.empty:
        sub_g_valid = sub_g[sub_g["value"] > 0]
        if not sub_g_valid.empty:
            l1 = ax.plot(
                sub_g_valid["step"],
                sub_g_valid["value"],
                color="#dc2626",
                linestyle="-",
                linewidth=2.0,
                alpha=0.95,
                label="Grad-Norm (log)",
                zorder=4,
            )
            val_at_s = sub_g_valid[sub_g_valid["step"] == selected_step]
            if not val_at_s.empty:
                val = float(val_at_s["value"].iloc[0])
                ax.scatter([selected_step], [val], color="#dc2626", s=35, zorder=6, edgecolors="#ffffff", linewidths=1.0)

    if not sub_w.empty:
        l2 = ax_w.plot(
            sub_w["step"],
            sub_w["value"],
            color="#7c3aed",
            linestyle="--",
            linewidth=2.0,
            alpha=0.95,
            label="Gewichts-Norm ||W||",
            zorder=4,
        )
        val_w_at_s = sub_w[sub_w["step"] == selected_step]
        if not val_w_at_s.empty:
            val_w = float(val_w_at_s["value"].iloc[0])
            ax_w.scatter([selected_step], [val_w], color="#7c3aed", s=35, zorder=6, edgecolors="#ffffff", linewidths=1.0)

    ax.axvline(
        x=selected_step,
        color="#475569",
        linestyle="--",
        linewidth=1.4,
        alpha=0.85,
        label=f"t={selected_step:,}",
        zorder=5,
    )

    ax.set_yscale("log")
    ax.set_xlabel("Trainings-Schritt ($t$)", fontsize=9.5)
    ax.set_ylabel("Grad-Norm (total, log)", color="#dc2626", fontsize=9.5)
    ax.tick_params(axis="y", labelcolor="#dc2626")
    ax_w.set_ylabel("Gewichts-Norm ||W||", color="#7c3aed", fontsize=9.5)
    ax_w.tick_params(axis="y", labelcolor="#7c3aed")
    ax.grid(True, linestyle=":", alpha=0.6, which="both")

    # Einheitliche Legende
    lines = []
    if l1:
        lines.extend(l1)
    if l2:
        lines.extend(l2)
    lines.append(plt.Line2D([0], [0], color="#475569", linestyle="--", linewidth=1.4, label=f"t={selected_step:,}"))
    labels = [line.get_label() for line in lines]
    legend_loc = "upper right"
    ax.legend(lines, labels, loc=legend_loc, fontsize=6.8, framealpha=0.88, labelspacing=0.25, handlelength=1.4, borderpad=0.35)


def plot_attack_auc_panel(
    ax: plt.Axes,
    df_metrics: pd.DataFrame,
    target_run_id: str,
    all_run_ids: list[str],
    is_canary: bool,
    real_loss_type: Literal["ce", "mse"],
    selected_step: int,
    informia_auc: float | None = None,
):
    """Spalte 1: Zeichnet Attack AUC über die Zeit (Schritte) mit Referenzlinie bei 0.5 und InfoRMIA am Schritt t."""
    if df_metrics.empty:
        ax.text(0.5, 0.5, "Keine Parquet-Daten", ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.set_ylabel("Attack AUC", fontsize=9)
        ax.set_xlabel("Schritt ($t$)", fontsize=9)
        return

    if not is_canary:
        real_auc_key = "eval/attack/ce_loss/auc" if real_loss_type == "ce" else "eval/attack/mse_loss/auc"
        other_auc_key = "eval/attack/mse_loss/auc" if real_loss_type == "ce" else "eval/attack/ce_loss/auc"
    else:
        real_auc_key = "eval/attack/canary_ce_loss/auc" if real_loss_type == "ce" else "eval/attack/canary_mse_loss/auc"
        other_auc_key = "eval/attack/canary_mse_loss/auc" if real_loss_type == "ce" else "eval/attack/canary_ce_loss/auc"

    curve_configs = [
        (real_auc_key, f"Realer Loss ({real_loss_type.upper()})", "#059669", "-", 2.0),
        (other_auc_key, f"Anderer Loss ({('mse' if real_loss_type == 'ce' else 'ce').upper()})", "#7c3aed", "-", 2.0),
    ]

    for m_key, label_name, color, linestyle, lw in curve_configs:
        sub_df = df_metrics[df_metrics["metric_name"] == m_key]
        if sub_df.empty:
            continue

        target_sub = sub_df[sub_df["run_id"] == target_run_id].sort_values("step")
        if not target_sub.empty:
            t_clean = target_sub.drop_duplicates(subset=["step"])
            ax.plot(
                t_clean["step"],
                t_clean["value"],
                label=label_name,
                color=color,
                linestyle=linestyle,
                linewidth=lw,
                alpha=0.95,
                zorder=4,
            )
            val_at_step = t_clean[t_clean["step"] == selected_step]
            if not val_at_step.empty:
                val = float(val_at_step["value"].iloc[0])
                ax.scatter([selected_step], [val], color=color, s=35, zorder=6, edgecolors="#ffffff", linewidths=1.0)

    if informia_auc is not None:
        ax.scatter(
            [selected_step],
            [informia_auc],
            color="#ea580c",
            s=95,
            marker="*",
            zorder=8,
            label=f"InfoRMIA ($t$): {informia_auc:.3f}",
            edgecolors="#ffffff",
            linewidths=1.2,
        )

    ax.axhline(y=0.5, color="#94a3b8", linestyle=":", linewidth=1.2, label="Zufall (0.50)", zorder=3)
    ax.axvline(
        x=selected_step,
        color="#475569",
        linestyle="--",
        linewidth=1.4,
        alpha=0.85,
        zorder=5,
    )

    ax.set_xlabel("Trainings-Schritt ($t$)", fontsize=9.5)
    ax.set_ylabel("Attack ROC AUC", fontsize=9.5)
    ax.set_ylim(0.40, 1.03)
    ax.grid(True, linestyle=":", alpha=0.6)
    legend_loc = "upper right" if not is_canary else "center right"
    ax.legend(loc=legend_loc, fontsize=7.0, framealpha=0.88, labelspacing=0.25, handlelength=1.4, borderpad=0.35)


def plot_attack_tpr_panel(
    ax: plt.Axes,
    df_metrics: pd.DataFrame,
    target_run_id: str,
    all_run_ids: list[str],
    is_canary: bool,
    real_loss_type: Literal["ce", "mse"],
    selected_step: int,
    informia_tpr1: float | None = None,
):
    """Spalte 2: Zeichnet Attack TPR @ 1% FPR über die Zeit (Schritte) mit Referenzlinie bei 0.01 und InfoRMIA am Schritt t."""
    if df_metrics.empty:
        ax.text(0.5, 0.5, "Keine Parquet-Daten", ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.set_ylabel("TPR @ 1% FPR", fontsize=9)
        ax.set_xlabel("Schritt ($t$)", fontsize=9)
        return

    if not is_canary:
        real_tpr_key = "eval/attack/ce_loss/tpr-at-fpr/1" if real_loss_type == "ce" else "eval/attack/mse_loss/tpr-at-fpr/1"
        other_tpr_key = "eval/attack/mse_loss/tpr-at-fpr/1" if real_loss_type == "ce" else "eval/attack/ce_loss/tpr-at-fpr/1"
    else:
        real_tpr_key = "eval/attack/canary_ce_loss/tpr-at-fpr/1" if real_loss_type == "ce" else "eval/attack/canary_mse_loss/tpr-at-fpr/1"
        other_tpr_key = "eval/attack/canary_mse_loss/tpr-at-fpr/1" if real_loss_type == "ce" else "eval/attack/canary_ce_loss/tpr-at-fpr/1"

    curve_configs = [
        (real_tpr_key, f"Realer Loss ({real_loss_type.upper()})", "#059669", "-", 2.0),
        (other_tpr_key, f"Anderer Loss ({('mse' if real_loss_type == 'ce' else 'ce').upper()})", "#7c3aed", "-", 2.0),
    ]

    for m_key, label_name, color, linestyle, lw in curve_configs:
        sub_df = df_metrics[df_metrics["metric_name"] == m_key]
        if sub_df.empty:
            continue

        target_sub = sub_df[sub_df["run_id"] == target_run_id].sort_values("step")
        if not target_sub.empty:
            t_clean = target_sub.drop_duplicates(subset=["step"])
            ax.plot(
                t_clean["step"],
                t_clean["value"],
                label=label_name,
                color=color,
                linestyle=linestyle,
                linewidth=lw,
                alpha=0.95,
                zorder=4,
            )
            val_at_step = t_clean[t_clean["step"] == selected_step]
            if not val_at_step.empty:
                val = float(val_at_step["value"].iloc[0])
                ax.scatter([selected_step], [val], color=color, s=35, zorder=6, edgecolors="#ffffff", linewidths=1.0)

    if informia_tpr1 is not None:
        ax.scatter(
            [selected_step],
            [informia_tpr1],
            color="#ea580c",
            s=95,
            marker="*",
            zorder=8,
            label=f"InfoRMIA ($t$): {informia_tpr1:.3f}",
            edgecolors="#ffffff",
            linewidths=1.2,
        )

    ax.axhline(y=0.01, color="#94a3b8", linestyle=":", linewidth=1.2, label="Zufall (0.01)", zorder=3)
    ax.axvline(
        x=selected_step,
        color="#475569",
        linestyle="--",
        linewidth=1.4,
        alpha=0.85,
        zorder=5,
    )

    ax.set_xlabel("Trainings-Schritt ($t$)", fontsize=9.5)
    ax.set_ylabel("TPR @ 1% FPR", fontsize=9.5)
    ax.set_ylim(-0.02, 1.03)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))
    ax.grid(True, linestyle=":", alpha=0.6)
    legend_loc = "upper right" if not is_canary else "center right"
    ax.legend(loc=legend_loc, fontsize=7.0, framealpha=0.88, labelspacing=0.25, handlelength=1.4, borderpad=0.35)


def plot_2row_11col_figure(
    target_model_name: str,
    logits_res: dict[str, tuple[np.ndarray, np.ndarray]],
    real_loss_label: str,
    real_loss_type: Literal["ce", "mse"],
    real_loss_res: dict[str, np.ndarray],
    other_loss_label: str,
    other_loss_type: Literal["ce", "mse"],
    other_loss_res: dict[str, np.ndarray],
    tc_mse_res: dict[str, np.ndarray],
    informia_res: dict[str, np.ndarray],
    attack_results: dict,
    df_metrics: pd.DataFrame | None,
    target_run_id: str,
    all_run_ids: list[str],
    step: int,
    output_path: Path,
):
    """Erstellt eine optimierte 2x11 Abbildung pro Modell mit Dynamikkurven über die Zeit (Loss, Normen) und Verteilungen."""
    fig, axes = plt.subplots(2, 11, figsize=(51, 10.5))
    fig.suptitle(
        f"Verteilungs- & Dynamikanalyse: {target_model_name} (Schritt t = {step:,})",
        fontsize=18,
        fontweight="bold",
        y=0.985,
    )

    row_configs = [
        ("Reguläre Daten", "regular", "train_reg", "test_reg", "Train", "Test", False),
        ("Canary Daten", "canary", "train_canary", "test_canary", "Canary Train", "Canary Test", True),
    ]

    real_loss_header = (
        f"Realer Loss: $\\log_{{10}}(\\text{{CE}})$"
        if real_loss_type == "ce"
        else f"Realer Loss: One-Hot MSE"
    )
    other_loss_header = (
        f"Anderer Loss: $\\log_{{10}}(\\text{{CE}})$"
        if other_loss_type == "ce"
        else f"Anderer Loss: One-Hot MSE"
    )

    col_headers = [
        "Genauigkeit & Loss-Overlap\nüber Zeit",
        "Train- & Test-Loss\nüber Zeit",
        "Gradienten- & Gewichts-Norm\nüber Zeit",
        "Attack AUC\nüber Zeit",
        "Attack TPR @ 1% FPR\nüber Zeit",
        "Logit-Verteilung: Richtig ($y$)\nTrain vs. Test",
        "Logit-Verteilung: Falsch ($j \\neq y$)\nTrain vs. Test",
        f"{real_loss_header}\nTrain vs. Test",
        f"{other_loss_header}\nTrain vs. Test",
        "Klassen-MSE: $(z_y - 1)^2$\nTrain vs. Test",
        "InfoRMIA Score\nTrain vs. Test",
    ]

    for c_idx, head in enumerate(col_headers):
        axes[0, c_idx].set_title(head, fontsize=11.5, fontweight="bold", pad=12)

    df_m = df_metrics if df_metrics is not None else pd.DataFrame()

    for row_idx, (row_label, split_key, in_key, out_key, in_name, out_name, is_canary) in enumerate(row_configs):
        split_att = attack_results.get(split_key, {})
        auc_real = split_att.get("real_loss", (None,))[0]
        auc_other = split_att.get("other_loss", (None,))[0]
        auc_tc_mse = split_att.get("tc_mse", (None,))[0]
        auc_logit = split_att.get("logit_corr", (None,))[0]
        auc_info = split_att.get("informia", (None, None, None))[0]
        fpr_info = split_att.get("informia", (None, None, None))[1]
        tpr_info = split_att.get("informia", (None, None, None))[2]
        tpr1_info = compute_tpr_at_fpr(fpr_info, tpr_info, 0.01) if fpr_info is not None and tpr_info is not None else None

        # Spalte 0: Genauigkeit & Loss-Overlap über Zeit
        plot_accuracy_and_overlap_panel(
            ax=axes[row_idx, 0],
            df_metrics=df_m,
            target_run_id=target_run_id,
            all_run_ids=all_run_ids,
            is_canary=is_canary,
            real_loss_type=real_loss_type,
            selected_step=step,
        )

        # Spalte 1: Train- und Test-Loss über Zeit (log-scale)
        plot_loss_over_time_panel(
            ax=axes[row_idx, 1],
            df_metrics=df_m,
            target_run_id=target_run_id,
            all_run_ids=all_run_ids,
            is_canary=is_canary,
            real_loss_type=real_loss_type,
            selected_step=step,
        )

        # Spalte 2: Gradienten- und Gewichtsnorm über Zeit (Doppel-Y-Achse)
        plot_norms_over_time_panel(
            ax=axes[row_idx, 2],
            df_metrics=df_m,
            target_run_id=target_run_id,
            all_run_ids=all_run_ids,
            is_canary=is_canary,
            real_loss_type=real_loss_type,
            selected_step=step,
        )

        # Spalte 3: Attack AUC über Zeit
        plot_attack_auc_panel(
            ax=axes[row_idx, 3],
            df_metrics=df_m,
            target_run_id=target_run_id,
            all_run_ids=all_run_ids,
            is_canary=is_canary,
            real_loss_type=real_loss_type,
            selected_step=step,
            informia_auc=auc_info,
        )

        # Spalte 4: Attack TPR @ 1% FPR über Zeit
        plot_attack_tpr_panel(
            ax=axes[row_idx, 4],
            df_metrics=df_m,
            target_run_id=target_run_id,
            all_run_ids=all_run_ids,
            is_canary=is_canary,
            real_loss_type=real_loss_type,
            selected_step=step,
            informia_tpr1=tpr1_info,
        )

        corr_in, wrong_in = logits_res.get(in_key, (np.array([]), np.array([])))
        corr_out, wrong_out = logits_res.get(out_key, (np.array([]), np.array([])))

        # Spalte 5: Logit-Verteilung Richtig (Train vs. Test)
        ax_l_corr = axes[row_idx, 5]
        if len(corr_in) > 0 and len(corr_out) > 0:
            c_min = min(float(np.min(corr_in)), float(np.min(corr_out)))
            c_max = max(float(np.max(corr_in)), float(np.max(corr_out)))
            if np.isclose(c_min, c_max):
                c_max = c_min + 1.0
            bins_c = np.linspace(c_min, c_max, 45)
            ax_l_corr.hist(
                corr_in,
                bins=bins_c,
                density=True,
                alpha=0.5,
                color="#2563eb",
                label=f"{in_name} (N={len(corr_in):,})",
                edgecolor="#1d4ed8",
                linewidth=0.8,
            )
            ax_l_corr.hist(
                corr_out,
                bins=bins_c,
                density=True,
                alpha=0.45,
                color="#f59e0b",
                label=f"{out_name} (N={len(corr_out):,})",
                edgecolor="#d97706",
                linewidth=0.8,
            )
        elif len(corr_in) > 0:
            ax_l_corr.hist(corr_in, bins=40, density=True, alpha=0.5, color="#2563eb", label=f"{in_name} (N={len(corr_in):,})")
        elif len(corr_out) > 0:
            ax_l_corr.hist(corr_out, bins=40, density=True, alpha=0.45, color="#f59e0b", label=f"{out_name} (N={len(corr_out):,})")

        if auc_logit is not None:
            ax_l_corr.text(
                0.04,
                0.92,
                f"Attack AUC: {auc_logit:.3f}",
                transform=ax_l_corr.transAxes,
                fontsize=9.5,
                fontweight="bold",
                color="#0f172a",
                bbox=dict(boxstyle="round,pad=0.28", facecolor="#ffffff", edgecolor="#94a3b8", alpha=0.92, linewidth=1.0),
                zorder=10,
            )

        ax_l_corr.set_xlabel("Logit-Wert ($z_y$)", fontsize=9.5)
        ax_l_corr.set_ylabel("Dichte", fontsize=9.5)
        ax_l_corr.grid(True, linestyle=":", alpha=0.6)
        if len(corr_in) > 0 or len(corr_out) > 0:
            ax_l_corr.legend(fontsize=7.2, loc="upper right")

        # Spalte 6: Logit-Verteilung Falsch (Train vs. Test)
        ax_l_wrong = axes[row_idx, 6]
        if len(wrong_in) > 0 and len(wrong_out) > 0:
            w_min = min(float(np.min(wrong_in)), float(np.min(wrong_out)))
            w_max = max(float(np.max(wrong_in)), float(np.max(wrong_out)))
            if np.isclose(w_min, w_max):
                w_max = w_min + 1.0
            bins_w = np.linspace(w_min, w_max, 50)
            ax_l_wrong.hist(
                wrong_in,
                bins=bins_w,
                density=True,
                alpha=0.5,
                color="#dc2626",
                label=f"{in_name} (N={len(wrong_in):,})",
                edgecolor="#b91c1c",
                linewidth=0.8,
            )
            ax_l_wrong.hist(
                wrong_out,
                bins=bins_w,
                density=True,
                alpha=0.45,
                color="#7c3aed",
                label=f"{out_name} (N={len(wrong_out):,})",
                edgecolor="#6d28d9",
                linewidth=0.8,
            )
        elif len(wrong_in) > 0:
            ax_l_wrong.hist(wrong_in, bins=50, density=True, alpha=0.5, color="#dc2626", label=f"{in_name} (N={len(wrong_in):,})")
        elif len(wrong_out) > 0:
            ax_l_wrong.hist(wrong_out, bins=50, density=True, alpha=0.45, color="#7c3aed", label=f"{out_name} (N={len(wrong_out):,})")

        ax_l_wrong.set_xlabel("Logit-Wert ($z_j, j \\neq y$)", fontsize=9.5)
        ax_l_wrong.set_ylabel("Dichte", fontsize=9.5)
        ax_l_wrong.grid(True, linestyle=":", alpha=0.6)
        if len(wrong_in) > 0 or len(wrong_out) > 0:
            ax_l_wrong.legend(fontsize=7.2, loc="upper right")

        # Spalte 7: Realer Loss (Train vs. Test)
        plot_loss_overlay(
            ax=axes[row_idx, 7],
            in_vals=real_loss_res.get(in_key, np.array([])),
            out_vals=real_loss_res.get(out_key, np.array([])),
            in_name=in_name,
            out_name=out_name,
            loss_type=real_loss_type,
            color_in="#059669",
            color_out="#f59e0b",
            auc=auc_real,
        )

        # Spalte 8: Anderer Loss (Train vs. Test)
        plot_loss_overlay(
            ax=axes[row_idx, 8],
            in_vals=other_loss_res.get(in_key, np.array([])),
            out_vals=other_loss_res.get(out_key, np.array([])),
            in_name=in_name,
            out_name=out_name,
            loss_type=other_loss_type,
            color_in="#7c3aed",
            color_out="#ec4899",
            auc=auc_other,
        )

        # Spalte 9: Klassen-MSE (Train vs. Test): (z_y - 1)^2
        plot_loss_overlay(
            ax=axes[row_idx, 9],
            in_vals=tc_mse_res.get(in_key, np.array([])),
            out_vals=tc_mse_res.get(out_key, np.array([])),
            in_name=in_name,
            out_name=out_name,
            loss_type="mse_tc",
            color_in="#d97706",
            color_out="#6366f1",
            auc=auc_tc_mse,
        )

        # Spalte 10: InfoRMIA Score (Train vs. Test)
        plot_informia_overlay(
            ax=axes[row_idx, 10],
            in_vals=informia_res.get(in_key, np.array([])),
            out_vals=informia_res.get(out_key, np.array([])),
            in_name=in_name,
            out_name=out_name,
            color_in="#ea580c",
            color_out="#06b6d4",
            auc=auc_info,
        )

        # Zeilenbeschriftung ganz links
        axes[row_idx, 0].text(
            -0.34,
            0.5,
            row_label,
            transform=axes[row_idx, 0].transAxes,
            fontsize=12,
            fontweight="bold",
            va="center",
            ha="center",
            rotation=90,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8fafc", edgecolor="#94a3b8", linewidth=1.2),
        )

    plt.subplots_adjust(left=0.035, right=0.99, top=0.90, bottom=0.08, hspace=0.30, wspace=0.32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Optimierter 2x11 Verteilungs- & Dynamik-Plot gespeichert unter:\n  -> {output_path}")


# Kompatibilitäts-Aliase
plot_2row_10col_figure = plot_2row_11col_figure
plot_2row_9col_figure = plot_2row_11col_figure
plot_2row_8col_figure = plot_2row_11col_figure
plot_2row_5col_figure = plot_2row_11col_figure


def compute_roc_and_auc(in_scores: np.ndarray, out_scores: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Berechnet AUC sowie FPR und TPR für eine Attacke (in=1, out=0)."""
    if len(in_scores) == 0 or len(out_scores) == 0:
        return 0.5, np.array([0.0, 1.0]), np.array([0.0, 1.0])
    y_true = np.concatenate([np.ones(len(in_scores)), np.zeros(len(out_scores))])
    y_score = np.concatenate([in_scores, out_scores])
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = float(auc(fpr, tpr))
    return roc_auc, fpr, tpr


def plot_auc_comparison(
    ce_attack_results: dict,
    mse_attack_results: dict,
    step: int,
    output_path: Path,
):
    """Erstellt den vergleichenden AUC & ROC-Kurven-Plot für beide Modelle."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"MIA-Vergleich: Loss-Angriffe vs. InfoRMIA (Schritt t = {step})",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    models_data = [
        (ce_attack_results.get("model_name", "CE-Modell"), ce_attack_results, 0),
        (mse_attack_results.get("model_name", "MSE-Modell"), mse_attack_results, 1),
    ]

    for model_name, res, row_idx in models_data:
        ax_reg = axes[row_idx, 0]
        ax_can = axes[row_idx, 1]

        for ax, split_type, title_suffix in [
            (ax_reg, "regular", "Reguläre Daten (Train vs. Test)"),
            (ax_can, "canary", "Canary Daten (Canary Train vs. Canary Test)"),
        ]:
            data = res[split_type]
            auc_real, fpr_real, tpr_real = data["real_loss"]
            auc_other, fpr_other, tpr_other = data["other_loss"]
            auc_info, fpr_info, tpr_info = data["informia"]
            opt_a = data.get("optimal_a", 0.5)

            ax.plot(
                fpr_real,
                tpr_real,
                color="#059669",
                linewidth=2.2,
                label=f"Realer Loss ({data['real_loss_name']}): AUC = {auc_real:.3f}",
            )
            ax.plot(
                fpr_other,
                tpr_other,
                color="#7c3aed",
                linewidth=2.2,
                label=f"Anderer Loss ({data['other_loss_name']}): AUC = {auc_other:.3f}",
            )
            if "tc_mse" in data:
                auc_tc, fpr_tc, tpr_tc = data["tc_mse"]
                ax.plot(
                    fpr_tc,
                    tpr_tc,
                    color="#d97706",
                    linewidth=2.0,
                    linestyle=":",
                    label=f"Klassen-MSE $(z_y-1)^2$: AUC = {auc_tc:.3f}",
                )
            ax.plot(
                fpr_info,
                tpr_info,
                color="#ea580c",
                linewidth=2.5,
                linestyle="-",
                label=f"InfoRMIA (a={opt_a:.1f}): AUC = {auc_info:.3f}",
            )

            ax.plot([0, 1], [0, 1], color="#94a3b8", linestyle="--", linewidth=1.2, label="Zufall (AUC = 0.500)")

            ax.set_xlim([-0.02, 1.02])
            ax.set_ylim([-0.02, 1.04])
            ax.set_xlabel("False Positive Rate (FPR)", fontsize=11)
            ax.set_ylabel("True Positive Rate (TPR)", fontsize=11)
            ax.set_title(f"{model_name}\n{title_suffix}", fontsize=12, fontweight="bold", pad=8)
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(fontsize=9, loc="lower right", framealpha=0.95)

    plt.subplots_adjust(left=0.07, right=0.96, top=0.91, bottom=0.07, hspace=0.28, wspace=0.20)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"AUC-Vergleichs-Plot gespeichert unter:\n  -> {output_path}")


def evaluate_model_group(
    model_name: str,
    run_ids: list[str],
    real_loss_type: Literal["ce", "mse"],
    step: int,
    device: str,
    tracking_uri: str,
    output_dir: Path,
    fallback_a: float = 0.5,
    skip_val_tuning: bool = False,
    df_metrics: pd.DataFrame | None = None,
) -> dict:
    """Führt die vollständige Auswertung für eine 6-Run-Modellgruppe durch.
    
    1. Lädt Target-Modell (run_ids[0]) und berechnet Realen/Anderen Loss sowie Logits.
    2. Lädt Referenz-Modelle (run_ids[2:]).
    3. Lädt Validation-Modell (run_ids[1]) und tuned optimal_a in 0.1 Schritten (0.0 bis 1.0)
       separat für reguläre Daten und Canary-Daten.
    4. Berechnet InfoRMIA Scores für das Target-Modell unter Verwendung der optimalen a-Werte.
    """
    print(f"\n=======================================================")
    print(f"Starte Auswertung für: {model_name}")
    print(f"Target Run ID: {run_ids[0]} (model_index 0)")
    print(f"Val Run ID:    {run_ids[1]} (model_index 1)")
    print(f"Ref Run IDs:   {run_ids[2:]}")
    print(f"=======================================================")

    # 1. Target Modell und Daten laden
    print(f"  Lade Target-Modell ({run_ids[0]})...")
    m_target, dc_target, _ = load_model_and_data(run_ids[0], step, device, tracking_uri)

    norm_mean = dc_target.normalization.mean if dc_target.normalization else None
    norm_std = dc_target.normalization.std if dc_target.normalization else None

    # Signale für Target Modell
    print("  Berechne Signale für Target-Modell...")
    t_corr, t_wrong, t_ce, t_mse, t_p = evaluate_model_signals(
        m_target, dc_target.train, device, norm_mean, norm_std
    )
    te_corr, te_wrong, te_ce, te_mse, te_p = evaluate_model_signals(
        m_target, dc_target.test, device, norm_mean, norm_std
    )
    tc_corr, tc_wrong, tc_ce, tc_mse, tc_p = evaluate_model_signals(
        m_target, dc_target.train_canary, device, norm_mean, norm_std
    )
    tec_corr, tec_wrong, tec_ce, tec_mse, tec_p = evaluate_model_signals(
        m_target, dc_target.test_canary, device, norm_mean, norm_std
    )

    logits_res = {
        "train_reg": (t_corr, t_wrong),
        "test_reg": (te_corr, te_wrong),
        "train_canary": (tc_corr, tc_wrong),
        "test_canary": (tec_corr, tec_wrong),
    }

    if real_loss_type == "ce":
        real_loss_res = {"train_reg": t_ce, "test_reg": te_ce, "train_canary": tc_ce, "test_canary": tec_ce}
        other_loss_res = {"train_reg": t_mse, "test_reg": te_mse, "train_canary": tc_mse, "test_canary": tec_mse}
        other_loss_type = "mse"
        real_loss_label = "Cross-Entropy"
        other_loss_label = "One-Hot MSE"
    else:
        real_loss_res = {"train_reg": t_mse, "test_reg": te_mse, "train_canary": tc_mse, "test_canary": tec_mse}
        other_loss_res = {"train_reg": t_ce, "test_reg": te_ce, "train_canary": tc_ce, "test_canary": tec_ce}
        other_loss_type = "ce"
        real_loss_label = "One-Hot MSE"
        other_loss_label = "Cross-Entropy"

    # 2. Referenz-Modelle laden und evaluieren
    ref_models = []
    ref_dcs = []
    for r_id in run_ids[2:]:
        print(f"  Lade Referenzmodell ({r_id})...")
        m_ref, dc_ref, _ = load_model_and_data(r_id, step, device, tracking_uri)
        ref_models.append(m_ref)
        ref_dcs.append(dc_ref)

    print("  Berechne Vorhersagen der Referenzmodelle auf Target-Daten...")
    ref_p_train = np.array([evaluate_probs_only(m, dc_target.train, device, norm_mean, norm_std) for m in ref_models])
    ref_p_test = np.array([evaluate_probs_only(m, dc_target.test, device, norm_mean, norm_std) for m in ref_models])
    ref_p_ctrain = np.array([evaluate_probs_only(m, dc_target.train_canary, device, norm_mean, norm_std) for m in ref_models])
    ref_p_ctest = np.array([evaluate_probs_only(m, dc_target.test_canary, device, norm_mean, norm_std) for m in ref_models])

    # Memberships für Target-Daten
    train_idx = dc_target.train.indices
    ctrain_idx = dc_target.train_canary.indices

    mem_train = np.array([[1.0 if idx in set(dc.train.indices) else 0.0 for idx in train_idx] for dc in ref_dcs])
    mem_test = np.zeros((len(ref_dcs), len(te_p)))
    mem_ctrain = np.array([[1.0 if idx in set(dc.train_canary.indices) else 0.0 for idx in ctrain_idx] for dc in ref_dcs])
    mem_ctest = np.zeros((len(ref_dcs), len(tec_p)))

    # 3. Validation-Modell evaluieren & optimal_a in 0.1 Schritten bestimmen
    opt_a_reg = fallback_a
    opt_a_canary = fallback_a
    val_auc_reg = None
    val_auc_canary = None
    val_run_id = run_ids[1] if len(run_ids) > 1 else None

    if val_run_id and not skip_val_tuning:
        try:
            print(f"  Lade Validation-Modell ({val_run_id}) für optimal_a Tuning...")
            m_val, dc_val, _ = load_model_and_data(val_run_id, step, device, tracking_uri)
            norm_val_mean = dc_val.normalization.mean if dc_val.normalization else None
            norm_val_std = dc_val.normalization.std if dc_val.normalization else None

            print("  Berechne Vorhersagen für Validation-Modell...")
            val_p_train = evaluate_probs_only(m_val, dc_val.train, device, norm_val_mean, norm_val_std)
            val_p_test = evaluate_probs_only(m_val, dc_val.test, device, norm_val_mean, norm_val_std)
            val_p_ctrain = evaluate_probs_only(m_val, dc_val.train_canary, device, norm_val_mean, norm_val_std)
            val_p_ctest = evaluate_probs_only(m_val, dc_val.test_canary, device, norm_val_mean, norm_val_std)

            print("  Berechne Referenzmodell-Vorhersagen auf Validation-Daten...")
            ref_p_val_train = np.array([evaluate_probs_only(m, dc_val.train, device, norm_val_mean, norm_val_std) for m in ref_models])
            ref_p_val_test = np.array([evaluate_probs_only(m, dc_val.test, device, norm_val_mean, norm_val_std) for m in ref_models])
            ref_p_val_ctrain = np.array([evaluate_probs_only(m, dc_val.train_canary, device, norm_val_mean, norm_val_std) for m in ref_models])
            ref_p_val_ctest = np.array([evaluate_probs_only(m, dc_val.test_canary, device, norm_val_mean, norm_val_std) for m in ref_models])

            val_train_idx = dc_val.train.indices
            val_ctrain_idx = dc_val.train_canary.indices

            mem_val_train = np.array([[1.0 if idx in set(dc.train.indices) else 0.0 for idx in val_train_idx] for dc in ref_dcs])
            mem_val_test = np.zeros((len(ref_dcs), len(val_p_test)))
            mem_val_ctrain = np.array([[1.0 if idx in set(dc.train_canary.indices) else 0.0 for idx in val_ctrain_idx] for dc in ref_dcs])
            mem_val_ctest = np.zeros((len(ref_dcs), len(val_p_ctest)))

            print("  [Tuning] Prüfe optimal_a in 0.1 Schritten (0.0 bis 1.0) auf Validation-Modell...")
            opt_a_reg, val_auc_reg, sweep_reg = tune_optimal_a(
                val_p_in=val_p_train,
                val_p_out=val_p_test,
                ref_p_val_in=ref_p_val_train,
                ref_p_val_out=ref_p_val_test,
                mem_val_in=mem_val_train,
                mem_val_out=mem_val_test,
                val_pop_target_p=val_p_test,
                val_pop_ref_p=ref_p_val_test,
                split_name="regular",
            )
            print(f"    -> Regulär:  optimal_a = {opt_a_reg:.1f} (Val AUC: {val_auc_reg:.4f})")

            opt_a_canary, val_auc_canary, sweep_canary = tune_optimal_a(
                val_p_in=val_p_ctrain,
                val_p_out=val_p_ctest,
                ref_p_val_in=ref_p_val_ctrain,
                ref_p_val_out=ref_p_val_ctest,
                mem_val_in=mem_val_ctrain,
                mem_val_out=mem_val_ctest,
                val_pop_target_p=val_p_test,
                val_pop_ref_p=ref_p_val_test,
                split_name="canary",
            )
            print(f"    -> Canaries: optimal_a = {opt_a_canary:.1f} (Val AUC: {val_auc_canary:.4f})")

            print(f"    Sweep-Details für {model_name}:")
            for a_cand in sorted(sweep_reg.keys()):
                reg_m = " <-- OPT" if a_cand == opt_a_reg else ""
                can_m = " <-- OPT" if a_cand == opt_a_canary else ""
                print(f"      a = {a_cand:3.1f} | Val AUC (Reg): {sweep_reg[a_cand]:.4f}{reg_m:<8} | Val AUC (Canary): {sweep_canary[a_cand]:.4f}{can_m}")

        except Exception as e:
            print(f"  [HINWEIS] Validation-Modell ({val_run_id}) konnte nicht geladen werden ({e}).")
            print(f"            Verwende Fallback: optimal_a = {fallback_a:.1f}")
            opt_a_reg = fallback_a
            opt_a_canary = fallback_a
    else:
        print(f"  Verwende festen Wert offline_a = {fallback_a:.1f} (kein Validation-Tuning)")

    # 4. InfoRMIA Scores für Target berechnen
    print(f"  Berechne InfoRMIA Scores für Target (Regulär: a={opt_a_reg:.1f}, Canary: a={opt_a_canary:.1f})...")
    informia_train = compute_informia_scores(t_p, ref_p_train, mem_train, te_p, ref_p_test, offline_a=opt_a_reg)
    informia_test = compute_informia_scores(te_p, ref_p_test, mem_test, te_p, ref_p_test, offline_a=opt_a_reg)
    informia_ctrain = compute_informia_scores(tc_p, ref_p_ctrain, mem_ctrain, te_p, ref_p_test, offline_a=opt_a_canary)
    informia_ctest = compute_informia_scores(tec_p, ref_p_ctest, mem_ctest, te_p, ref_p_test, offline_a=opt_a_canary)

    informia_res = {
        "train_reg": informia_train,
        "test_reg": informia_test,
        "train_canary": informia_ctrain,
        "test_canary": informia_ctest,
    }

    # 5. Klassen-MSE (nur wahre Klasse): (z_y - 1)^2
    t_tc_mse = (t_corr - 1.0) ** 2
    te_tc_mse = (te_corr - 1.0) ** 2
    tc_tc_mse = (tc_corr - 1.0) ** 2
    tec_tc_mse = (tec_corr - 1.0) ** 2

    tc_mse_res = {
        "train_reg": t_tc_mse,
        "test_reg": te_tc_mse,
        "train_canary": tc_tc_mse,
        "test_canary": tec_tc_mse,
    }

    # 6. AUC Berechnung für ROC-Plot und Dynamik-Annotationen
    real_loss_in_reg = -real_loss_res["train_reg"]
    real_loss_out_reg = -real_loss_res["test_reg"]
    real_loss_in_can = -real_loss_res["train_canary"]
    real_loss_out_can = -real_loss_res["test_canary"]

    other_loss_in_reg = -other_loss_res["train_reg"]
    other_loss_out_reg = -other_loss_res["test_reg"]
    other_loss_in_can = -other_loss_res["train_canary"]
    other_loss_out_can = -other_loss_res["test_canary"]

    tc_mse_in_reg = -tc_mse_res["train_reg"]
    tc_mse_out_reg = -tc_mse_res["test_reg"]
    tc_mse_in_can = -tc_mse_res["train_canary"]
    tc_mse_out_can = -tc_mse_res["test_canary"]

    attack_results = {
        "model_name": model_name,
        "regular": {
            "real_loss": compute_roc_and_auc(real_loss_in_reg, real_loss_out_reg),
            "other_loss": compute_roc_and_auc(other_loss_in_reg, other_loss_out_reg),
            "tc_mse": compute_roc_and_auc(tc_mse_in_reg, tc_mse_out_reg),
            "logit_corr": compute_roc_and_auc(t_corr, te_corr),
            "informia": compute_roc_and_auc(informia_train, informia_test),
            "optimal_a": opt_a_reg,
            "val_auc": val_auc_reg,
            "real_loss_name": real_loss_label,
            "other_loss_name": other_loss_label,
        },
        "canary": {
            "real_loss": compute_roc_and_auc(real_loss_in_can, real_loss_out_can),
            "other_loss": compute_roc_and_auc(other_loss_in_can, other_loss_out_can),
            "tc_mse": compute_roc_and_auc(tc_mse_in_can, tc_mse_out_can),
            "logit_corr": compute_roc_and_auc(tc_corr, tec_corr),
            "informia": compute_roc_and_auc(informia_ctrain, informia_ctest),
            "optimal_a": opt_a_canary,
            "val_auc": val_auc_canary,
            "real_loss_name": real_loss_label,
            "other_loss_name": other_loss_label,
        },
    }

    # Optimierter 2x11 Verteilungs- & Dynamikplot speichern
    dist_out_path = output_dir / f"distribution_{model_name}_step_{step}.png"
    plot_2row_11col_figure(
        target_model_name=model_name,
        logits_res=logits_res,
        real_loss_label=real_loss_label,
        real_loss_type=real_loss_type,
        real_loss_res=real_loss_res,
        other_loss_label=other_loss_label,
        other_loss_type=other_loss_type,
        other_loss_res=other_loss_res,
        tc_mse_res=tc_mse_res,
        informia_res=informia_res,
        attack_results=attack_results,
        df_metrics=df_metrics,
        target_run_id=run_ids[0],
        all_run_ids=run_ids,
        step=step,
        output_path=dist_out_path,
    )

    return attack_results


def main():
    parser = argparse.ArgumentParser(
        description="Visualisiert Dynamiken über die Zeit und Logit-, Loss- und InfoRMIA-Verteilungen (2x8 Grid) sowie AUC-Vergleich."
    )
    parser.add_argument(
        "--step",
        "-t",
        type=int,
        default=DEFAULT_STEP,
        help=f"Checkpoint-Schritt t (Default: {DEFAULT_STEP})",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=DEFAULT_TRACKING_URI,
        help=f"MLflow Tracking-URI (Default: {DEFAULT_TRACKING_URI})",
    )
    parser.add_argument(
        "--parquet",
        type=str,
        default=str(DEFAULT_PARQUET),
        help=f"Pfad zur Parquet-Datei für Dynamiken über die Zeit (Default: {DEFAULT_PARQUET})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Ausgabeverzeichnis für die Diagramme (Default: plots)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda / cpu)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="madd",
        choices=["madd", "mnist_mlp", "no_grok_mnist_mlp"],
        help="Modell-Preset: 'madd' (MADD Transformer), 'mnist_mlp' (GROK MNIST MLP) oder 'no_grok_mnist_mlp'",
    )
    parser.add_argument(
        "--offline-a",
        type=float,
        default=0.5,
        help="Fallback offline_a Parameter (Default: 0.5)",
    )
    parser.add_argument(
        "--skip-val-tuning",
        action="store_true",
        help="Überspringt das Tuning auf dem Validation-Modell und verwendet direkt --offline-a",
    )

    args = parser.parse_args()

    Logger().setup()

    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.preset == "mnist_mlp":
        ce_model_name = "GROK_MNIST_CE_MLP"
        mse_model_name = "GROK_MNIST_MSE_MLP"
        ce_runs = CE_RUNS_MNIST
        mse_runs = MSE_RUNS_MNIST
    elif args.preset == "no_grok_mnist_mlp":
        ce_model_name = "NO_GROK_MNIST_CE_MLP"
        mse_model_name = "NO_GROK_MNIST_MSE_MLP"
        ce_runs = [
            "0b3934d3bf794857809d3ad1937f63f2",
            "9868e73cd699490dad741637936a64cd",
            "f0bf6393069b4b9c935e0a9a562199a0",
            "3b7d5fe0e3a34231ad2558676c6acf81",
            "ff21f1138d3e41c6b01764c2de3119e6",
            "088b42f7979f4c769bd6256d29b08ee0",
        ]
        mse_runs = [
            "8316021fe2b04ff181119c35e9b04c9e",
            "5b1baa4e4bb84b6389ea9a1cbe75d230",
            "c0825a6c1d684a319bd053d5ae5cf700",
            "16b26006f6e2472c86a7cf3b2943daaf",
            "4cf3d1ba5ff64058b20cf3fc02cee8f6",
            "93b1d61dad86440b88c43e777ed09205",
        ]
    else:
        ce_model_name = "NO_GROK_MADD_CE_TRANSFORMER"
        mse_model_name = "NO_GROK_MADD_MSE_TRANSFORMER"
        ce_runs = CE_RUNS_MADD
        mse_runs = MSE_RUNS_MADD

    parquet_path = Path(args.parquet)
    all_runs = list(dict.fromkeys(ce_runs + mse_runs))
    print(f"Lade Dynamik-Metriken aus Parquet: {parquet_path}")
    df_metrics = load_parquet_metrics(parquet_path, all_runs)
    if not df_metrics.empty:
        print(f"  -> {len(df_metrics):,} Einträge für {len(all_runs)} Runs geladen.")
    else:
        print("  -> Keine Parquet-Daten geladen (Zeitreihen-Spalten bleiben leer).")

    print(f"=== Gesamt-Analyse für Schritt t = {args.step} (Optimiertes 2x8 Layout) ===")
    print(f"Preset:       {args.preset}")
    print(f"CE Modell:    {ce_model_name}")
    print(f"MSE Modell:   {mse_model_name}")
    print(f"Tracking URI: {args.tracking_uri}")
    print(f"Device:       {args.device}")

    # 1. CE Gruppe
    ce_attack_res = evaluate_model_group(
        model_name=ce_model_name,
        run_ids=ce_runs,
        real_loss_type="ce",
        step=args.step,
        device=args.device,
        tracking_uri=args.tracking_uri,
        output_dir=output_dir,
        fallback_a=args.offline_a,
        skip_val_tuning=args.skip_val_tuning,
        df_metrics=df_metrics,
    )

    # 2. MSE Gruppe
    mse_attack_res = evaluate_model_group(
        model_name=mse_model_name,
        run_ids=mse_runs,
        real_loss_type="mse",
        step=args.step,
        device=args.device,
        tracking_uri=args.tracking_uri,
        output_dir=output_dir,
        fallback_a=args.offline_a,
        skip_val_tuning=args.skip_val_tuning,
        df_metrics=df_metrics,
    )

    # 3. Gemeinsamer AUC & ROC Plot
    auc_plot_path = output_dir / f"auc_comparison_ce_vs_mse_informia_step_{args.step}.png"
    plot_auc_comparison(
        ce_attack_results=ce_attack_res,
        mse_attack_results=mse_attack_res,
        step=args.step,
        output_path=auc_plot_path,
    )

    print("\n" + "=" * 96)
    print(f"ERGEBNIS-ÜBERSICHT: AUC-Werte bei Schritt t = {args.step}")
    print("=" * 96)
    print(f"{'Modell':<24} | {'Datensatz':<9} | {'Realer Loss':<11} | {'Anderer Loss':<12} | {'Klassen-MSE':<11} | {'opt a':<6} | {'InfoRMIA':<8}")
    print("-" * 96)

    for m_name, res in [
        (ce_model_name, ce_attack_res),
        (mse_model_name, mse_attack_res),
    ]:
        for s_key, s_label in [("regular", "Regulär"), ("canary", "Canary")]:
            d = res[s_key]
            auc_r = d["real_loss"][0]
            auc_o = d["other_loss"][0]
            auc_tc = d.get("tc_mse", (0.0,))[0]
            auc_i = d["informia"][0]
            opt_a = d.get("optimal_a", args.offline_a)
            print(f"{m_name:<24} | {s_label:<9} | {auc_r:<11.4f} | {auc_o:<12.4f} | {auc_tc:<11.4f} | {opt_a:<6.1f} | {auc_i:<8.4f}")
    print("=" * 96)


if __name__ == "__main__":
    main()
