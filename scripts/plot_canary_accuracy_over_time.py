"""Plottet Train, Test, Canary Train und Canary Test Accuracy über die Zeit (Schritte).
Hebt den ausgewählten Schritt (Default: t = 20000) mit vertikaler und horizontalen
Referenzlinien für die Genauigkeitswerte hervor.

Alle Daten stammen aus der lokalen Parquet-Datei:
cache/canary-selection-v1_mlflow_export.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_PARQUET = PROJECT_ROOT / "cache" / "canary-selection-v1_mlflow_export.parquet"
DEFAULT_STEP = 20000

# 6 Runs für CE Transformer
CE_RUNS = [
    "0a341d15fecf4c3fa471fb6a35fb5e26",  # Target (model_index 0)
    "d0c6dd6ca6694ebfb978fa6575f77163",  # Validation (model_index 1)
    "175346abb6054cdcbcd9c7b1c8fa7bb8",  # Ref 0
    "4166ce0921d142b49bb629eef2e4a493",  # Ref 1
    "4071bcece5604426bedb1521bcfb40ad",  # Ref 2
    "3978800e6aa542068b75fb8f599b96d5",  # Ref 3
]

# 6 Runs für MSE Transformer
MSE_RUNS = [
    "94558a882eca40b9b4a15972b2ee6f20",  # Target (model_index 0)
    "bb77cb85eb3742a5843a7f6d445dc71f",  # Validation (model_index 1)
    "f6be708ddce6408892f15adc948826a7",  # Ref 0
    "41d3bfe6a9f44aac8f733be73e761e0e",  # Ref 1
    "bbbda71e139b4a51ba3496c942167e74",  # Ref 2
    "f9001c82258f42bbac54f7727ca78d5d",  # Ref 3
]

METRICS = [
    ("eval/train/accuracy", "Train Accuracy", "#2563eb", "-"),
    ("eval/test/accuracy", "Test Accuracy", "#059669", "-"),
    ("eval/train/canary_accuracy", "Canary Train Accuracy", "#dc2626", "--"),
    ("eval/test/canary_accuracy", "Canary Test Accuracy", "#9333ea", ":"),
]


def plot_accuracy_panel(
    ax: plt.Axes,
    df_group: pd.DataFrame,
    target_run_id: str,
    all_run_ids: list[str],
    model_title: str,
    selected_step: int,
    show_all_runs_band: bool = True,
):
    """Zeichnet die Genauigkeitskurven für eine Modellgruppe mit Referenzlinien bei selected_step."""
    step_target_vals: dict[str, float] = {}

    for metric_key, label_name, color, linestyle in METRICS:
        m_df = df_group[df_group["metric_name"] == metric_key]
        if m_df.empty:
            continue

        # Target Kurve
        target_df = m_df[m_df["run_id"] == target_run_id].sort_values("step")
        if not target_df.empty:
            # Duplikate pro Step entfernen falls vorhanden
            t_clean = target_df.drop_duplicates(subset=["step"])
            ax.plot(
                t_clean["step"],
                t_clean["value"],
                label=f"{label_name} (Target)",
                color=color,
                linestyle=linestyle,
                linewidth=2.4,
                alpha=0.95,
                zorder=4,
            )

            # Wert am ausgewählten Schritt finden
            val_at_step = t_clean[t_clean["step"] == selected_step]
            if not val_at_step.empty:
                step_target_vals[label_name] = float(val_at_step["value"].iloc[0])

        # Mittelwert und Band über alle 6 Runs
        if show_all_runs_band and len(all_run_ids) > 1:
            all_clean = m_df.groupby("step")["value"].agg(["mean", "std"]).reset_index()
            ax.fill_between(
                all_clean["step"],
                np.clip(all_clean["mean"] - all_clean["std"], 0.0, 1.0),
                np.clip(all_clean["mean"] + all_clean["std"], 0.0, 1.0),
                color=color,
                alpha=0.12,
                zorder=2,
            )

    # Vertikale Linie für den ausgewählten Schritt t
    ax.axvline(
        x=selected_step,
        color="#334155",
        linestyle="--",
        linewidth=1.8,
        label=f"Ausgewählter Schritt ($t = {selected_step:,}$)",
        zorder=5,
    )

    # Horizontale Linien für die Target-Genauigkeiten am Schritt
    text_lines = [f"Werte bei $t = {selected_step:,}$:"]
    # Sortieren nach Y-Wert, um Label-Überlappung zu handhaben
    sorted_items = sorted(step_target_vals.items(), key=lambda x: x[1])

    # Versatz für Textannotationen
    for label_name, y_val in sorted_items:
        # Farbe zuordnen
        match_color = "#334155"
        for m_k, l_n, c, _ in METRICS:
            if l_n == label_name:
                match_color = c
                break

        # Horizontale Linie
        ax.axhline(
            y=y_val,
            color=match_color,
            linestyle=":",
            linewidth=1.3,
            alpha=0.8,
            zorder=3,
        )

        # Punkt am Schnittpunkt
        ax.scatter(
            [selected_step],
            [y_val],
            color=match_color,
            s=55,
            edgecolors="#ffffff",
            linewidths=1.5,
            zorder=6,
        )

        pct_str = f"{y_val * 100:.1f}%"
        text_lines.append(f"  • {label_name}: {pct_str} ({y_val:.4f})")

    # Annotations-Box mit den Werten am Schritt
    info_box_text = "\n".join(text_lines)
    ax.text(
        0.03,
        0.50,
        info_box_text,
        transform=ax.transAxes,
        fontsize=9.5,
        fontfamily="monospace",
        verticalalignment="center",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#f8fafc",
            edgecolor="#cbd5e1",
            alpha=0.92,
            linewidth=1.2,
        ),
        zorder=10,
    )

    ax.set_title(model_title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Trainings-Schritt ($t$)", fontsize=11)
    ax.set_ylabel("Genauigkeit (Accuracy)", fontsize=11)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlim(0, 50500)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.92)


def main():
    parser = argparse.ArgumentParser(
        description="Plottet Train, Test, Canary Accuracy über die Zeit mit horizontaler/vertikaler Referenzlinie."
    )
    parser.add_argument(
        "--parquet",
        type=str,
        default=str(DEFAULT_PARQUET),
        help=f"Pfad zur Parquet-Datei (Default: {DEFAULT_PARQUET})",
    )
    parser.add_argument(
        "--step",
        "-t",
        type=int,
        default=DEFAULT_STEP,
        help=f"Ausgewählter Schritt für die Referenzlinie (Default: {DEFAULT_STEP})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "plots"),
        help="Ausgabeverzeichnis für die Diagramme",
    )

    args = parser.parse_args()
    parquet_path = Path(args.parquet)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not parquet_path.is_file():
        raise FileNotFoundError(f"Parquet-Datei nicht gefunden: {parquet_path}")

    print(f"Lade Daten aus {parquet_path}...")
    columns = ["run_id", "run_name", "metric_name", "step", "value"]
    all_runs = CE_RUNS + MSE_RUNS
    df = pd.read_parquet(parquet_path, columns=columns)
    df_filtered = df[df["run_id"].isin(all_runs)]

    print(f"Gefundene Metriken für {len(all_runs)} Runs: {len(df_filtered):,} Zeilen")

    fig, axes = plt.subplots(1, 2, figsize=(18, 7.5), sharey=True)
    fig.suptitle(
        f"Trainingsverlauf & Canary-Memorierung über die Zeit\nVergleich CE vs. MSE mit Referenzlinien bei Schritt t = {args.step:,}",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )

    # Linker Plot: CE Transformer
    df_ce = df_filtered[df_filtered["run_id"].isin(CE_RUNS)]
    plot_accuracy_panel(
        ax=axes[0],
        df_group=df_ce,
        target_run_id=CE_RUNS[0],
        all_run_ids=CE_RUNS,
        model_title="NO_GROK_MADD_CE_TRANSFORMER (Label Noise)\nTarget Run: " + CE_RUNS[0][:8],
        selected_step=args.step,
    )

    # Rechter Plot: MSE Transformer
    df_mse = df_filtered[df_filtered["run_id"].isin(MSE_RUNS)]
    plot_accuracy_panel(
        ax=axes[1],
        df_group=df_mse,
        target_run_id=MSE_RUNS[0],
        all_run_ids=MSE_RUNS,
        model_title="NO_GROK_MADD_MSE_TRANSFORMER (Label Noise)\nTarget Run: " + MSE_RUNS[0][:8],
        selected_step=args.step,
    )

    plt.tight_layout()
    out_png = output_dir / f"accuracy_over_time_step_{args.step}.png"
    out_pdf = output_dir / f"accuracy_over_time_step_{args.step}.pdf"

    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()

    print(f"Diagramm erfolgreich gespeichert:\n  -> {out_png}\n  -> {out_pdf}")


if __name__ == "__main__":
    main()
