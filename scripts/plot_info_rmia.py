import argparse
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Canonical order and display names for canary types (consistent with plot_canary_accuracy.py)
CANARY_TYPE_MAP = {
    'GAUSSIAN_NOISE': 'Gaussian Noise',
    'UNIFORM_NOISE': 'Uniform Noise',
    'OOD_NATURAL': 'OOD Natural',
    'SQUARE_WATERMARK': 'Watermark',
    'WATERMARK': 'Watermark',
    'LABEL_NOISE': 'Label Noise'
}

CANARY_ORDER = [
    'Gaussian Noise',
    'Uniform Noise',
    'OOD Natural',
    'Watermark',
    'Label Noise'
]

# Styling definitions
ATTACK_STYLES = {
    'InfoRMIA (Canary)': {
        'color': '#059669',     # Vibrant emerald green
        'linestyle': '--',
        'marker': 'o',
        'markersize': 5,
        'linewidth': 2.0,
    },
    'InfoRMIA (Non-Canary)': {
        'color': '#2563eb',     # Royal blue
        'linestyle': '-',
        'marker': 's',
        'markersize': 4.5,
        'linewidth': 2.0,
    },
}

ACCURACY_STYLES = {
    'Train': {'color': '#1f77b4', 'linestyle': '-', 'linewidth': 1.8},
    'Test': {'color': '#ff7f0e', 'linestyle': '-', 'linewidth': 1.8},
    'Canary Train': {'color': '#2ca02c', 'linestyle': '--', 'linewidth': 1.8},
    'Canary Test': {'color': '#d62728', 'linestyle': '--', 'linewidth': 1.8},
}


def parse_run_name(rname: str) -> tuple[str, str]:
    """Extract (canary_label, model_group) from a run_name string."""
    if not isinstance(rname, str):
        return 'Unknown Canary', 'Default_Model'
    for k, v in CANARY_TYPE_MAP.items():
        if rname.startswith(k + '_'):
            model = rname[len(k) + 1:]
            return v, model
    return 'Unknown Canary', rname


def determine_used_loss(run_name: str) -> str:
    """Determine which loss objective was used during model training."""
    r_lower = run_name.lower()
    if 'mse' in r_lower:
        return 'mse'
    if 'ce' in r_lower or 'cross_entropy' in r_lower:
        return 'ce'
    return 'unknown'


def setup_axes(ax: plt.Axes, log_x: bool = False, max_step: int | None = None, show_xlabel: bool = True):
    """Set up x-axis and y-axis scaling starting at 0."""
    if show_xlabel:
        ax.set_xlabel('Step', fontsize=10)
    if log_x:
        ax.set_xscale('log')
    else:
        if max_step is not None:
            ax.set_xlim(left=0, right=max_step)
        else:
            ax.set_xlim(left=0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3, linestyle='--')


def plot_attacks_subplot(
    ax: plt.Axes,
    info_sub_df: pd.DataFrame,
    mlflow_sub_df: pd.DataFrame,
    title: str,
    used_loss: str,
    log_x: bool = False,
    max_step: int | None = None,
    show_ylabel: bool = True,
    show_legend: bool = True,
    show_xlabel: bool = True
):
    """Plots InfoRMIA (Canary vs Non-Canary) and both MSE & CE loss-based attack AUCs over steps."""
    plotted = False

    # 1. InfoRMIA curves
    if not info_sub_df.empty:
        for is_canary, label in [(True, 'InfoRMIA (Canary)'), (False, 'InfoRMIA (Non-Canary)')]:
            canary_rows = info_sub_df[info_sub_df['canary'] == is_canary]
            if canary_rows.empty:
                continue
            agg = canary_rows.groupby('step')['auc'].agg(['mean', 'std']).reset_index()
            if agg.empty:
                continue
            plotted = True
            style = ATTACK_STYLES[label]
            p = ax.plot(
                agg['step'],
                agg['mean'],
                label=label,
                color=style['color'],
                linestyle=style['linestyle'],
                marker=style['marker'],
                markersize=style['markersize'],
                linewidth=style['linewidth'],
                alpha=0.9
            )
            color = p[0].get_color()
            if not agg['std'].isna().all() and (agg['std'] > 0).any():
                ax.fill_between(
                    agg['step'],
                    agg['mean'] - agg['std'],
                    agg['mean'] + agg['std'],
                    color=color,
                    alpha=0.2
                )

    # 2. Loss-based attack curves: report both MSE and CE loss, noting the used loss
    if not mlflow_sub_df.empty:
        # Order: put the used loss first, then the alternate loss, followed by any canary loss attacks
        preferred_losses = ['mse', 'ce'] if used_loss == 'mse' else ['ce', 'mse']
        loss_specs = []
        for l_type in preferred_losses:
            loss_specs.append((f'eval/attack/{l_type}_loss/auc', l_type, False))
        for l_type in preferred_losses:
            loss_specs.append((f'eval/attack/canary_{l_type}_loss/auc', l_type, True))

        for metric_key, loss_type, is_canary in loss_specs:
            loss_rows = mlflow_sub_df[mlflow_sub_df['metric_name'] == metric_key]
            if loss_rows.empty:
                continue
            agg_loss = loss_rows.groupby('step')['value'].agg(['mean', 'std']).reset_index()
            if agg_loss.empty:
                continue
            plotted = True

            is_used = (loss_type == used_loss)
            loss_upper = loss_type.upper()

            if is_canary:
                label = f"Loss Attack {loss_upper} (Canary)"
                color = "#d97706" if loss_type == "mse" else "#0d9488"
                linestyle = "--"
                marker = "d" if loss_type == "mse" else "*"
                linewidth = 1.8
            else:
                if is_used:
                    label = f"Loss Attack {loss_upper} [Used]"
                    color = "#dc2626"
                    linestyle = "-."
                    marker = "^"
                    linewidth = 1.8
                else:
                    label = f"Loss Attack {loss_upper}"
                    color = "#7c3aed"
                    linestyle = ":"
                    marker = "v"
                    linewidth = 1.5

            p = ax.plot(
                agg_loss['step'],
                agg_loss['mean'],
                label=label,
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=4.5,
                linewidth=linewidth,
                alpha=0.85
            )
            color_plot = p[0].get_color()
            if not agg_loss['std'].isna().all() and (agg_loss['std'] > 0).any():
                ax.fill_between(
                    agg_loss['step'],
                    agg_loss['mean'] - agg_loss['std'],
                    agg_loss['mean'] + agg_loss['std'],
                    color=color_plot,
                    alpha=0.15
                )

    # Reference random guessing baseline
    ax.axhline(0.5, color='#6b7280', linestyle=':', linewidth=1.2, label='Random (0.5)')

    if not plotted:
        ax.text(0.5, 0.5, 'No Attack Data', ha='center', va='center', transform=ax.transAxes)

    ax.set_title(title, fontsize=11, fontweight='bold')
    setup_axes(ax, log_x=log_x, max_step=max_step, show_xlabel=show_xlabel)
    if show_ylabel:
        ax.set_ylabel('Attack AUC', fontsize=10)

    if show_legend and plotted:
        ax.legend(title='Attack Method', fontsize='small', loc='lower right', framealpha=0.85)


def plot_accuracy_subplot(
    ax: plt.Axes,
    mlflow_sub_df: pd.DataFrame,
    title: str,
    log_x: bool = False,
    max_step: int | None = None,
    show_ylabel: bool = True,
    show_legend: bool = True,
    show_xlabel: bool = False
):
    """Plots task accuracy (Train, Test, Canary Train, Canary Test) over steps."""
    metric_map = {
        'eval/train/accuracy': 'Train',
        'eval/test/accuracy': 'Test',
        'eval/train/canary_accuracy': 'Canary Train',
        'eval/test/canary_accuracy': 'Canary Test',
    }

    plotted = False
    if not mlflow_sub_df.empty:
        for m_name, label in metric_map.items():
            rows = mlflow_sub_df[mlflow_sub_df['metric_name'] == m_name]
            if rows.empty:
                continue
            agg = rows.groupby('step')['value'].agg(['mean', 'std']).reset_index()
            if agg.empty:
                continue
            plotted = True
            style = ACCURACY_STYLES.get(label, {'color': '#6b7280', 'linestyle': '-', 'linewidth': 1.5})
            p = ax.plot(
                agg['step'],
                agg['mean'],
                label=label,
                color=style['color'],
                linestyle=style['linestyle'],
                linewidth=style.get('linewidth', 1.8),
                alpha=0.9
            )
            color = p[0].get_color()
            if not agg['std'].isna().all() and (agg['std'] > 0).any():
                ax.fill_between(
                    agg['step'],
                    agg['mean'] - agg['std'],
                    agg['mean'] + agg['std'],
                    color=color,
                    alpha=0.18
                )

    if not plotted:
        ax.text(0.5, 0.5, 'No Accuracy Data', ha='center', va='center', transform=ax.transAxes)

    ax.set_title(title, fontsize=11, fontweight='bold')
    setup_axes(ax, log_x=log_x, max_step=max_step, show_xlabel=show_xlabel)
    if show_ylabel:
        ax.set_ylabel('Accuracy', fontsize=10)

    if show_legend and plotted:
        ax.legend(title='Metric', fontsize='small', loc='best', framealpha=0.85)


def generate_combined_plots(
    model_name: str,
    ordered_canaries: list[str],
    model_info_df: pd.DataFrame,
    model_mlflow_df: pd.DataFrame,
    output_dir: str,
    log_x: bool = False,
    save_png: bool = False
):
    """Generates a 2-row figure: Row 1 = (a) Accuracy, Row 2 = (b) Attack AUC without suptitle."""
    n_canaries = len(ordered_canaries)
    if n_canaries == 0:
        return

    # Determine maximum step for setting linear xlim
    max_steps = []
    if not model_info_df.empty and 'step' in model_info_df.columns:
        max_steps.append(model_info_df['step'].max())
    if not model_mlflow_df.empty and 'step' in model_mlflow_df.columns:
        max_steps.append(model_mlflow_df['step'].max())
    max_step = max(max_steps) if max_steps else 150000

    fig_w = 7.5 if n_canaries == 1 else max(5.5 * n_canaries, 8.0)
    fig_h = 6.8
    fig, axes = plt.subplots(2, n_canaries, figsize=(fig_w, fig_h), sharex='col', sharey='row', squeeze=False)

    for idx, canary_name in enumerate(ordered_canaries):
        ax_acc = axes[0, idx]
        ax_atk = axes[1, idx]

        canary_info_df = model_info_df[model_info_df['canary_label'] == canary_name]
        canary_mlflow_df = model_mlflow_df[model_mlflow_df['canary_label'] == canary_name]

        run_name = ""
        if not canary_info_df.empty:
            run_name = canary_info_df['run_name'].iloc[0]
        elif not canary_mlflow_df.empty:
            run_name = canary_mlflow_df['run_name'].iloc[0]

        used_loss = determine_used_loss(run_name)
        show_ylabel = (idx == 0)

        # Title formatting: if 1 canary, strictly "(a) Accuracy" and "(b) Attack AUC"
        if n_canaries == 1:
            title_acc = "(a) Accuracy"
            title_atk = "(b) Attack AUC"
        else:
            title_acc = f"(a) Accuracy - {canary_name}"
            title_atk = f"(b) Attack AUC - {canary_name}"

        # Top row: Accuracy (no x-label, shared with bottom row)
        plot_accuracy_subplot(
            ax=ax_acc,
            mlflow_sub_df=canary_mlflow_df,
            title=title_acc,
            log_x=log_x,
            max_step=max_step,
            show_ylabel=show_ylabel,
            show_legend=(idx == 0),
            show_xlabel=False
        )

        # Bottom row: Attacks (starts at 0 on y-axis, has x-label)
        plot_attacks_subplot(
            ax=ax_atk,
            info_sub_df=canary_info_df,
            mlflow_sub_df=canary_mlflow_df,
            title=title_atk,
            used_loss=used_loss,
            log_x=log_x,
            max_step=max_step,
            show_ylabel=show_ylabel,
            show_legend=(idx == 0),
            show_xlabel=True
        )

    plt.tight_layout()

    clean_name = str(model_name).replace("/", "-").replace(":", "-").replace(" ", "_")
    out_pdf = os.path.join(output_dir, f"{clean_name}_combined.pdf")
    plt.savefig(out_pdf, bbox_inches='tight')
    if save_png:
        out_png = os.path.join(output_dir, f"{clean_name}_combined.png")
        plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_pdf}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize InfoRMIA results and loss-based attacks comparing canary vs non-canary over time."
    )

    default_informia = "cache/canary-selection/informia_results.parquet"
    if not os.path.exists(default_informia) and os.path.exists("informia_results.parquet"):
        default_informia = "informia_results.parquet"

    default_mlflow = "cache/canary-selection_mlflow_export.parquet"
    if not os.path.exists(default_mlflow) and os.path.exists("canary-selection_mlflow_export.parquet"):
        default_mlflow = "canary-selection_mlflow_export.parquet"

    parser.add_argument(
        "--informia-input", "-i",
        type=str,
        default=default_informia,
        help="Path to InfoRMIA parquet results (default: cache/canary-selection/informia_results.parquet)"
    )
    parser.add_argument(
        "--mlflow-input", "-m",
        type=str,
        default=default_mlflow,
        help="Path to MLflow metrics export parquet (default: canary-selection_mlflow_export.parquet)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="plots/info_rmia",
        help="Directory to save the generated PDF plots (default: plots/info_rmia)"
    )
    parser.add_argument(
        "--plot-type",
        choices=["combined", "attacks", "both"],
        default="combined",
        help="Type of plots to generate: 'combined' (default), 'attacks', or 'both'"
    )
    parser.add_argument(
        "--log-x",
        action="store_true",
        help="Use logarithmic scale for x-axis instead of linear scale (default is linear)"
    )
    parser.add_argument(
        "--save-png",
        action="store_true",
        help="Also export PNG images (300 DPI) alongside the PDF reports (default: PDF only)"
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load InfoRMIA results
    if not os.path.exists(args.informia_input):
        print(f"Error: InfoRMIA results file '{args.informia_input}' not found.")
        return

    print(f"Loading InfoRMIA results from '{args.informia_input}'...")
    df_info = pd.read_parquet(args.informia_input)
    canaries, models = zip(*df_info['run_name'].map(parse_run_name))
    df_info['canary_label'] = canaries
    df_info['model_group'] = models

    # 2. Load MLflow metrics export
    df_mlflow = pd.DataFrame()
    if os.path.exists(args.mlflow_input):
        print(f"Loading MLflow metrics from '{args.mlflow_input}'...")
        needed_metrics = {
            'eval/train/accuracy',
            'eval/test/accuracy',
            'eval/train/canary_accuracy',
            'eval/test/canary_accuracy',
            'eval/attack/ce_loss/auc',
            'eval/attack/mse_loss/auc',
            'eval/attack/canary_ce_loss/auc',
            'eval/attack/canary_mse_loss/auc',
        }
        df_mlflow = pd.read_parquet(
            args.mlflow_input,
            columns=['run_name', 'metric_name', 'value', 'step', 'run_id']
        )
        df_mlflow = df_mlflow[df_mlflow['metric_name'].isin(needed_metrics)]
        canaries_ml, models_ml = zip(*df_mlflow['run_name'].map(parse_run_name))
        df_mlflow['canary_label'] = canaries_ml
        df_mlflow['model_group'] = models_ml
    else:
        print(f"Warning: MLflow export '{args.mlflow_input}' not found. Only InfoRMIA data will be plotted.")

    # 3. Identify all model groups present in either dataset
    info_models = set(df_info['model_group'].unique())
    mlflow_models = set(df_mlflow['model_group'].unique()) if not df_mlflow.empty else set()
    all_models = sorted(info_models | mlflow_models)

    print(f"Found {len(all_models)} model configurations to process.")

    for model_name in all_models:
        model_info_df = df_info[df_info['model_group'] == model_name] if not df_info.empty else pd.DataFrame()
        model_mlflow_df = df_mlflow[df_mlflow['model_group'] == model_name] if not df_mlflow.empty else pd.DataFrame()

        avail_canaries = set()
        if not model_info_df.empty:
            avail_canaries.update(model_info_df['canary_label'].unique())
        if not model_mlflow_df.empty:
            if not model_info_df.empty:
                avail_canaries = set(model_info_df['canary_label'].unique())
            else:
                avail_canaries.update(model_mlflow_df['canary_label'].unique())

        ordered_canaries = [c for c in CANARY_ORDER if c in avail_canaries]
        for c in sorted(avail_canaries):
            if c not in ordered_canaries:
                ordered_canaries.append(c)

        if not ordered_canaries:
            continue

        print(f"\nProcessing model: {model_name} (Canaries: {ordered_canaries})")

        generate_combined_plots(
            model_name=model_name,
            ordered_canaries=ordered_canaries,
            model_info_df=model_info_df,
            model_mlflow_df=model_mlflow_df,
            output_dir=args.output_dir,
            log_x=args.log_x,
            save_png=args.save_png
        )

    print(f"\nAll visualizations successfully generated in '{args.output_dir}'.")


if __name__ == '__main__':
    main()
