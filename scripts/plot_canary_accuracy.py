import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Canonical order and display names for canary types
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

METRIC_STYLES = {
    'Train': {'color': '#1f77b4', 'linestyle': '-'},          # Blue solid
    'Test': {'color': '#ff7f0e', 'linestyle': '-'},           # Orange solid
    'Canary Train': {'color': '#2ca02c', 'linestyle': '--'},   # Green dashed
    'Canary Test': {'color': '#d62728', 'linestyle': '--'},    # Red dashed
}

def get_polished_name(m):
    m_lower = m.lower()
    if 'canary' in m_lower and 'train' in m_lower:
        return 'Canary Train'
    if 'canary' in m_lower and 'test' in m_lower:
        return 'Canary Test'
    if 'train' in m_lower and ('acc' in m_lower or 'accuracy' in m_lower):
        return 'Train'
    if 'test' in m_lower and ('acc' in m_lower or 'accuracy' in m_lower):
        return 'Test'
    return m

def parse_run_name(rname):
    if not isinstance(rname, str):
        return 'Unknown Canary', 'Default_Model'
    for k, v in CANARY_TYPE_MAP.items():
        if rname.startswith(k + '_'):
            model = rname[len(k) + 1:]
            return v, model
    return 'Unknown Canary', rname

def plot_canary_subplot(ax, df_plot, metric_names, title, ylabel='Accuracy', linear_x=False, show_ylabel=True, show_legend=True):
    if df_plot.empty:
        ax.text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center')
        ax.set_title(title, fontsize=12)
        return

    colors = plt.cm.tab10.colors
    c_idx = 0
    plotted = False

    for m in metric_names:
        m_data = df_plot[df_plot['metric_name'] == m]
        if m_data.empty:
            continue
        plotted = True
        agg_data = m_data.groupby('step')['value'].agg(['mean', 'std']).reset_index()
        label_name = get_polished_name(m)
        
        style = METRIC_STYLES.get(label_name, {'color': colors[c_idx % len(colors)], 'linestyle': '-'})
        color = style['color']
        linestyle = style.get('linestyle', '-')
        
        p = ax.plot(agg_data['step'], agg_data['mean'], label=label_name, color=color, linestyle=linestyle, linewidth=1.8)
        c = p[0].get_color()
        if not agg_data['std'].isna().all():
            ax.fill_between(agg_data['step'], agg_data['mean'] - agg_data['std'], agg_data['mean'] + agg_data['std'], color=c, alpha=0.2)
        c_idx += 1
        
    if not plotted:
        ax.text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center')

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Step', fontsize=10)
    if not linear_x:
        ax.set_xscale('log')
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3, linestyle='--')
    if show_legend and plotted:
        ax.legend(title='Metric', fontsize='small', loc='best', framealpha=0.8)

def main():
    parser = argparse.ArgumentParser(description="Generate aggregated canary accuracy plots per model configuration")
    default_input = "cache/canary-selection_mlflow_export.parquet"
    if not os.path.exists(default_input) and os.path.exists("canary-selection_mlflow_export.parquet"):
        default_input = "canary-selection_mlflow_export.parquet"
    parser.add_argument("--input", "-i", type=str, default=default_input, help="Path to parquet export")
    parser.add_argument("--output_dir", "-o", type=str, default="plots/canary_accuracy", help="Directory to save the generated PDF plots")
    parser.add_argument("--linear_x", action="store_true", help="Use linear scale for x-axis instead of log scale")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.exists(args.input) and os.path.exists("canary-selection_mlflow_export.parquet"):
        args.input = "canary-selection_mlflow_export.parquet"
    print(f"Loading data from {args.input}...")
    df = pd.read_parquet(args.input)
    
    # Parse canary type and model group
    if 'run_name' in df.columns:
        canaries, models = zip(*df['run_name'].map(parse_run_name))
        df['canary_label'] = canaries
        df['model_group'] = models
    else:
        param_cols = [c for c in df.columns if c.startswith('params.')]
        group_cols = [c for c in param_cols if not c.endswith('.seed')]
        df['model_group'] = df[group_cols].astype(str).agg('_'.join, axis=1) if group_cols else 'Default_Model'
        df['canary_label'] = 'Default Canary'

    all_metrics = df['metric_name'].unique()
    
    # Preferred order of metrics
    preferred_order = [
        'eval/train/accuracy',
        'eval/test/accuracy',
        'eval/train/canary_accuracy',
        'eval/test/canary_accuracy'
    ]
    target_accuracy_metrics = [pref for pref in preferred_order if pref in all_metrics]
    if not target_accuracy_metrics:
        target_accuracy_metrics = [
            m for m in all_metrics 
            if ('accuracy' in m.lower() or 'acc' in m.lower()) and ('train' in m.lower() or 'test' in m.lower() or 'canary' in m.lower())
        ]
        
    print(f"Identified accuracy metrics: {target_accuracy_metrics}")
    
    grouped_models = df.groupby('model_group')
    print(f"Found {len(grouped_models)} unique model configurations.")

    num_saved = 0
    for model_name, model_df in grouped_models:
        clean_model_name = str(model_name).replace("/", "-").replace(":", "-").replace(" ", "_")
        
        # Determine unique canaries for this model in canonical order
        avail_canaries = model_df['canary_label'].unique().tolist()
        ordered_canaries = [c for c in CANARY_ORDER if c in avail_canaries]
        for c in sorted(avail_canaries):
            if c not in ordered_canaries:
                ordered_canaries.append(c)
                
        n_canaries = len(ordered_canaries)
        if n_canaries == 0:
            continue
            
        n_rows, n_cols = 2, 3
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14.0, 7.5), sharey=True)
        axes_flat = axes.flatten()
        
        for idx, canary_name in enumerate(ordered_canaries):
            if idx >= len(axes_flat):
                break
            ax = axes_flat[idx]
            canary_df = model_df[model_df['canary_label'] == canary_name]
            
            letter_prefix = chr(ord('a') + idx)
            subplot_title = f"({letter_prefix}) {canary_name}"
            
            show_ylabel = (idx % n_cols == 0)
            
            plot_canary_subplot(
                ax, canary_df, target_accuracy_metrics,
                title=subplot_title,
                ylabel='Accuracy',
                linear_x=args.linear_x,
                show_ylabel=show_ylabel,
                show_legend=(idx == 0 or idx == n_canaries - 1)
            )
            
        # Turn off any unused subplots (e.g., the 6th cell when 5 canaries are present)
        for idx in range(n_canaries, len(axes_flat)):
            axes_flat[idx].axis("off")
            
        plt.tight_layout()
        out_pdf = os.path.join(args.output_dir, f"{clean_model_name}_canary_accuracy.pdf")
        plt.savefig(out_pdf)
        plt.close(fig)
        num_saved += 1
        print(f"[{num_saved}/{len(grouped_models)}] Saved {out_pdf}")

    print(f"\nAll {num_saved} model PDFs saved to '{args.output_dir}'.")

if __name__ == "__main__":
    main()
