import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def get_model_name(row, param_cols):
    parts = []
    for col in param_cols:
        val = row[col]
        if pd.notna(val) and val != "None" and val != "":
            clean_col = col.replace("params.", "")
            parts.append(f"{clean_col}={val}")
    if not parts:
        return "Default_Model"
    return "_".join(parts).replace("/", "-").replace(":", "-").replace(" ", "_")


def parse_frequency(val):
    if isinstance(val, int):
        return val
    val_s = str(val).strip().lower()
    if val_s.endswith("k"):
        return int(float(val_s[:-1]) * 1000)
    elif val_s.endswith("m"):
        return int(float(val_s[:-1]) * 1000000)
    return int(val_s)


def get_polished_name(m):
    if m == "eval/train/accuracy":
        return "Train"
    if m == "eval/test/accuracy":
        return "Test"
    if "loss" in m and "train" in m:
        return "Train"
    if "loss" in m and "test" in m:
        return "Test"
    if m == "eval/weight_norm/total":
        return "Total"
    if m.startswith("eval/weight_norm/"):
        return "Last Layer"
    if m == "eval/nc/nc1":
        return "NC1"
    if m == "eval/nc/rnc1/train":
        return "RNC1"
    if m == "eval/nc/rnc1/test":
        return "RNC1 Test"
    if m.startswith("eval/nc/nc2"):
        name = m.replace("eval/nc/", "").replace("_", " ").title()
        return name.replace("Nc2", "NC2")
    if m.startswith("eval/nc/nc3"):
        return "NC3"
    if m.startswith("eval/nc/nc4"):
        return "NC4"
    return m


def plot_aggregated(
    ax,
    df_plot,
    metric_names,
    title,
    ylabel,
    twin_ax=None,
    twin_metrics=None,
    show_legend=True,
    vline_x=None,
    show_xlabel=True,
):
    if df_plot.empty:
        ax.text(0.5, 0.5, "No data", horizontalalignment="center", verticalalignment="center")
        if title:
            ax.set_title(title)
        return

    colors = plt.cm.tab10.colors
    c_idx = 0
    plotted = False

    for m in metric_names:
        m_data = df_plot[df_plot["metric_name"] == m]
        if m_data.empty:
            continue
        plotted = True
        agg_data = m_data.groupby("step")["value"].agg(["mean", "std"]).reset_index()
        label_name = get_polished_name(m)
        p = ax.plot(
            agg_data["step"], agg_data["mean"], label=label_name, color=colors[c_idx % len(colors)]
        )
        color = p[0].get_color()
        if not agg_data["std"].isna().all():
            ax.fill_between(
                agg_data["step"],
                agg_data["mean"] - agg_data["std"],
                agg_data["mean"] + agg_data["std"],
                color=color,
                alpha=0.2,
            )
        c_idx += 1

    if not plotted:
        ax.text(0.5, 0.5, "No data", horizontalalignment="center", verticalalignment="center")

    if vline_x is not None:
        ax.axvline(x=vline_x, color="red", linewidth=2, linestyle="--")

    if title:
        ax.set_title(title)
    if show_xlabel:
        ax.set_xlabel("Step")
    ax.set_xscale("log")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    if twin_ax is not None and twin_metrics is not None:
        for m in twin_metrics:
            m_data = df_plot[df_plot["metric_name"] == m]
            if m_data.empty:
                continue
            agg_data = m_data.groupby("step")["value"].agg(["mean", "std"]).reset_index()
            label_name = get_polished_name(m)
            p = twin_ax.plot(
                agg_data["step"],
                agg_data["mean"],
                label=label_name,
                color=colors[c_idx % len(colors)],
                linestyle="--",
            )
            color = p[0].get_color()
            if not agg_data["std"].isna().all():
                twin_ax.fill_between(
                    agg_data["step"],
                    agg_data["mean"] - agg_data["std"],
                    agg_data["mean"] + agg_data["std"],
                    color=color,
                    alpha=0.2,
                )
            c_idx += 1
        twin_ax.set_ylabel("RNC1")

    # Combine legends if twin_ax exists
    if show_legend:
        if twin_ax is not None:
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = twin_ax.get_legend_handles_labels()
            ax.legend(
                lines + lines2, labels + labels2, title="Metric", fontsize="small", loc="best"
            )
        else:
            ax.legend(title="Metric", fontsize="small", loc="best")


def plot_general_metrics_row(
    axes_row,
    model_df,
    all_metrics,
    show_col_titles=True,
    show_xlabel=True,
    row_label=None,
):
    train_acc_data = model_df[model_df["metric_name"] == "eval/train/accuracy"]
    first_step_100 = None
    if not train_acc_data.empty:
        steps_100 = train_acc_data[train_acc_data["value"] >= 0.999]["step"]
        if not steps_100.empty:
            first_step_100 = steps_100.min()

    # Subplot (a) Train and Test Accuracy
    acc_metrics = [
        m
        for m in all_metrics
        if "accuracy" in m.lower() and ("train" in m.lower() or "test" in m.lower())
    ]
    if not acc_metrics:
        acc_metrics = [m for m in all_metrics if "acc" in m.lower()]

    title_a = "(a) Train and Test Accuracy" if show_col_titles else None
    plot_aggregated(
        axes_row[0],
        model_df,
        acc_metrics,
        title_a,
        "Accuracy",
        vline_x=first_step_100,
        show_xlabel=show_xlabel,
    )
    axes_row[0].set_ylim(0, 1)

    if row_label:
        axes_row[0].annotate(
            row_label,
            xy=(-0.30, 0.5),
            xycoords="axes fraction",
            fontsize=13,
            fontweight="bold",
            ha="center",
            va="center",
            rotation=90,
        )

    # Subplot (b) Train and Test Loss
    loss_function = "cross_entropy"
    if "params.loss_function" in model_df.columns:
        loss_function = str(model_df["params.loss_function"].iloc[0]).lower()
    elif "run_name" in model_df.columns:
        rname = str(model_df["run_name"].iloc[0]).lower()
        if "mse" in rname:
            loss_function = "mse"

    if "mse" in loss_function:
        loss_metrics = [
            m for m in all_metrics if "loss/mse/mean" in m and ("train" in m or "test" in m)
        ]
    else:
        loss_metrics = [
            m
            for m in all_metrics
            if "loss/cross_entropy/mean" in m and ("train" in m or "test" in m)
        ]

    title_b = "(b) Train and Test Loss" if show_col_titles else None
    plot_aggregated(
        axes_row[1],
        model_df,
        loss_metrics,
        title_b,
        "Loss",
        vline_x=first_step_100,
        show_xlabel=show_xlabel,
    )
    axes_row[1].set_yscale("log")

    # Subplot (c) Weight Norm
    wn_metrics = []
    if "eval/weight_norm/total" in all_metrics:
        wn_metrics.append("eval/weight_norm/total")

    group_metrics = model_df["metric_name"].unique()
    model_name = "unknown"
    if "params.model_name" in model_df.columns:
        model_name = str(model_df["params.model_name"].iloc[0]).lower()
    elif "run_name" in model_df.columns:
        rname = str(model_df["run_name"].iloc[0]).lower()
        if "mlp" in rname:
            model_name = "mlp"
        elif "vit" in rname:
            model_name = "vit_torchvision"
        elif "transformer" in rname:
            model_name = "modular_transformer"

    if model_name == "mlp":
        target_suffix = "/fc3.weight"
    elif model_name == "resnet_torchvision":
        target_suffix = ".fc.weight"
    elif model_name == "vit_torchvision":
        target_suffix = "/vit.heads.head.weight"
    elif model_name == "modular_transformer":
        target_suffix = "/head.weight"
    else:
        target_suffix = None

    last_layer = None
    if target_suffix:
        for m in group_metrics:
            if m.startswith("eval/weight_norm/") and m.endswith(target_suffix):
                last_layer = m
                break

    if last_layer:
        wn_metrics.append(last_layer)

    title_c = "(c) Weight Norm" if show_col_titles else None
    plot_aggregated(
        axes_row[2],
        model_df,
        wn_metrics,
        title_c,
        "Weight Norm",
        vline_x=first_step_100,
        show_xlabel=show_xlabel,
    )


def plot_nc_metrics_row(
    axes_row,
    model_df,
    all_metrics,
    show_col_titles=True,
    show_xlabel=True,
    row_label=None,
):
    train_acc_data = model_df[model_df["metric_name"] == "eval/train/accuracy"]
    first_step_100 = None
    if not train_acc_data.empty:
        steps_100 = train_acc_data[train_acc_data["value"] >= 0.999]["step"]
        if not steps_100.empty:
            first_step_100 = steps_100.min()

    # Subplot (a) NC1 on the left axis and RNC1 on the right axis
    nc1_metrics = [m for m in all_metrics if m == "eval/nc/nc1"]
    rnc1_metrics = [m for m in all_metrics if m == "eval/nc/rnc1/train"]

    ax_a = axes_row[0]
    ax_a_twin = ax_a.twinx() if rnc1_metrics else None
    title_a = "(a) NC1 & RNC1" if show_col_titles else None
    plot_aggregated(
        ax_a,
        model_df,
        nc1_metrics,
        title_a,
        "NC1",
        twin_ax=ax_a_twin,
        twin_metrics=rnc1_metrics,
        vline_x=first_step_100,
        show_xlabel=show_xlabel,
    )

    if row_label:
        ax_a.annotate(
            row_label,
            xy=(-0.30, 0.5),
            xycoords="axes fraction",
            fontsize=13,
            fontweight="bold",
            ha="center",
            va="center",
            rotation=90,
        )

    # Subplot (b) NC2 (Features)
    nc2_metrics_noweights = [
        m
        for m in all_metrics
        if m
        in (
            "eval/nc/nc2_equinorm",
            "eval/nc/nc2_equiangularity",
            "eval/nc/nc2_maximal_angle_equiangularity",
        )
    ]
    ax_b = axes_row[1]
    title_b = "(b) NC2 (Features)" if show_col_titles else None
    plot_aggregated(
        ax_b,
        model_df,
        nc2_metrics_noweights,
        title_b,
        "NC2",
        vline_x=first_step_100,
        show_xlabel=show_xlabel,
    )

    # Subplot (c) NC2 (Weights Only)
    nc2_metrics_weights = [
        m
        for m in all_metrics
        if m
        in (
            "eval/nc/nc2_equinorm_weights",
            "eval/nc/nc2_equiangularity_weights",
            "eval/nc/nc2_maximal_angle_equiangularity_weights",
        )
    ]
    ax_c = axes_row[2]
    title_c = "(c) NC2 (Weights)" if show_col_titles else None
    plot_aggregated(
        ax_c,
        model_df,
        nc2_metrics_weights,
        title_c,
        "NC2 Weights",
        vline_x=first_step_100,
        show_xlabel=show_xlabel,
    )

    # Subplot (d) NC3 and NC4
    nc3_metrics = [m for m in all_metrics if m == "eval/nc/nc3"]
    nc4_metrics = [m for m in all_metrics if m == "eval/nc/nc4"]
    ax_d = axes_row[3]
    ax_d_twin = ax_d.twinx() if nc4_metrics else None
    title_d = "(d) NC3 & NC4" if show_col_titles else None
    plot_aggregated(
        ax_d,
        model_df,
        nc3_metrics,
        title_d,
        "NC3",
        twin_ax=ax_d_twin,
        twin_metrics=nc4_metrics,
        vline_x=first_step_100,
        show_xlabel=show_xlabel,
    )
    if ax_d_twin:
        ax_d_twin.set_ylabel("NC4")


GROUP_CONFIGS = [
    {
        "id": "GROK_MNIST",
        "title": "GROK MNIST",
        "models": [
            ("GROK_MNIST_MSE_MLP", "(I) MLP (MSE)"),
            ("GROK_MNIST_CE_MLP", "(II) MLP (CE)"),
            ("GROK_MNIST_CE_VIT", "(III) ViT (CE)"),
        ],
    },
    {
        "id": "MADD",
        "title": "MADD",
        "models": [
            ("GROK_MADD_CE_TRANSFORMER", "(I) Transformer (CE)"),
            ("GROK_MADD_MSE_TRANSFORMER", "(II) Transformer (MSE)"),
        ],
    },
    {
        "id": "NO_GROK_MNIST",
        "title": "NOGROK MNIST",
        "models": [
            ("NO_GROK_MNIST_MSE_MLP", "(I) MLP (MSE)"),
            ("NO_GROK_MNIST_CE_MLP", "(II) MLP (CE)"),
            ("NO_GROK_MNIST_CE_VIT", "(III) ViT (CE)"),
        ],
    },
]


def extract_model_df(df, run_name):
    if "run_name" in df.columns:
        match = df[df["run_name"] == run_name]
        if not match.empty:
            return match

    r_lower = run_name.lower()
    sub = df.copy()
    if "mse" in r_lower and "params.loss_function" in sub.columns:
        sub = sub[sub["params.loss_function"] == "mse"]
    elif "ce" in r_lower and "params.loss_function" in sub.columns:
        sub = sub[sub["params.loss_function"] == "cross_entropy"]

    if "mlp" in r_lower and "params.model_name" in sub.columns:
        sub = sub[sub["params.model_name"] == "mlp"]
    elif "vit" in r_lower and "params.model_name" in sub.columns:
        sub = sub[sub["params.model_name"] == "vit_torchvision"]
    elif "transformer" in r_lower and "params.model_name" in sub.columns:
        sub = sub[sub["params.model_name"] == "modular_transformer"]

    if ("no_grok" in r_lower or "nogrok" in r_lower) and "params.train_size" in sub.columns:
        sub = sub[sub["params.train_size"] == "25000"]
    elif "grok" in r_lower and "mnist" in r_lower and "params.train_size" in sub.columns:
        sub = sub[sub["params.train_size"] == "1000"]

    return sub


def backfill_initial_nans(df: pd.DataFrame) -> pd.DataFrame:
    """Detects and backfills initial NaNs (e.g. step 0 for zero-initialized weights in ViT).

    Prints a clear, visible notice describing which values are backfilled and why.
    """
    nan_mask = df["value"].isna()
    if not nan_mask.any():
        return df

    model_col = (
        "params.model_name"
        if "params.model_name" in df.columns
        else ("run_name" if "run_name" in df.columns else None)
    )
    group_keys = [c for c in ["metric_name", model_col, "step"] if c and c in df.columns]
    nan_summary = df[nan_mask].groupby(group_keys).size().reset_index(name="count")

    print("\n" + "=" * 80)
    print("[NOTICE] NaN values detected and backfilled with next valid evaluation step:")
    for _, row in nan_summary.iterrows():
        m_name = row["metric_name"]
        m_model = row[model_col] if model_col and model_col in row else "all"
        m_step = row.get("step", "?")
        cnt = row["count"]
        print(
            f"  * Metric: '{m_name}' | Model: '{m_model}' | Step: {m_step} ({cnt} rows backfilled)"
        )
    print("  Reason: Zero-initialized classifier weights in ViT yield 0/0 (NaN) at Step 0.")
    print("=" * 80 + "\n")

    sort_cols = [c for c in ["run_id", "run_name", "metric_name", "step"] if c in df.columns]
    bfill_groups = [c for c in ["run_id", "run_name", "metric_name"] if c in df.columns]
    if not bfill_groups:
        bfill_groups = ["metric_name"]
    df = df.sort_values(sort_cols)
    df["value"] = df.groupby(bfill_groups)["value"].bfill()
    return df


def plot_groups(df, all_metrics, output_dir):
    for grp in GROUP_CONFIGS:
        grp_id = grp["id"]
        models = grp["models"]
        n_rows = len(models)
        print(f"Generating group plot for '{grp['title']}' ({n_rows} models)...")

        # 1. Group General Metrics Plot (n_rows x 3)
        fig_gm, axes_gm = plt.subplots(n_rows, 3, figsize=(18, 4.5 * n_rows))
        if n_rows == 1:
            axes_gm = axes_gm.reshape(1, 3)

        for row_idx, (rname, rlabel) in enumerate(models):
            m_df = extract_model_df(df, rname)
            plot_general_metrics_row(
                axes_gm[row_idx],
                m_df,
                all_metrics,
                show_col_titles=(row_idx == 0),
                show_xlabel=(row_idx == n_rows - 1),
                row_label=rlabel,
            )

        plt.tight_layout()
        out_gm = os.path.join(output_dir, f"{grp_id}_general_metrics.pdf")
        plt.savefig(out_gm, bbox_inches="tight")
        plt.close(fig_gm)

        # 2. Group NC Metrics Plot (n_rows x 4)
        fig_nc, axes_nc = plt.subplots(n_rows, 4, figsize=(24, 4.5 * n_rows))
        if n_rows == 1:
            axes_nc = axes_nc.reshape(1, 4)

        for row_idx, (rname, rlabel) in enumerate(models):
            m_df = extract_model_df(df, rname)
            plot_nc_metrics_row(
                axes_nc[row_idx],
                m_df,
                all_metrics,
                show_col_titles=(row_idx == 0),
                show_xlabel=(row_idx == n_rows - 1),
                row_label=rlabel,
            )

        plt.tight_layout()
        out_nc = os.path.join(output_dir, f"{grp_id}_nc_metrics.pdf")
        plt.savefig(out_nc, bbox_inches="tight")
        plt.close(fig_nc)

    print(f"All group plots saved to '{output_dir}'.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate plots for reproduction-nc-grokking experiment"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="cache/reproduction-nc-grokking-v1_mlflow_export.parquet",
        help="Path to parquet export",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="plots/reproduction_nc_grokking-v1",
        help="Directory to save the generated PDF plots",
    )
    parser.add_argument(
        "--same-resolution",
        "--uniform-resolution",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Filter datapoints so every graph has the same resolution "
            "(((step < 50 and step % 10 == 0) or (step < log_frequency and step % 100 == 0) "
            "or (step % log_frequency == 0)). Enabled by default."
        ),
    )
    parser.add_argument(
        "--log-frequency",
        type=parse_frequency,
        default=1000,
        help=(
            "Log frequency (e.g. 1000 or 1k) used when --same-resolution is enabled "
            "(default: 1000)."
        ),
    )
    parser.add_argument(
        "--groups-only",
        action="store_true",
        help="Only generate grouped grid plots, skipping individual model plots.",
    )
    parser.add_argument(
        "--individual-only",
        action="store_true",
        help="Only generate individual model plots, skipping grouped plots.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Loading data from {args.input}...")
    df = pd.read_parquet(args.input)

    if args.same_resolution:
        log_frequency = args.log_frequency
        initial_len = len(df)
        mask = (
            ((df["step"] < 50) & (df["step"] % 10 == 0))
            | ((df["step"] < log_frequency) & (df["step"] % 100 == 0))
            | (df["step"] % log_frequency == 0)
        )
        df = df[mask].copy()
        print(
            f"Applied same-resolution filter (log_frequency={log_frequency}): "
            f"retained {len(df)}/{initial_len} rows ({len(df['step'].unique())} unique steps)."
        )

    df = backfill_initial_nans(df)

    all_metrics = df["metric_name"].unique()

    if not args.groups_only:
        if "run_name" in df.columns:
            group_cols = ["run_name"]
        else:
            param_cols = [c for c in df.columns if c.startswith("params.")]
            group_cols = [c for c in param_cols if not c.endswith(".seed")]
            if not group_cols:
                df["model_group"] = "All"
                group_cols = ["model_group"]

        df[group_cols] = df[group_cols].astype(str)
        grouped = df.groupby(group_cols)
        print(f"Found {len(grouped)} unique model configurations.")

        for name, group_df in grouped:
            if "run_name" in group_cols:
                group_name = str(name[0]) if isinstance(name, tuple) else str(name)
                group_name = group_name.replace("/", "-").replace(":", "-").replace(" ", "_")
            else:
                group_name = get_model_name(group_df.iloc[0], group_cols)
                if len(group_name) > 150:
                    group_name = group_name[:140] + f"_{abs(hash(group_name)) % 100000}"

            print(f"Processing individual model {group_name}...")

            # 1. General Metrics Plot (1 x 3)
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            plot_general_metrics_row(axes, group_df, all_metrics, show_col_titles=True)
            plt.tight_layout()
            plt.savefig(
                os.path.join(args.output_dir, f"{group_name}_general_metrics.pdf"),
                bbox_inches="tight",
            )
            plt.close(fig)

            # 2. Neural Collapse Metrics Plot (2 x 2)
            fig, axes = plt.subplots(2, 2, figsize=(12, 7))
            plot_nc_metrics_row(axes.flatten(), group_df, all_metrics, show_col_titles=True)
            plt.tight_layout()
            plt.savefig(
                os.path.join(args.output_dir, f"{group_name}_nc_metrics.pdf"), bbox_inches="tight"
            )
            plt.close(fig)

        print(f"All individual plots saved to '{args.output_dir}'.")

    if not args.individual_only:
        plot_groups(df, all_metrics, args.output_dir)


if __name__ == "__main__":
    main()
