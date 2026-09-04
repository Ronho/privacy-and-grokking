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
):
    if df_plot.empty:
        ax.text(0.5, 0.5, "No data", horizontalalignment="center", verticalalignment="center")
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

    ax.set_title(title)
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


def main():
    parser = argparse.ArgumentParser(
        description="Generate plots for reproduction-nc-grokking experiment"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="cache/reproduction-nc-grokking_mlflow_export.parquet",
        help="Path to parquet export",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="plots/reproduction_nc_grokking",
        help="Directory to save the generated PDF plots",
    )
    parser.add_argument(
        "--same-resolution",
        "--uniform-resolution",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter datapoints so every graph has the same resolution ((step < 50) or (step < log_frequency and step %% 100 == 0) or (step %% log_frequency == 0)). Enabled by default.",
    )
    parser.add_argument(
        "--log-frequency",
        type=parse_frequency,
        default=1000,
        help="Log frequency (e.g. 1000 or 1k) used when --same-resolution is enabled (default: 1000).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Loading data from {args.input}...")
    df = pd.read_parquet(args.input)

    if args.same_resolution:
        log_frequency = args.log_frequency
        initial_len = len(df)
        mask = (
            (df["step"] < 50)
            | ((df["step"] < log_frequency) & (df["step"] % 100 == 0))
            | (df["step"] % log_frequency == 0)
        )
        df = df[mask].copy()
        print(
            f"Applied same-resolution filter (log_frequency={log_frequency}): retained {len(df)}/{initial_len} rows ({len(df['step'].unique())} unique steps)."
        )

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

    all_metrics = df["metric_name"].unique()

    for name, group_df in grouped:
        if "run_name" in group_cols:
            group_name = str(name[0]) if isinstance(name, tuple) else str(name)
            group_name = group_name.replace("/", "-").replace(":", "-").replace(" ", "_")
        else:
            group_name = get_model_name(group_df.iloc[0], group_cols)
            # Truncate group name to avoid long path errors on Windows
            if len(group_name) > 150:
                group_name = group_name[:140] + f"_{abs(hash(group_name)) % 100000}"

        print(f"Processing {group_name}...")

        train_acc_data = group_df[group_df["metric_name"] == "eval/train/accuracy"]
        first_step_100 = None
        if not train_acc_data.empty:
            steps_100 = train_acc_data[train_acc_data["value"] >= 0.999]["step"]
            if not steps_100.empty:
                first_step_100 = steps_100.min()

        # 1. General Metrics Plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Subplot (a) Train and Test Accuracy
        acc_metrics = [
            m
            for m in all_metrics
            if "accuracy" in m.lower() and ("train" in m.lower() or "test" in m.lower())
        ]
        if not acc_metrics:
            acc_metrics = [m for m in all_metrics if "acc" in m.lower()]
        plot_aggregated(
            axes[0],
            group_df,
            acc_metrics,
            "(a) Train and Test Accuracy",
            "Accuracy",
            vline_x=first_step_100,
        )
        axes[0].set_ylim(0, 1)

        # Subplot (b) Train and Test Loss
        loss_function = "cross_entropy"
        if "params.loss_function" in group_df.columns:
            loss_function = str(group_df["params.loss_function"].iloc[0]).lower()

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

        plot_aggregated(
            axes[1],
            group_df,
            loss_metrics,
            "(b) Train and Test Loss",
            "Loss",
            vline_x=first_step_100,
        )
        axes[1].set_yscale("log")

        # Subplot (c) Weight norm for the last layer and total weight norm
        wn_metrics = []
        if "eval/weight_norm/total" in all_metrics:
            wn_metrics.append("eval/weight_norm/total")

        group_metrics = group_df["metric_name"].unique()
        model_name = "unknown"
        if "params.model_name" in group_df.columns:
            model_name = str(group_df["params.model_name"].iloc[0]).lower()

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

        plot_aggregated(
            axes[2], group_df, wn_metrics, "(c) Weight Norm", "Weight Norm", vline_x=first_step_100
        )

        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f"{group_name}_general_metrics.pdf"))
        plt.close(fig)

        # 2. Neural Collapse Metrics Plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 7))

        # Subplot (a) NC1 on the left axis and RNC1 on the right axis
        nc1_metrics = [m for m in all_metrics if m == "eval/nc/nc1"]
        rnc1_metrics = [m for m in all_metrics if m == "eval/nc/rnc1/train"]

        ax2a = axes[0, 0]
        ax2a_twin = ax2a.twinx() if rnc1_metrics else None
        plot_aggregated(
            ax2a,
            group_df,
            nc1_metrics,
            "(a) NC1 & RNC1",
            "NC1",
            twin_ax=ax2a_twin,
            twin_metrics=rnc1_metrics,
            vline_x=first_step_100,
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
        ax2b = axes[0, 1]
        plot_aggregated(
            ax2b,
            group_df,
            nc2_metrics_noweights,
            "(b) NC2 (Features)",
            "NC2",
            vline_x=first_step_100,
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
        ax2c = axes[1, 0]
        plot_aggregated(
            ax2c,
            group_df,
            nc2_metrics_weights,
            "(c) NC2 (Weights)",
            "NC2 Weights",
            vline_x=first_step_100,
        )

        # Subplot (d) NC3 and NC4
        nc3_metrics = [m for m in all_metrics if m == "eval/nc/nc3"]
        nc4_metrics = [m for m in all_metrics if m == "eval/nc/nc4"]
        ax2d = axes[1, 1]
        ax2d_twin = ax2d.twinx() if nc4_metrics else None
        plot_aggregated(
            ax2d,
            group_df,
            nc3_metrics,
            "(d) NC3 & NC4",
            "NC3",
            twin_ax=ax2d_twin,
            twin_metrics=nc4_metrics,
            vline_x=first_step_100,
        )
        if ax2d_twin:
            ax2d_twin.set_ylabel("NC4")

        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f"{group_name}_nc_metrics.pdf"))
        plt.close(fig)

    print(f"All plots saved to '{args.output_dir}'.")


if __name__ == "__main__":
    main()
