import matplotlib.pyplot as plt
import polars as pl

SETUPS = {
    "GROK_MNIST_MSE_MLP": "(I) MLP (MSE) Grokking",
    "NO_GROK_MNIST_MSE_MLP": "(II) MLP (MSE) Baseline",
    "GROK_MNIST_CE_MLP": "(III) MLP (CE) Grokking",
    "NO_GROK_MNIST_CE_MLP": "(IV) MLP (CE) Baseline",
    "GROK_MNIST_MSE_VIT": "(V) ViT (MSE) Grokking",
    "NO_GROK_MNIST_MSE_VIT": "(VI) ViT (MSE) Baseline",
    "GROK_MNIST_CE_VIT": "(VII) ViT (CE) Grokking",
    "NO_GROK_MNIST_CE_VIT": "(VIII) ViT (CE) Baseline",
    "GROK_MADD_CE_TRANSFORMER": "(IX) MADD (CE) Grokking",
}
BATCHES = [list(SETUPS.keys())[:4], list(SETUPS.keys())[4:]]

def setup():
    plt.style.use('./style.mplstyle')

_pattern = "(" + "|".join([key for key in SETUPS.keys()]) + ")"

def preprocess_info_rmia(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(pl.col("run_name").str.extract(_pattern, group_index=1).replace(SETUPS).alias("title"))
    return df

def preprocess_metrics_df(df: pl.DataFrame, cols: list[str] = []) -> pl.DataFrame:
    df = df.pivot(index=["run_id", "run_name", "step", "timestamp", *cols], on="metric_name", values="value", aggregate_function="first").sort(["run_id", "step"])
    df = df.with_columns(pl.col("run_name").str.extract(_pattern, group_index=1).replace(SETUPS).alias("title"), pl.from_epoch(pl.col("timestamp"), time_unit="ms").alias("timestamp"))
    return df

