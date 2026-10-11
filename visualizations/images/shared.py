import matplotlib.pyplot as plt
import polars as pl

SETUPS = {
    "GROK_MNIST_MSE_MLP": "(I) MLP (MSE)\nGrokking",
    "NO_GROK_MNIST_MSE_MLP": "(II) MLP (MSE)\nNo Grokking",
    "GROK_MNIST_CE_MLP": "(III) MLP (CE)\nGrokking",
    "NO_GROK_MNIST_CE_MLP": "(IV) MLP (CE)\nNo Grokking",
    "GROK_MNIST_CE_VIT": "(V) ViT (CE)\nGrokking",
    "NO_GROK_MNIST_CE_VIT": "(VI) ViT (CE)\nNo Grokking",
    "GROK_MADD_CE_TRANSFORMER": "(VII) MADD (CE)\nGrokking"
}

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

