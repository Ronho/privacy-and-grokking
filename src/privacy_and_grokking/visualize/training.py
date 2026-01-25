import json
import matplotlib.pyplot as plt
import polars as pl

from typing import Any

from ..config import TrainConfig
from ..datasets import get_dataset
from ..path_keeper import get_path_keeper

def _flatten_dict(d: dict[str, Any], parent_key: str = "", sep: str = "_") -> dict[str, Any]:
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def _aggregate_logits_per_step(df: pl.DataFrame, num_classes: int) -> pl.DataFrame:
    labels = list(range(num_classes))

    correct_logits_exprs = [
        pl.when(pl.col("correct_label") == label).then(pl.col(f"logit_{label}"))
        for label in labels
    ]

    df_with_logits = df.with_columns([
        pl.coalesce(*correct_logits_exprs).alias("correct_logit"),
        pl.sum_horizontal([f"logit_{i}" for i in labels]).alias("sum_all_logits")
    ]).with_columns([
        ((pl.col("sum_all_logits") - pl.col("correct_logit")) / (num_classes - 1)).alias("avg_wrong_logit")
    ])

    logits_by_step = (
        df_with_logits
        .group_by("step")
        .agg([
            pl.col("correct_logit").mean().alias("avg_correct_logit"),
            pl.col("correct_logit").std().alias("std_correct_logit"),
            pl.col("avg_wrong_logit").mean().alias("avg_wrong_logit"),
            pl.col("avg_wrong_logit").std().alias("std_wrong_logit"),
            pl.col("correct_logit").count().alias("count")
        ])
        .sort("step")
    )

    return logits_by_step

def _process_single_step_logits(df: pl.DataFrame) -> pl.DataFrame:
    labels = df["correct_label"].unique().sort().to_list() 
    expressions = [
        pl.when(pl.col("correct_label") == label).then(pl.col(f"logit_{label}"))
        for label in labels
    ]
    return df.with_columns(
        pl.coalesce(*expressions).alias("correct_logit")
    )

def visualize(cfg: TrainConfig):
    pk = get_path_keeper()
    
    metrics = json.loads(pk.TRAIN_METRICS.read_text())
    flat_metrics = [_flatten_dict(m) for m in metrics]
    df = pl.DataFrame(flat_metrics)

    # Accuracy, Weight Norm, Loss
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.set_xlabel("Optimization Steps")
    ax1.set_xscale("log")
    ax1.set_ylabel("Accuracy")
    ax1.plot(df["step"], df["train_accuracy"], label="Train Accuracy")
    ax1.plot(df["step"], df["test_accuracy"], label="Test Accuracy")
    ax1.legend(loc=(0.15, 0.7))
    ax1.grid(True, alpha=0.3)
    
    ax1_twin = ax1.twinx()
    ax1_twin.set_ylabel("Weight Norm")
    ax1_twin.plot(df["step"], df["norm"], color="purple", label="Weight Norm")
    ax1_twin.plot(df["step"], df["last_layer_norm"], color="pink", label="Last Layer Weight Norm")
    ax1_twin.legend(loc=(0.05, 0.55))
    
    ax1.set_title("Accuracy & Weight Norm")

    ax2.set_xlabel("Optimization Steps")
    ax2.set_xscale("log")
    ax2.set_ylabel("Loss")
    ax2.plot(df["step"], df["train_loss"], label="Train")
    ax2.plot(df["step"], df["test_loss"], label="Test")
    ax2.legend(loc="upper right")
    ax2.set_title("Loss")
    ax2.grid(True, alpha=0.3)

    fig.text(0.02, 0.01, f"Model: {cfg.name}", fontsize=9, alpha=0.6)
    fig.tight_layout()
    fig.savefig(pk.IMAGE_FOLDER / "training_accuracy_weightnorm_loss.png")
    plt.close(fig)

    # Logits Analysis
    pk.set_params({"step": "*"})

    df_train = pl.read_parquet(pk.TRAIN_LOGITS, glob=True)
    df_test = pl.read_parquet(pk.TEST_LOGITS, glob=True)

    _, _, _, _, num_classes, _ = get_dataset(
        name=cfg.dataset.name,
        train_ratio=cfg.dataset.train_ratio,
        train_size=None,
        canary=None,
    )

    train_logits = _aggregate_logits_per_step(df_train, num_classes)
    test_logits = _aggregate_logits_per_step(df_test, num_classes)

    # Train/Test Logit Size Evolution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    steps_train = train_logits["step"].to_numpy()
    avg_correct_train = train_logits["avg_correct_logit"].to_numpy()
    std_correct_train = train_logits["std_correct_logit"].to_numpy()
    avg_wrong_train = train_logits["avg_wrong_logit"].to_numpy()
    std_wrong_train = train_logits["std_wrong_logit"].to_numpy()

    ax1.plot(steps_train, avg_correct_train, "b-", linewidth=2, label="Average Correct Logit")
    ax1.fill_between(steps_train, avg_correct_train - std_correct_train, avg_correct_train + std_correct_train, 
                    alpha=0.3, color="blue", label="Correct Logit ±1 Std")
    ax1.plot(steps_train, avg_wrong_train, "r-", linewidth=2, label="Average Wrong Logits")
    ax1.fill_between(steps_train, avg_wrong_train - std_wrong_train, avg_wrong_train + std_wrong_train, 
                    alpha=0.3, color="red", label="Wrong Logits ±1 Std")
    
    ax1.set_xlabel("Optimization Steps")
    ax1.set_ylabel("Logit Size")
    ax1.set_title("Training Data")
    ax1.set_xscale("log")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    steps_test = test_logits["step"].to_numpy()
    avg_correct_test = test_logits["avg_correct_logit"].to_numpy()
    std_correct_test = test_logits["std_correct_logit"].to_numpy()
    avg_wrong_test = test_logits["avg_wrong_logit"].to_numpy()
    std_wrong_test = test_logits["std_wrong_logit"].to_numpy()
    
    ax2.plot(steps_test, avg_correct_test, "b-", linewidth=2, label="Average Correct Logit")
    ax2.fill_between(steps_test, avg_correct_test - std_correct_test, avg_correct_test + std_correct_test, 
                    alpha=0.3, color="blue", label="Correct Logit ±1 Std")
    ax2.plot(steps_test, avg_wrong_test, "r-", linewidth=2, label="Average Wrong Logits")
    ax2.fill_between(steps_test, avg_wrong_test - std_wrong_test, avg_wrong_test + std_wrong_test, 
                    alpha=0.3, color="red", label="Wrong Logits ±1 Std")
    
    ax2.set_xlabel("Optimization Steps")
    ax2.set_ylabel("Logit Size")
    ax2.set_title("Test Data")
    ax2.set_xscale("log")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    fig.text(0.02, 0.01, f"Model: {cfg.name}", fontsize=9, alpha=0.6)
    fig.tight_layout()
    fig.savefig(pk.IMAGE_FOLDER / "training_train_test_logits_size_evolution.png")
    plt.close(fig)

    # Correct/Wrong Logit Size Evolution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    steps = train_logits["step"].to_numpy()

    avg_correct_train = train_logits["avg_correct_logit"].to_numpy()
    std_correct_train = train_logits["std_correct_logit"].to_numpy()
    avg_correct_test = test_logits["avg_correct_logit"].to_numpy()
    std_correct_test = test_logits["std_correct_logit"].to_numpy()

    ax1.plot(steps, avg_correct_train, "b-", linewidth=2, label="Train Average")
    ax1.fill_between(steps, avg_correct_train - std_correct_train, avg_correct_train + std_correct_train, 
                    alpha=0.3, color="blue", label="Train ±1 Std")
    ax1.plot(steps, avg_correct_test, "r-", linewidth=2, label="Test Average")
    ax1.fill_between(steps, avg_correct_test - std_correct_test, avg_correct_test + std_correct_test, 
                    alpha=0.3, color="red", label="Test ±1 Std")
    
    ax1.set_xlabel("Optimization Steps")
    ax1.set_ylabel("Logit Size")
    ax1.set_title("Correct Logits")
    ax1.set_xscale("log")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    avg_wrong_train = train_logits["avg_wrong_logit"].to_numpy()
    std_wrong_train = train_logits["std_wrong_logit"].to_numpy()
    avg_wrong_test = test_logits["avg_wrong_logit"].to_numpy()
    std_wrong_test = test_logits["std_wrong_logit"].to_numpy()
    
    ax2.plot(steps, avg_wrong_train, "b-", linewidth=2, label="Train Average")
    ax2.fill_between(steps, avg_wrong_train - std_wrong_train, avg_wrong_train + std_wrong_train, 
                    alpha=0.3, color="blue", label="Train ±1 Std")
    ax2.plot(steps, avg_wrong_test, "r-", linewidth=2, label="Test Average")
    ax2.fill_between(steps, avg_wrong_test - std_wrong_test, avg_wrong_test + std_wrong_test, 
                    alpha=0.3, color="red", label="Test ±1 Std")
    
    ax2.set_xlabel("Optimization Steps")
    ax2.set_ylabel("Logit Size")
    ax2.set_title("Wrong Logits")
    ax2.set_xscale("log")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.text(0.02, 0.01, f"Model: {cfg.name}", fontsize=9, alpha=0.6)
    fig.tight_layout()
    fig.savefig(pk.IMAGE_FOLDER / "training_correct_wrong_logits_size_evolution.png")
    plt.close(fig)

    # Single Step Logit Distribution
    pk.set_params({"step": cfg.optimization_steps})
    df_train = pl.read_parquet(pk.TRAIN_LOGITS)
    df_test = pl.read_parquet(pk.TEST_LOGITS)
    train_logits, test_logits = _process_single_step_logits(df_train), _process_single_step_logits(df_test)

    plt.figure(figsize=(10, 6))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(train_logits["correct_logit"].to_numpy(), bins=50, alpha=0.6, label="Train", density=True)
    ax.hist(test_logits["correct_logit"].to_numpy(), bins=50, alpha=0.6, label="Test", density=True)
    ax.set_xlabel("Correct Logit")
    ax.set_ylabel("Density")
    ax.set_yscale("log")
    ax.axvline(x=train_logits["correct_logit"].mean(), color="blue", linestyle="--", label="Train Mean")
    ax.axvline(x=test_logits["correct_logit"].mean(), color="orange", linestyle="--", label="Test Mean")
    ax.axvline(x=train_logits["correct_logit"].median(), color="blue", linestyle=":", label="Train Median")
    ax.axvline(x=test_logits["correct_logit"].median(), color="orange", linestyle=":", label="Test Median")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.text(0.02, 0.01, f"Model: {cfg.name}", fontsize=9, alpha=0.6)
    fig.tight_layout()
    fig.savefig(pk.IMAGE_FOLDER / f"training_correct_logit_distribution_step_{cfg.optimization_steps}.png")
    plt.close(fig)
