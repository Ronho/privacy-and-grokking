from privacy_and_grokking.config.model import AdamW, MSELoss, TrainConfig
from privacy_and_grokking.datasets import (
    DatasetConfig,
    Datasets,
    MaskingConfig,
    Maskings,
    SquareWatermarkCanaryConfig,
)
from privacy_and_grokking.utils import get_git_commit_id, get_package_version

VERSION = get_package_version()
COMMIT_ID = get_git_commit_id()
BATCH_SIZE = 200
LOG_FREQUENCY = 500
OPTIMIZATION_STEPS = 250_000
SEED = 128
LOSS = MSELoss()
OPTIMIZER = AdamW(learning_rate=1e-3, weight_decay=0.01)


def get_configs() -> list[TrainConfig]:
    mask = MaskingConfig(name=Maskings.INDEPENDENT_STRATIFIED, num_models=256, p=0.5, seed=1)
    configs = []

    configs.append(
        TrainConfig(
            name="MNIST_MLP_NOGROK_NOCAN_SMALL_DATASET",
            code_version=VERSION,
            commit_id=COMMIT_ID,
            batch_size=BATCH_SIZE,
            log_frequency=LOG_FREQUENCY,
            optimization_steps=OPTIMIZATION_STEPS,
            seed=SEED,
            loss=LOSS,
            optimizer=OPTIMIZER,
            model="mlp",
            dataset=DatasetConfig(
                name=Datasets.MNIST, train_size=1_000, canary_share=0, canary_config=None, seed=1
            ),
            dataset_mask=mask,
            initialization_scale=None,
        )
    )
    configs.append(
        TrainConfig(
            name="MNIST_MLP_NOGROK_NOCAN",
            code_version=VERSION,
            commit_id=COMMIT_ID,
            batch_size=BATCH_SIZE,
            log_frequency=LOG_FREQUENCY,
            optimization_steps=OPTIMIZATION_STEPS,
            seed=SEED,
            loss=LOSS,
            optimizer=OPTIMIZER,
            model="mlp",
            dataset=DatasetConfig(
                name=Datasets.MNIST, train_size=None, canary_share=0, canary_config=None, seed=1
            ),
            dataset_mask=mask,
            initialization_scale=None,
        )
    )
    configs.append(
        TrainConfig(
            name="MNIST_MLP_GROK_TRAIN_NOCAN",
            code_version=VERSION,
            commit_id=COMMIT_ID,
            batch_size=BATCH_SIZE,
            log_frequency=LOG_FREQUENCY,
            optimization_steps=OPTIMIZATION_STEPS,
            seed=SEED,
            loss=LOSS,
            optimizer=OPTIMIZER,
            model="mlp",
            dataset=DatasetConfig(
                name=Datasets.MNIST, train_size=1_000, canary_share=0, canary_config=None, seed=1
            ),
            dataset_mask=mask,
            initialization_scale=8.0,
        )
    )
    configs.append(
        TrainConfig(
            name="MNIST_MLP_NOGROK_TRAIN_WATERMARK",
            code_version=VERSION,
            commit_id=COMMIT_ID,
            batch_size=BATCH_SIZE,
            log_frequency=LOG_FREQUENCY,
            optimization_steps=OPTIMIZATION_STEPS,
            seed=SEED,
            loss=LOSS,
            optimizer=OPTIMIZER,
            model="mlp",
            dataset=DatasetConfig(
                name=Datasets.MNIST,
                train_size=None,
                canary_share=0.01,
                canary_config=SquareWatermarkCanaryConfig(square_size=3),
                seed=1,
            ),
            dataset_mask=mask,
            initialization_scale=None,
        )
    )
    configs.append(
        TrainConfig(
            name="MNIST_MLP_GROK_TRAIN_WATERMARK",
            code_version=VERSION,
            commit_id=COMMIT_ID,
            batch_size=BATCH_SIZE,
            log_frequency=LOG_FREQUENCY,
            optimization_steps=OPTIMIZATION_STEPS,
            seed=SEED,
            loss=LOSS,
            optimizer=OPTIMIZER,
            model="mlp",
            dataset=DatasetConfig(
                name=Datasets.MNIST,
                train_size=1_000,
                canary_share=0.01,
                canary_config=SquareWatermarkCanaryConfig(square_size=3),
                seed=1,
            ),
            dataset_mask=mask,
            initialization_scale=8.0,
        )
    )

    return configs
