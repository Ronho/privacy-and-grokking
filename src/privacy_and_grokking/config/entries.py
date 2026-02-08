from privacy_and_grokking.datasets import DatasetConfig, Datasets, SquareWatermarkCanaryConfig, MaskingConfig, Maskings

from ..utils import get_package_version
from .model import AdamW, MSELoss, TrainConfig


def get_configs() -> list[TrainConfig]:
    VERSION = get_package_version()
    BATCH_SIZE = 200
    LOG_FREQUENCY = 500
    OPTIMIZATION_STEPS = 250_000
    OPTIMIZATION_STEPS_LONG = 1_000_000
    SEED = 128
    LOSS = MSELoss()
    OPTIMIZER = AdamW(learning_rate=1e-3, weight_decay=0.01)
    mask = MaskingConfig(name=Maskings.INDEPENDENT_STRATIFIED, num_models=256, p=0.5, seed=1)

    configs = []

    configs.append(
        TrainConfig(
            name="MNIST_MLP_NOGROK_NOCAN_SMALL_DATASET",
            code_version=VERSION,
            batch_size=BATCH_SIZE,
            log_frequency=LOG_FREQUENCY,
            optimization_steps=OPTIMIZATION_STEPS_LONG,
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
            batch_size=BATCH_SIZE,
            log_frequency=LOG_FREQUENCY,
            optimization_steps=OPTIMIZATION_STEPS_LONG,
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
            batch_size=BATCH_SIZE,
            log_frequency=LOG_FREQUENCY,
            optimization_steps=OPTIMIZATION_STEPS_LONG,
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
