from .model import TrainConfig, MSELoss, AdamW, DatasetConfig, GaussianNoiseCanary
from ..utils import get_package_version


def get_configs() -> list[TrainConfig]:
    VERSION = get_package_version()
    BATCH_SIZE = 200
    LOG_FREQUENCY = 500
    OPTIMIZATION_STEPS = 250_000
    SEED=128
    LOSS = MSELoss()
    OPTIMIZER = AdamW(learning_rate=1e-3, weight_decay=0.01)
    MNIST_TRAIN_RATIO = 0.5
    GAUSSIAN_NOISE_CANARY_SEED = 64

    configs = []

    configs.append(TrainConfig(
        name="MLP_V1",
        code_version=VERSION,
        batch_size=BATCH_SIZE,
        log_frequency=LOG_FREQUENCY,
        optimization_steps=OPTIMIZATION_STEPS,
        seed=SEED,
        loss=LOSS,
        optimizer=OPTIMIZER,
        model="mlp",
        dataset=DatasetConfig(
            name="mnist",
            train_ratio=MNIST_TRAIN_RATIO,
            train_size=None,
            canary=None,
        ),
        initialization_scale=None,
    ))
    configs.append(TrainConfig(
        name="MLP_CAN_NOISE_V1",
        code_version=VERSION,
        batch_size=BATCH_SIZE,
        log_frequency=LOG_FREQUENCY,
        optimization_steps=OPTIMIZATION_STEPS,
        seed=SEED,
        loss=LOSS,
        optimizer=OPTIMIZER,
        model="mlp",
        dataset=DatasetConfig(
            name="mnist",
            train_ratio=0.5,
            train_size=None,
            canary=GaussianNoiseCanary(
                noise_scale=1.0,
                seed=GAUSSIAN_NOISE_CANARY_SEED,
            ),
        ),
        initialization_scale=None,
    ))
    configs.append(TrainConfig(
        name="MLP_GROK_V1",
        code_version=VERSION,
        batch_size=BATCH_SIZE,
        log_frequency=LOG_FREQUENCY,
        optimization_steps=OPTIMIZATION_STEPS,
        seed=SEED,
        loss=LOSS,
        optimizer=OPTIMIZER,
        model="mlp",
        dataset=DatasetConfig(
            name="mnist",
            train_ratio=MNIST_TRAIN_RATIO,
            train_size=1000,
            canary=None,
        ),
        initialization_scale=8.0,
    ))
    configs.append(TrainConfig(
        name="MLP_GROK_CAN_NOISE_V1",
        code_version=VERSION,
        batch_size=BATCH_SIZE,
        log_frequency=LOG_FREQUENCY,
        optimization_steps=OPTIMIZATION_STEPS,
        seed=SEED,
        loss=LOSS,
        optimizer=OPTIMIZER,
        model="mlp",
        dataset=DatasetConfig(
            name="mnist",
            train_ratio=MNIST_TRAIN_RATIO,
            train_size=1000,
            canary=GaussianNoiseCanary(
                noise_scale=1.0,
                seed=GAUSSIAN_NOISE_CANARY_SEED,
            ),
        ),
        initialization_scale=8.0,
    ))

    # CNN:
    configs.append(TrainConfig(
        name="CNN_V1",
        code_version=VERSION,
        batch_size=BATCH_SIZE,
        log_frequency=LOG_FREQUENCY,
        optimization_steps=OPTIMIZATION_STEPS,
        seed=SEED,
        loss=LOSS,
        optimizer=OPTIMIZER,
        model="cnn",
        dataset=DatasetConfig(
            name="mnist",
            train_ratio=MNIST_TRAIN_RATIO,
            train_size=None,
            canary=None,
        ),
        initialization_scale=None,
    ))
    configs.append(TrainConfig(
        name="CNN_CAN_NOISE_V1",
        code_version=VERSION,
        batch_size=BATCH_SIZE,
        log_frequency=LOG_FREQUENCY,
        optimization_steps=OPTIMIZATION_STEPS,
        seed=SEED,
        loss=LOSS,
        optimizer=OPTIMIZER,
        model="cnn",
        dataset=DatasetConfig(
            name="mnist",
            train_ratio=0.5,
            train_size=None,
            canary=GaussianNoiseCanary(
                noise_scale=1.0,
                seed=GAUSSIAN_NOISE_CANARY_SEED,
            ),
        ),
        initialization_scale=None,
    ))
    configs.append(TrainConfig(
        name="CNN_GROK_V1",
        code_version=VERSION,
        batch_size=BATCH_SIZE,
        log_frequency=LOG_FREQUENCY,
        optimization_steps=OPTIMIZATION_STEPS,
        seed=SEED,
        loss=LOSS,
        optimizer=OPTIMIZER,
        model="cnn",
        dataset=DatasetConfig(
            name="mnist",
            train_ratio=MNIST_TRAIN_RATIO,
            train_size=1000,
            canary=None,
        ),
        initialization_scale=8.0,
    ))
    configs.append(TrainConfig(
        name="CNN_GROK_CAN_NOISE_V1",
        code_version=VERSION,
        batch_size=BATCH_SIZE,
        log_frequency=LOG_FREQUENCY,
        optimization_steps=OPTIMIZATION_STEPS,
        seed=SEED,
        loss=LOSS,
        optimizer=OPTIMIZER,
        model="cnn",
        dataset=DatasetConfig(
            name="mnist",
            train_ratio=MNIST_TRAIN_RATIO,
            train_size=1000,
            canary=GaussianNoiseCanary(
                noise_scale=1.0,
                seed=GAUSSIAN_NOISE_CANARY_SEED,
            ),
        ),
        initialization_scale=8.0,
    ))

    return configs
