
from functools import partial
from .metrics import Metrics
from .parameters import AdamW, MSELoss, ModelArchitectures, Parameters
from .registry import TrainingRegistry
from .train import train

_BATCH_SIZE = 200
_LOG_FREQUENCY = 500
_OPTIMIZATION_STEPS = 100_000

TrainingRegistry.register("MLP_V1", partial(train, params=Parameters(
    name="MLP_V1",
    batch_size=_BATCH_SIZE,
    initialization_scale=None,
    log_frequency=_LOG_FREQUENCY,
    loss=MSELoss(),
    model=ModelArchitectures.MLP,
    optimization_steps=_OPTIMIZATION_STEPS,
    optimizer=AdamW(
        learning_rate=1e-3,
        weight_decay=0.01,
    ),
    sample_size=None,
    seed=64,
)))

TrainingRegistry.register("MLP_GROK_V1", partial(train, params=Parameters(
    name="MLP_GROK_V1",
    batch_size=_BATCH_SIZE,
    initialization_scale=8.0,
    log_frequency=_LOG_FREQUENCY,
    loss=MSELoss(),
    model=ModelArchitectures.MLP,
    optimization_steps=_OPTIMIZATION_STEPS,
    optimizer=AdamW(
        learning_rate=1e-3,
        weight_decay=0.01,
    ),
    sample_size=1_000,
    seed=64,
)))

TrainingRegistry.register("CNN_V1", partial(train, params=Parameters(
    name="CNN_V1",
    batch_size=_BATCH_SIZE,
    initialization_scale=None,
    log_frequency=_LOG_FREQUENCY,
    loss=MSELoss(),
    model=ModelArchitectures.CNN,
    optimization_steps=_OPTIMIZATION_STEPS,
    optimizer=AdamW(
        learning_rate=1e-3,
        weight_decay=0.01,
    ),
    sample_size=None,
    seed=64,
)))

TrainingRegistry.register("CNN_GROK_V1", partial(train, params=Parameters(
    name="CNN_GROK_V1",
    batch_size=_BATCH_SIZE,
    initialization_scale=8.0,
    log_frequency=_LOG_FREQUENCY,
    loss=MSELoss(),
    model=ModelArchitectures.CNN,
    optimization_steps=_OPTIMIZATION_STEPS,
    optimizer=AdamW(
        learning_rate=1e-3,
        weight_decay=0.01,
    ),
    sample_size=1_000,
    seed=64,
)))

__all__ = ["TrainingRegistry", "Metrics"]
