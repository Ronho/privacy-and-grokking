from pydantic import BaseModel


class ModeMetrics(BaseModel):
    loss: float
    loss_std: float
    accuracy: float


class Metrics(BaseModel):
    step: int
    train: ModeMetrics
    test: ModeMetrics
    norm: float
    last_layer_norm: float
