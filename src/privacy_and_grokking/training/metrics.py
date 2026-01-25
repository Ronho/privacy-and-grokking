from pydantic import BaseModel


class ModeMetrics(BaseModel):
    loss: float
    accuracy: float


class Metrics(BaseModel):
    step: int
    train: ModeMetrics
    test: ModeMetrics
    norm: float
    last_layer_norm: float
