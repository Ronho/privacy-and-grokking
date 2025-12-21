import numpy as np
import random
import torch

from privacy_and_grokking.config import TrainingRegistry, TrainConfig
from privacy_and_grokking.models import MLP, CNN
from pydantic import BaseModel

class ModelConfig(BaseModel):
    id: str
    config: TrainConfig
    updates: list[int]

TrainingRegistry.load_defaults()

SEED = 64
DTYPE = torch.float32
NUM_CLASSES = 10

RUN_ID = "v2.0.0"
STEP = 100_000
MODELS = [
    ModelConfig(id=RUN_ID, config=TrainingRegistry.get("MLP_V1"), updates=[100_000, 250_000]),
    ModelConfig(id=RUN_ID, config=TrainingRegistry.get("MLP_CAN_NOISE_V1"), updates=[100_000, 250_000]),
    ModelConfig(id=RUN_ID, config=TrainingRegistry.get("MLP_GROK_V1"), updates=[100, 5_000, 100_000, 250_000]),
    ModelConfig(id=RUN_ID, config=TrainingRegistry.get("MLP_GROK_CAN_NOISE_V1"), updates=[100, 5_000, 100_000, 250_000]),

    ModelConfig(id=RUN_ID, config=TrainingRegistry.get("CNN_V1"), updates=[100_000, 250_000]),
    ModelConfig(id=RUN_ID, config=TrainingRegistry.get("CNN_CAN_NOISE_V1"), updates=[100_000, 250_000]),
    ModelConfig(id=RUN_ID, config=TrainingRegistry.get("CNN_GROK_V1"), updates=[1_000, 30_000, 100_000, 250_000]),
    ModelConfig(id=RUN_ID, config=TrainingRegistry.get("CNN_GROK_CAN_NOISE_V1"), updates=[1_000, 30_000, 100_000, 250_000]),
]

torch.set_default_dtype(DTYPE)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
np.random.seed(SEED)
