import numpy as np
import random
import torch

from privacy_and_grokking.models import MLP, CNN
from pydantic import BaseModel

class ModelConfig(BaseModel):
    id: str
    name: str
    model_class: type
    updates: list[int]

SEED = 64
DTYPE = torch.float32
NUM_CLASSES = 10

RUN_ID = "v1.2.4"
STEP = 100_000
MODELS = [
    ModelConfig(id=RUN_ID, name="MLP_GROK_V1", model_class=MLP, updates=[100, 5000, 100000]),
    ModelConfig(id=RUN_ID, name="MLP_V1", model_class=MLP, updates=[100000]),
    ModelConfig(id=RUN_ID, name="CNN_GROK_V1", model_class=CNN, updates=[1000, 30000, 100000]),
    ModelConfig(id=RUN_ID, name="CNN_V1", model_class=CNN, updates=[100000]),
]

torch.set_default_dtype(DTYPE)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
np.random.seed(SEED)
