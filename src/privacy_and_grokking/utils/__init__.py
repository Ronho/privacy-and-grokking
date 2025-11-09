import contextlib
import torch

def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

@contextlib.contextmanager
def eval_mode(model):
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        model.train()

__all__ = ["get_device", "eval_mode"]
