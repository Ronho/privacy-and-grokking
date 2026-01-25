import contextlib
from importlib.metadata import version

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


def get_package_version() -> str:
    """Get the version of the privacy_and_grokking package."""
    try:
        return version("privacy-and-grokking")
    except Exception:
        return "unknown"


def set_all_seeds(seed: int) -> None:
    """Set the seed for all relevant random number generators."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


__all__ = ["get_device", "eval_mode"]
