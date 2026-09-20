import hashlib
from pathlib import Path

# --- Common Configurations ---
AVAILABLE_GPUS = []
LOAD_ALL_TO_GPU = True

# --- Shared Utility Functions ---

def get_deterministic_seed(*args, salt):
    """Generates a deterministic integer seed from arguments to ensure idempotency."""
    s = str(salt) + "_" + "_".join(str(a) for a in args)
    return int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % 1000000

def config_priority(config_path: Path):
    """
    Returns a priority tuple for sorting config files.
    Ensures a consistent and deterministic order of configs.
    """
    name = config_path.name.lstrip("+_")

    a = 1 if "NO_GROK" in name else 0
    b = 0 if "MNIST" in name else 1
    c = 0 if "MLP" in name else 1
    d = 0 if "CE" in name else 1
    
    return (a, b, c, d)

def get_steps(config_path: Path):
    """Returns the default number of training steps based on the config name."""
    name = config_path.name
    if "MADD" in name:
        return 150_000
    if "MNIST" in name:
        return 150_000
    return 150_000
