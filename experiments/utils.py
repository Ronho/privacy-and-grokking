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
    
    # Primary priority
    if "MNIST" in name and "MSE" in name and "MLP" in name:
        p = 0
    elif "MNIST" in name and "CE" in name and "MLP" in name:
        p = 1
    elif "VIT" in name:
        p = 2
    elif "MADD" in name and "CE" in name:
        p = 3
    elif "MADD" in name and "MSE" in name:
        p = 4
    elif "CE" in name:
        p = 5
    elif "MSE" in name:
        p = 6
    else:
        p = 99
        
    # Secondary priority
    sub = 0 if "NO_GROK" in name else 1
    
    return (p, sub)

def get_steps(config_path: Path):
    """Returns the default number of training steps based on the config name."""
    name = config_path.name
    if "MADD" in name:
        return 150_000
    if "MNIST" in name:
        return 150_000
    return 150_000
