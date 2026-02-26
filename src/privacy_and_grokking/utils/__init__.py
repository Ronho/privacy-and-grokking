import contextlib
from importlib.metadata import version

import torch

from privacy_and_grokking.utils.logger import Logger


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


def get_git_commit_id() -> str:
    """Get the current git commit ID."""
    try:
        import subprocess

        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("ascii")
            .strip()
        )
    except Exception:
        return "unknown"


def get_git_changes() -> dict[str, str]:
    """Get staged and unstaged git changes."""
    try:
        import subprocess

        # Get staged changes (diff between HEAD and index)
        staged = subprocess.check_output(
            ["git", "diff", "--cached", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8")

        # Get unstaged changes (diff between index and working tree)
        unstaged = subprocess.check_output(
            ["git", "diff", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8")

        return {
            "comnmit": get_git_commit_id(),
            "staged": staged if staged else "No staged changes",
            "unstaged": unstaged if unstaged else "No unstaged changes",
        }
    except Exception as e:
        return {
            "staged": f"Error getting staged changes: {e}",
            "unstaged": f"Error getting unstaged changes: {e}",
        }


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


__all__ = [
    "get_device",
    "eval_mode",
    "get_package_version",
    "get_git_commit_id",
    "get_git_changes",
    "set_all_seeds",
    "Logger",
]
