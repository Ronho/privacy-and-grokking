"""Factory and interfaces for the unified tracking abstraction."""

import os
from typing import Optional
from pathlib import Path

from privacy_and_grokking.tracking.base import RunTracker
from privacy_and_grokking.tracking.mlflow_tracker import MLflowTracker
from privacy_and_grokking.tracking.offline_tracker import OfflineLocalTracker


def get_tracker(run_id: str, offline: bool = False, local_base_path: Optional[str | Path] = None) -> RunTracker:
    """
    Returns an appropriate RunTracker instance.
    
    Args:
        run_id: The ID of the current run.
        offline: If True, uses the OfflineLocalTracker which bypasses MLflow entirely.
        local_base_path: If using MLflowTracker, this path enables quasi-local reading
            (bypassing the MLflow download API). Also can be set via MLFLOW_LOCAL_ARTIFACTS_PATH.
            
    Returns:
        RunTracker instance (either MLflowTracker or OfflineLocalTracker)
    """
    if offline or os.environ.get("PAG_OFFLINE_MODE", "").lower() in ("true", "1"):
        # For offline tracker, local_base_path is the base cache dir (if provided)
        return OfflineLocalTracker(run_id, offline_base_dir=local_base_path)
    
    # Try to load local base path from env if not provided
    if local_base_path is None:
        local_base_path = os.environ.get("MLFLOW_LOCAL_ARTIFACTS_PATH")
        
    return MLflowTracker(run_id, local_base_path=local_base_path)


__all__ = ["RunTracker", "MLflowTracker", "OfflineLocalTracker", "get_tracker"]
