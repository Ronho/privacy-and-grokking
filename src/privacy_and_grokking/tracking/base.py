"""Base interface for the tracking abstraction layer."""

import abc
import concurrent.futures
import json
import time
import zipfile
from pathlib import Path
from typing import Any

from privacy_and_grokking.utils.logger import Logger


class RunTracker(abc.ABC):
    """Backwards-compatible abstraction for logging and loading artifacts."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._metrics_buffer: list[dict[str, Any]] = []
        
        # Status monitoring config
        self._last_status_time = time.time()
        self._last_status_step = -1
        self.status_update_steps = 100
        self.status_update_seconds = 30.0

    def __enter__(self) -> "RunTracker":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # --- Metrics ---

    @abc.abstractmethod
    def log_metric(self, key: str, value: float, step: int | None = None, flush: bool = False) -> None:
        """Logs a single metric."""
        pass

    @abc.abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int | None = None, flush: bool = False) -> None:
        """Logs multiple metrics."""
        pass

    @abc.abstractmethod
    def flush(self) -> None:
        """Forces all buffered metrics to be written."""
        pass

    # --- Text Logs & Status ---

    def get_logger(self, name: str | None = None) -> Logger:
        """
        Returns a configured instance of the custom Logger.
        """
        return Logger.get(name)

    def update_status(self, step: int, total_steps: int, loss: float, steps_per_sec: float) -> None:
        """
        Updates a live-status for monitoring (e.g. status.json).
        Only writes if configured thresholds (steps or seconds) are met.
        """
        current_time = time.time()
        time_elapsed = current_time - self._last_status_time
        steps_elapsed = step - self._last_status_step
        
        if (time_elapsed >= self.status_update_seconds) or (steps_elapsed >= self.status_update_steps):
            self._write_status(step, total_steps, loss, steps_per_sec)
            self._last_status_time = current_time
            self._last_status_step = step

    @abc.abstractmethod
    def _write_status(self, step: int, total_steps: int, loss: float, steps_per_sec: float) -> None:
        """Actual implementation to write the status (local or remote)."""
        pass

    # --- Artifacts ---

    @abc.abstractmethod
    def log_artifact(self, local_path: str | Path, artifact_path: str | None = None, block: bool = False) -> None:
        """Logs a local file or directory as an artifact."""
        pass

    def log_artifact_archive(self, local_dir: str | Path, artifact_name: str, block: bool = False) -> None:
        """
        Zips a directory in the background and logs it as a single file artifact.
        
        Args:
            local_dir: The directory to zip.
            artifact_name: The target filename (e.g., 'checkpoints/1000.zip').
            block: If True, blocks until complete.
        """
        def _zip_and_log():
            import tempfile
            local_path = Path(local_dir)
            with tempfile.TemporaryDirectory() as tmpdir:
                zip_path = Path(tmpdir) / Path(artifact_name).name
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for file_path in local_path.rglob('*'):
                        if file_path.is_file():
                            zf.write(file_path, file_path.relative_to(local_path))
                
                # Use synchronous underlying log
                self._log_artifact_sync(zip_path, str(Path(artifact_name).parent))

        if block:
            _zip_and_log()
        else:
            self._executor.submit(_zip_and_log)

    @abc.abstractmethod
    def _log_artifact_sync(self, local_path: str | Path, artifact_path: str | None = None) -> None:
        """Synchronous implementation of artifact logging."""
        pass

    @abc.abstractmethod
    def log_dict(self, dictionary: dict[str, Any], artifact_file: str, block: bool = False) -> None:
        """Logs a python dictionary as a JSON artifact."""
        pass

    # --- Loading ---

    @abc.abstractmethod
    def get_local_artifact_path(self, run_id: str, artifact_path: str, experiment_id: str | None = None) -> Path:
        """
        Resolves an artifact to a local filesystem path via globbing.
        Automatically unzips archives if they exist (e.g. searching for checkpoints/1000
        will unzip checkpoints/1000.zip into cache if needed).
        """
        pass

    @abc.abstractmethod
    def load_dict(self, run_id: str, artifact_path: str) -> dict[str, Any]:
        """Loads a JSON artifact as a dictionary."""
        pass

    def close(self) -> None:
        """Flushes buffers and waits for IO threads to finish."""
        self.flush()
        self._executor.shutdown(wait=True)
