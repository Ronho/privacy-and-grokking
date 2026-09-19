"""Offline implementation of RunTracker using local filesystem."""

import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from privacy_and_grokking.artifacts.local import get_cache_dir
from privacy_and_grokking.tracking.base import RunTracker


class OfflineLocalTracker(RunTracker):
    """Logs everything directly to local filesystem, bypassing MLflow."""

    def __init__(self, run_id: str, offline_base_dir: Path | None = None):
        super().__init__(run_id)
        if offline_base_dir is None:
            self.offline_dir = get_cache_dir() / "offline_runs" / run_id
        else:
            self.offline_dir = Path(offline_base_dir) / run_id

        self.offline_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.offline_dir / "metrics.jsonl"
        
        # We can flush metrics based on buffer size
        self.metrics_buffer_size = 1000

    def log_metric(self, key: str, value: float, step: int | None = None, flush: bool = False) -> None:
        self._metrics_buffer.append({"key": key, "value": value, "step": step})
        if flush or len(self._metrics_buffer) >= self.metrics_buffer_size:
            self.flush()

    def log_metrics(self, metrics: dict[str, float], step: int | None = None, flush: bool = False) -> None:
        for k, v in metrics.items():
            self._metrics_buffer.append({"key": k, "value": v, "step": step})
        if flush or len(self._metrics_buffer) >= self.metrics_buffer_size:
            self.flush()

    def flush(self) -> None:
        if not self._metrics_buffer:
            return
            
        def _write_metrics(metrics: list[dict]):
            # Append-only write to JSONL
            with open(self.metrics_file, 'a') as f:
                for m in metrics:
                    f.write(json.dumps(m) + '\n')

        # Create a copy and clear buffer
        buffer_copy = list(self._metrics_buffer)
        self._metrics_buffer.clear()
        
        # Write synchronously or submit to thread?
        # Actually flush should ideally block to guarantee it's written.
        _write_metrics(buffer_copy)

    def _write_status(self, step: int, total_steps: int, loss: float, steps_per_sec: float) -> None:
        status = {
            "step": step,
            "total_steps": total_steps,
            "loss": loss,
            "steps_per_sec": steps_per_sec
        }
        status_file = self.offline_dir / "status.json"
        
        def _write():
            # Atomic write via temp file to avoid partial reads
            with tempfile.NamedTemporaryFile('w', dir=str(self.offline_dir), delete=False) as tf:
                json.dump(status, tf)
                tf_path = tf.name
            Path(tf_path).replace(status_file)
            
            logger = self.get_logger("tracker")
            logger.info(
                f"[Run {self.run_id}] Step {step}/{total_steps} | "
                f"Loss: {loss:.4f} | {steps_per_sec:.1f} steps/s"
            )

        self._executor.submit(_write)

    def log_artifact(self, local_path: str | Path, artifact_path: str | None = None, block: bool = False) -> None:
        def _log():
            self._log_artifact_sync(local_path, artifact_path)
            
        if block:
            _log()
        else:
            self._executor.submit(_log)

    def _log_artifact_sync(self, local_path: str | Path, artifact_path: str | None = None) -> None:
        src = Path(local_path)
        if artifact_path:
            dst = self.offline_dir / artifact_path
            # If artifact_path implies we should keep the filename of src
            if not dst.suffix and src.is_file():
                 dst = dst / src.name
        else:
            dst = self.offline_dir / src.name

        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_file():
            shutil.copy2(src, dst)
        else:
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)

    def log_dict(self, dictionary: dict[str, Any], artifact_file: str, block: bool = False) -> None:
        def _log():
            dst = self.offline_dir / artifact_file
            dst.parent.mkdir(parents=True, exist_ok=True)
            with open(dst, 'w') as f:
                json.dump(dictionary, f, indent=2)
                
        if block:
            _log()
        else:
            self._executor.submit(_log)

    def get_local_artifact_path(self, run_id: str, artifact_path: str, experiment_id: str | None = None) -> Path:
        # For offline tracker, run_id must match our own (or we assume standard offline layout)
        # Note: If reading an older run, we just look in CACHE_DIR / "offline_runs"
        base_dir = get_cache_dir() / "offline_runs" / run_id
        
        target = base_dir / artifact_path
        
        # Check if the requested path is actually inside a zip archive
        # e.g., artifact_path="checkpoints/1000", maybe "checkpoints/1000.zip" exists
        target_zip = target.with_suffix('.zip')
        if target_zip.exists() and not target.exists():
            # We need to unzip it. Let's unzip to downloaded_artifacts to not mutate offline_runs.
            # Usually downloaded_artifacts has <run_id>/<artifact_path>
            cache_target = get_cache_dir() / "downloaded_artifacts" / run_id / artifact_path
            if not cache_target.exists():
                cache_target.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(target_zip, 'r') as zf:
                    zf.extractall(cache_target)
            return cache_target

        if not target.exists():
            raise FileNotFoundError(f"Offline artifact {target} does not exist.")
            
        return target

    def load_dict(self, run_id: str, artifact_path: str) -> dict[str, Any]:
        local_path = self.get_local_artifact_path(run_id, artifact_path)
        with open(local_path, 'r') as f:
            return json.load(f)
