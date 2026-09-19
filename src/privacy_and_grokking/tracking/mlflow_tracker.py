"""MLflow implementation of RunTracker."""

import json
import logging
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from privacy_and_grokking.artifacts.local import get_cache_dir
from privacy_and_grokking.tracking.base import RunTracker


class MLflowTracker(RunTracker):
    """Wrapper around MLflow for tracking metrics and artifacts."""

    def __init__(self, run_id: str, local_base_path: str | Path | None = None):
        super().__init__(run_id)
        self.client = MlflowClient()
        self.local_base_path = Path(local_base_path) if local_base_path else None
        
        # MLflow batches up to 1000 metrics per request
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

        # Prepare batch for MLflow
        # Note: mlflow_metrics must be instances of mlflow.entities.Metric
        metrics_to_log = []
        import time
        now = int(time.time() * 1000)
        for m in self._metrics_buffer:
            metrics_to_log.append(
                mlflow.entities.Metric(
                    key=m["key"],
                    value=m["value"],
                    timestamp=now,
                    step=m.get("step", 0)
                )
            )
            
        buffer_copy = list(metrics_to_log)
        self._metrics_buffer.clear()
        
        def _write():
            # Handle max batch size of 1000
            for i in range(0, len(buffer_copy), 1000):
                batch = buffer_copy[i:i + 1000]
                self.client.log_batch(self.run_id, metrics=batch)

        self._executor.submit(_write)

    def _write_status(self, step: int, total_steps: int, loss: float, steps_per_sec: float) -> None:
        status = {
            "step": step,
            "total_steps": total_steps,
            "loss": loss,
            "steps_per_sec": steps_per_sec
        }
        
        def _write():
            logger = self.get_logger("tracker")
            logger.info(
                f"[Run {self.run_id}] Step {step}/{total_steps} | "
                f"Loss: {loss:.4f} | {steps_per_sec:.1f} steps/s"
            )
            # Log status.json as artifact to MLflow root
            with tempfile.NamedTemporaryFile('w', delete=False, suffix='.json') as tf:
                json.dump(status, tf)
                tf_path = tf.name
            
            try:
                self.client.log_artifact(self.run_id, tf_path)
            finally:
                Path(tf_path).unlink()

        self._executor.submit(_write)

    def log_artifact(self, local_path: str | Path, artifact_path: str | None = None, block: bool = False) -> None:
        def _log():
            self._log_artifact_sync(local_path, artifact_path)
            
        if block:
            _log()
        else:
            self._executor.submit(_log)

    def _log_artifact_sync(self, local_path: str | Path, artifact_path: str | None = None) -> None:
        if Path(local_path).is_file():
            self.client.log_artifact(self.run_id, str(local_path), artifact_path)
        else:
            self.client.log_artifacts(self.run_id, str(local_path), artifact_path)

    def log_dict(self, dictionary: dict[str, Any], artifact_file: str, block: bool = False) -> None:
        def _log():
            with tempfile.NamedTemporaryFile('w', delete=False, suffix='.json') as tf:
                json.dump(dictionary, tf, indent=2)
                tf_path = tf.name
                
            try:
                # artifact_file might contain directories (e.g. "path/to/my.json")
                path_obj = Path(artifact_file)
                artifact_path = str(path_obj.parent) if str(path_obj.parent) != "." else None
                
                # We need to rename the temp file temporarily so MLflow keeps its name
                temp_dir = Path(tf_path).parent
                renamed_path = temp_dir / path_obj.name
                Path(tf_path).rename(renamed_path)
                
                self.client.log_artifact(self.run_id, str(renamed_path), artifact_path)
            finally:
                if renamed_path.exists():
                    renamed_path.unlink()
                if Path(tf_path).exists():
                    Path(tf_path).unlink()
                    
        if block:
            _log()
        else:
            self._executor.submit(_log)

    def get_local_artifact_path(self, run_id: str, artifact_path: str, experiment_id: str | None = None) -> Path:
        """
        Quasi-local loading: Try resolving locally first via self.local_base_path.
        If missing or not configured, use MLflow API download.
        Automatically unzips if a .zip archive is found.
        """
        def _unzip_if_needed(target_path: Path) -> Path:
            target_zip = target_path.with_suffix('.zip')
            if target_zip.exists() and not target_path.exists():
                cache_target = get_cache_dir() / "downloaded_artifacts" / run_id / artifact_path
                if not cache_target.exists():
                    cache_target.mkdir(parents=True, exist_ok=True)
                    with zipfile.ZipFile(target_zip, 'r') as zf:
                        zf.extractall(cache_target)
                return cache_target
            return target_path

        # 1. Try quasi-local globbing
        if self.local_base_path:
            # Layout: <base_path>/<exp_id>/<run_id>/artifacts/<artifact_path>
            if experiment_id:
                direct_path = self.local_base_path / experiment_id / run_id / "artifacts" / artifact_path
                if direct_path.exists() or direct_path.with_suffix('.zip').exists():
                    return _unzip_if_needed(direct_path)
            else:
                # Glob for experiment_id
                matches = list(self.local_base_path.glob(f"*/{run_id}/artifacts/{artifact_path}*"))
                if matches:
                    # Filter exact match or exact zip match
                    exact = [m for m in matches if m.name == Path(artifact_path).name or m.name == Path(artifact_path).name + '.zip']
                    if exact:
                        found = exact[0]
                        if found.suffix == '.zip':
                            return _unzip_if_needed(found.with_suffix(''))
                        return found
                
                # We failed, warn and fallback
                logger = self.get_logger("tracker")
                logger.warning(
                    f"Could not quickly glob '{artifact_path}' for run '{run_id}' in {self.local_base_path}. "
                    "Falling back to MLflow API. Providing 'experiment_id' avoids globbing."
                )

        # 2. Fallback: MLflow download API
        # We try to download the .zip first if it exists? No, we don't know if it's zipped.
        # But we can try to list artifacts or just let download_artifacts handle it.
        # If the user logged it as archive, the remote path is artifact_path + ".zip"
        # We can try to download the dir, if it fails, try the zip.
        cache_dir = get_cache_dir() / "downloaded_artifacts" / run_id
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            downloaded = mlflow.artifacts.download_artifacts(
                artifact_uri=f"runs:/{run_id}/{artifact_path}",
                dst_path=str(cache_dir)
            )
            return Path(downloaded)
        except Exception:
            # Maybe it's zipped?
            try:
                zip_uri = f"runs:/{run_id}/{artifact_path}.zip"
                downloaded_zip = mlflow.artifacts.download_artifacts(
                    artifact_uri=zip_uri,
                    dst_path=str(cache_dir)
                )
                
                # Unzip it
                extracted_dir = cache_dir / artifact_path
                if not extracted_dir.exists():
                    extracted_dir.mkdir(parents=True, exist_ok=True)
                    with zipfile.ZipFile(downloaded_zip, 'r') as zf:
                        zf.extractall(extracted_dir)
                return extracted_dir
            except Exception as e2:
                raise FileNotFoundError(f"Failed to download {artifact_path} or {artifact_path}.zip from MLflow. {e2}")

    def load_dict(self, run_id: str, artifact_path: str) -> dict[str, Any]:
        local_path = self.get_local_artifact_path(run_id, artifact_path)
        with open(local_path, 'r') as f:
            return json.load(f)
