"""Quasi-local MLflow artifact caching.

Provides a unified interface for downloading and caching artifacts from MLflow.
Local files are always preferred: if an artifact exists in ``cache/runs/{run_id}/...``,
it is returned immediately without contacting the MLflow tracking server.
When an artifact must be fetched, it is downloaded and persisted to disk so future
invocations are completely local and offline-friendly.
"""

from __future__ import annotations

import json
import os
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from privacy_and_grokking.artifacts.local import get_cache_dir, local_run_dir

if TYPE_CHECKING:
    from privacy_and_grokking.config import TrainConfig

DEFAULT_TRACKING_URI = "http://localhost:5051"


def get_default_tracking_uri() -> str:
    """Returns the default tracking URI from the environment or fallback."""
    return os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)


def get_artifact_path(
    run_id: str,
    artifact_path: str,
    tracking_uri: str | None = None,
    force_download: bool = False,
    timeout: float = 30.0,
) -> Path:
    """Get the local filesystem path for a run artifact, downloading and caching if needed.

    Args:
        run_id: The MLflow run ID.
        artifact_path: Relative path to the artifact (e.g. ``"training_config.json"``
            or ``"checkpoints/1000/model.pth"``).
        tracking_uri: MLflow tracking server URI. Defaults to ``MLFLOW_TRACKING_URI``
            environment variable or ``http://localhost:5051``.
        force_download: If True, re-download the artifact even if cached locally.
        timeout: Network timeout in seconds for HTTP download requests.

    Returns:
        Path to the locally cached artifact.

    Raises:
        FileNotFoundError: If the artifact cannot be retrieved or found.
    """
    # 1. Check primary cache location: cache/runs/{run_id}/{artifact_path}
    canonical_target = local_run_dir(run_id) / artifact_path
    if not force_download and (canonical_target.is_file() or canonical_target.is_dir()):
        return canonical_target

    # 2. Check legacy cache location: cache/downloaded_artifacts/{run_id}/{artifact_path}
    legacy_target = get_cache_dir() / "downloaded_artifacts" / run_id / artifact_path
    if not force_download and (legacy_target.is_file() or legacy_target.is_dir()):
        return legacy_target

    # 3. Artifact is not cached locally; fetch from MLflow
    uri = (tracking_uri or get_default_tracking_uri()).rstrip("/")
    canonical_target.parent.mkdir(parents=True, exist_ok=True)

    # For remote HTTP tracking servers, prioritize the HTTP /get-artifact endpoint.
    # Remote servers often store absolute cluster paths (e.g. file:///workspace/mlruns/...)
    # in SQLite which causes standard client.download_artifacts to fail locally.
    if uri.startswith("http://") or uri.startswith("https://"):
        download_url = f"{uri}/get-artifact?path={artifact_path}&run_uuid={run_id}"
        try:
            req = urllib.request.Request(
                download_url,
                headers={"User-Agent": "privacy-and-grokking/artifact-cache"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                content = resp.read()

            # Atomic write to canonical cache location
            _atomic_write_bytes(canonical_target, content)
            return canonical_target
        except urllib.error.HTTPError as e:
            # If 404, maybe it was logged as a zip archive?
            if e.code == 404 and not artifact_path.endswith(".zip"):
                zip_path = f"{artifact_path}.zip"
                zip_url = f"{uri}/get-artifact?path={zip_path}&run_uuid={run_id}"
                try:
                    req_zip = urllib.request.Request(
                        zip_url,
                        headers={"User-Agent": "privacy-and-grokking/artifact-cache"},
                    )
                    with urllib.request.urlopen(req_zip, timeout=timeout) as resp_zip:
                        zip_bytes = resp_zip.read()
                    return _extract_zip_bytes(zip_bytes, canonical_target)
                except Exception:
                    pass

    # 4. Fallback: MLflow client download
    try:
        import mlflow

        client = mlflow.tracking.MlflowClient(tracking_uri=uri)
        downloaded = client.download_artifacts(run_id, artifact_path, dst_path=str(canonical_target.parent))
        downloaded_path = Path(downloaded)
        if downloaded_path.is_file() and downloaded_path.suffix == ".zip" and not artifact_path.endswith(".zip"):
            return _extract_zip_file(downloaded_path, canonical_target)
        return downloaded_path
    except Exception as err:
        raise FileNotFoundError(
            f"Could not load or download artifact '{artifact_path}' for run '{run_id}' "
            f"from {uri}: {err}"
        ) from err


def get_artifact_bytes(
    run_id: str,
    artifact_path: str,
    tracking_uri: str | None = None,
    force_download: bool = False,
    timeout: float = 30.0,
) -> bytes:
    """Load raw artifact content as bytes, leveraging the local cache."""
    path = get_artifact_path(
        run_id=run_id,
        artifact_path=artifact_path,
        tracking_uri=tracking_uri,
        force_download=force_download,
        timeout=timeout,
    )
    return path.read_bytes()


def get_artifact_json(
    run_id: str,
    artifact_path: str,
    tracking_uri: str | None = None,
    force_download: bool = False,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Load and parse a JSON artifact, leveraging the local cache."""
    content = get_artifact_bytes(
        run_id=run_id,
        artifact_path=artifact_path,
        tracking_uri=tracking_uri,
        force_download=force_download,
        timeout=timeout,
    )
    return json.loads(content.decode("utf-8"))


def get_training_config(
    run_id: str,
    tracking_uri: str | None = None,
    force_download: bool = False,
    timeout: float = 30.0,
) -> TrainConfig:
    """Load the training configuration for a run as a validated TrainConfig instance."""
    from privacy_and_grokking.config import TrainConfig

    data = get_artifact_json(
        run_id=run_id,
        artifact_path="training_config.json",
        tracking_uri=tracking_uri,
        force_download=force_download,
        timeout=timeout,
    )
    return TrainConfig.model_validate(data)


# -------------------------------------------------------------------
#  Internal helper functions
# -------------------------------------------------------------------

def _atomic_write_bytes(target_file: Path, data: bytes) -> None:
    """Atomically write binary data to a file using a temporary file in the same directory."""
    target_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = target_file.with_name(f".{target_file.name}.tmp.{os.getpid()}")
    try:
        temp_file.write_bytes(data)
        temp_file.replace(target_file)
    except Exception:
        if temp_file.exists():
            temp_file.unlink()
        raise


def _extract_zip_bytes(zip_bytes: bytes, target_dir: Path) -> Path:
    """Extract in-memory zip bytes into target_dir."""
    target_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp.write(zip_bytes)
        tmp_path = Path(tmp.name)
    try:
        with zipfile.ZipFile(tmp_path, "r") as zf:
            zf.extractall(target_dir)
        return target_dir
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _extract_zip_file(zip_path: Path, target_dir: Path) -> Path:
    """Extract a zip file on disk into target_dir."""
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)
    return target_dir
