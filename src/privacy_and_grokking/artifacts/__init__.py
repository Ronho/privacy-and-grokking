"""Artifact path registry.

Central definitions for every generated artifact in this project.
Submodules split paths by storage location:

- ``remote``: MLflow ``runs:/`` URIs and ``artifact_path`` segments
- ``local``: Local filesystem paths under ``cache/``, ``plots/``, ``results/``
- ``cache``: Quasi-local artifact retrieval and disk caching
"""

from privacy_and_grokking.artifacts.cache import (
    get_artifact_bytes,
    get_artifact_json,
    get_artifact_path,
    get_training_config,
)

__all__ = [
    "get_artifact_bytes",
    "get_artifact_json",
    "get_artifact_path",
    "get_training_config",
]
