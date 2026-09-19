"""Artifact path registry.

Central definitions for every generated artifact in this project.
Submodules split paths by storage location:

- ``remote``: MLflow ``runs:/`` URIs and ``artifact_path`` segments
- ``local``: Local filesystem paths under ``cache/``, ``plots/``, ``results/``
"""
