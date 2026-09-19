# Tracking System (`RunTracker`)

The `privacy_and_grokking.tracking` module provides a backwards-compatible, robust abstraction over MLflow and local file logging.

## Core Concepts

### 1. The `RunTracker` Interface
The `RunTracker` is the abstract base class that defines how metrics, status updates, and artifacts are logged. It uses a **Context Manager** to ensure that all background I/O threads are properly flushed and closed when the run finishes.

```python
from privacy_and_grokking.tracking import get_tracker

with get_tracker("run-id", offline=True) as tracker:
    tracker.log_metric("loss", 0.5)
    tracker.log_artifact("model.pth", "checkpoints/model.pth")
```

### 2. Implementations
- **`OfflineLocalTracker`**: Bypasses MLflow entirely. Metrics are appended to a `metrics.jsonl` file in a local cache directory (`cache/offline_runs/<run_id>`). This is extremely fast and robust against crashes.
- **`MLflowTracker`**: Wraps the standard `mlflow` API. It batches metric uploads to prevent API bottlenecks.

### 3. Quasi-Local Artifact Resolution
To avoid the overhead of downloading artifacts via the MLflow API when they are already available on a shared network drive (e.g., in a cluster environment), the `MLflowTracker` implements a "Quasi-Local" approach.

If `local_base_path` is provided (or `MLFLOW_LOCAL_ARTIFACTS_PATH` is set), `tracker.get_local_artifact_path()` will attempt to resolve the artifact directly using the filesystem (`glob`). If it fails, it falls back to the MLflow API.

### 4. Transparent Archiving
To prevent thousands of small files (like checkpoints or plots) from overwhelming the filesystem or MLflow, the tracker supports zipping directories in the background via `log_artifact_archive`. 
When reading these artifacts via `get_local_artifact_path`, the tracker automatically unzips the archive into a cache directory and returns the path to the extracted folder. The consumer code never has to know it was archived.

### 5. Non-blocking I/O
All heavy I/O operations (logging artifacts, writing status JSONs) are dispatched to a background `ThreadPoolExecutor` to ensure the training loop is never blocked by disk or network latency.
