import os

import mlflow
import requests

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5050")


def setup_mlflow(exp_name: str = "default"):
    os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"

    try:
        response = requests.get(TRACKING_URI, timeout=5)
        response.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            f"MLflow tracking server is not reachable at {TRACKING_URI}. Start the server and retry."
        ) from e
    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            f"MLflow tracking server at {TRACKING_URI} returned an unexpected response: {e}"
        ) from e

    mlflow.set_tracking_uri(TRACKING_URI)
    if not mlflow.get_experiment_by_name(exp_name):
        mlflow.create_experiment(name=exp_name)
    mlflow.set_experiment(exp_name)
