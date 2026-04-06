import os
from pathlib import Path

import mlflow


def setup_mlflow(exp_name: str = "default"):
    os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"
    root = Path(__file__).parent.parent.parent.parent.resolve()
    tracking_dir = root / "data"
    db_path = tracking_dir / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    if not mlflow.get_experiment_by_name(exp_name):
        artifact_location = tracking_dir.joinpath("mlruns").as_uri()
        mlflow.create_experiment(
            name=exp_name,
            artifact_location=artifact_location,
        )
    mlflow.set_experiment(exp_name)
