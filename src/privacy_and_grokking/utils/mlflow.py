from pathlib import Path

import mlflow


def setup_mlflow(exp_name: str = "default"):
    root = Path(__file__).parent.parent.parent.parent.resolve()
    tracking_dir = root / "data"
    mlflow.set_tracking_uri(f"sqlite:///{tracking_dir / 'mlflow.db'}")
    if not mlflow.get_experiment_by_name(exp_name):
        mlflow.create_experiment(
            name=exp_name,
            artifact_location=str(tracking_dir / "mlruns"),
        )
    mlflow.set_experiment(exp_name)
