from datetime import datetime
from typing import Dict

import mlflow


def log_model_metrics(metrics: Dict[str, float], model_name: str):
    timestamp = datetime.now().isoformat()
    with mlflow.start_run(run_name=f"metrics_{timestamp}"):
        mlflow.log_metrics(metrics)
        mlflow.set_tag("model_name", model_name)
