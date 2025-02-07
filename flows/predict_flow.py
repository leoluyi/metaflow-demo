# flows/predict_flow.py
import pandas as pd
from metaflow import FlowSpec, step

from src.utils.config import load_config


class PredictionFlow(FlowSpec):
    @step
    def start(self):
        """Load config and data"""
        self.configs = {
            "paths": load_config("config/paths.ini"),
            "features": load_config("config/features.ini"),
        }
        self.df = pd.read_csv(self.configs["paths"]["data"]["predict"])
        self.next(self.predict)

    @step
    def predict(self):
        """Generate predictions"""
        import mlflow

        features = self.configs["features"]["features"]

        # Load production model from MLflow
        model = mlflow.sklearn.load_model("mlruns/models/model")
        self.predictions = model.predict(self.df[features])
        self.probabilities = model.predict_proba(self.df[features])
        self.next(self.end)

    @step
    def end(self):
        """Save predictions"""
        results = pd.DataFrame(
            {
                "prediction": self.predictions,
                "probability_0": self.probabilities[:, 0],
                "probability_1": self.probabilities[:, 1],
            }
        )
        results.to_csv(self.configs["paths"]["predictions"], index=False)


if __name__ == "__main__":
    PredictionFlow()
