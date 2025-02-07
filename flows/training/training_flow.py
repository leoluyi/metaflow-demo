# flows/train_flow.py
import mlflow
import pandas as pd
from metaflow import FlowSpec, Parameter, current, step
from sklearn.model_selection import train_test_split

from src.features.processors import FeatureProcessor
from src.models.model import Model
from src.utils.config import load_config
from src.utils.metrics import calculate_classification_metrics


class TrainingFlow(FlowSpec):
    experiment = Parameter(
        name="experiment", help="MLflow experiment name", default="default"
    )

    @step
    def start(self):
        """Initialize MLflow experiment"""
        self.configs = {
            "paths": load_config("config/paths.ini"),
            "model": load_config("config/model.ini"),
            "features": load_config("config/features.ini"),
        }
        mlflow.set_tracking_uri("mlruns")
        mlflow.set_experiment(self.experiment)
        self.next(self.load_data)

    @step
    def load_data(self):
        """Load and validate data"""
        self.df = pd.read_csv(self.configs["paths"]["train"])
        self.train_df, self.test_df = train_test_split(self.df, test_size=0.2)
        self.next(self.process_features)

    @step
    def process_features(self):
        """Process and engineer features"""
        self.processor = FeatureProcessor()
        features = self.configs["features"]["columns"].split(",")
        target = self.configs["features"]["target"]

        self.X_train = self.train_df[features]
        self.X_test = self.test_df[features]
        self.y_train = self.train_df[target]
        self.y_test = self.test_df[target]
        self.next(self.train_model)

    @step
    def train_model(self):
        """Train model and log to MLflow"""
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(self.configs["model"])

            # Train model
            self.model = Model(**self.configs["model"])
            self.model.fit(self.X_train, self.y_train)

            # Generate predictions
            train_pred = self.model.predict(self.X_train)
            test_pred = self.model.predict(self.X_test)
            train_prob = self.model.predict_proba(self.X_train)[:, 1]
            test_prob = self.model.predict_proba(self.X_test)[:, 1]

            # Calculate metrics
            train_metrics = calculate_classification_metrics(
                self.y_train, train_pred, train_prob
            )
            test_metrics = calculate_classification_metrics(
                self.y_test, test_pred, test_prob
            )

            # Log metrics
            for name, value in train_metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"train_{name}", value)

            for name, value in test_metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"test_{name}", value)

            # Log model
            mlflow.sklearn.log_model(self.model, "model")

            # Save run ID
            self.run_id = mlflow.active_run().info.run_id

        self.next(self.end)

    @step
    def end(self):
        """Save results"""
        print(f"Training completed. MLflow run ID: {self.run_id}")


if __name__ == "__main__":
    TrainingFlow()
