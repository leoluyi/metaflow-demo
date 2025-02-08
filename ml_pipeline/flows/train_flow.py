import logging

import mlflow
import pandas as pd
from joblib import dump
from metaflow import FlowSpec, Parameter, step
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# print(f"__package__: {__package__}")
# print(f"__name__: {__name__}")
# print(f"__file__: {__file__}")

__package__ = __package__ or "ml_pipeline.flows"

from ..models.architectures.random_forest import RFModel
from ..models.components.feature_processor import FeatureProcessor

logger = logging.getLogger(__name__ or "ml_pipeline")


class TrainingFlow(FlowSpec):
    env = Parameter("env", help="Environment", default="dev")
    experiment_name = Parameter(
        "experiment_name",
        help="MLflow experiment name",
        default="default_experiment",
    )

    @step
    def start(self):
        from ..config.config_parser import Config

        self.config = Config(self.env)
        self.next(self.prepare_data)

    @step
    def prepare_data(self):
        df = pd.read_csv(self.config.paths["data.train"])
        X = df.drop("target", axis=1)
        y = df["target"]

        self.processor = FeatureProcessor()
        X = self.processor.fit_transform(X)

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Save processor for prediction
        dump(self.processor, self.config.paths["model.preprocessor"])
        self.next(self.train_model)

    @step
    def train_model(self):
        # Set up MLflow tracking
        mlflow.set_tracking_uri(self.config.paths["mlflow.tracking_uri"])

        # Create or get experiment
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(self.experiment_name)
            else:
                experiment_id = experiment.experiment_id
        except Exception as e:
            logger.warn(f"Error setting up MLflow experiment: {e}")
            experiment_id = mlflow.create_experiment(self.experiment_name)

        mlflow.set_experiment(experiment_id)

        with mlflow.start_run():
            model = RFModel(n_estimators=100, random_state=42)
            model.train(self.X_train, self.y_train)

            y_pred = model.predict(self.X_val)
            metrics = {
                "accuracy": accuracy_score(self.y_val, y_pred),
                "f1": f1_score(self.y_val, y_pred),
            }

            mlflow.log_metrics(metrics)

            # Log model path as an artifact
            model_path = self.config.paths["model.latest"]
            mlflow.log_param("model_path", model_path)

            # Save model and print information
            print("\nModel Training Summary:")
            print("-" * 40)
            print("Validation Metrics:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  F1 Score: {metrics['f1']:.4f}")
            print(f"\nSaving model to: {model_path}")

            dump(model, model_path)
            print("Model saved successfully!")

        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    TrainingFlow()
