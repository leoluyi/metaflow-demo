import pandas as pd
from joblib import load
from metaflow import FlowSpec, Parameter, step


class PredictionFlow(FlowSpec):
    env = Parameter("env", help="Environment", default="dev")

    @step
    def start(self):
        from ml_pipeline.config.config_parser import Config

        self.config = Config(self.env)
        self.next(self.load_data)

    @step
    def load_data(self):
        self.data = pd.read_csv(self.config.paths["data.new"])
        self.model = load(self.config.paths["model.latest"])
        self.next(self.preprocess)

    @step
    def preprocess(self):
        processor = load(self.config.paths["model.preprocessor"])
        self.processed_data = processor.transform(self.data)
        self.next(self.predict)

    @step
    def predict(self):
        self.predictions = self.model.predict(self.processed_data)
        self.probabilities = self.model.predict_proba(self.processed_data)
        self.next(self.save_results)

    @step
    def save_results(self):
        results = pd.DataFrame(
            {"prediction": self.predictions, "probability": self.probabilities[:, 1]}
        )
        results.to_csv(self.config.paths["data.predicted"], index=False)
        self.next(self.end)

    @step
    def end(self):
        print(f"Predictions saved to {self.config.paths['data.predicted']}")


if __name__ == "__main__":
    PredictionFlow()
