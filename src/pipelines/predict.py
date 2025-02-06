from pathlib import Path

import joblib
import pandas as pd

from src.utils.config import load_config
from src.utils.logger import setup_logger


def predict(input_path: str, model_path: str, output_path: str):
    logger = setup_logger("predict", "logs/predict.log")

    # Load data and model
    data_config = load_config("configs/base/data.yaml")
    df = pd.read_csv(input_path)
    model = joblib.load(model_path)

    # Make predictions
    features = data_config["data"]["features"]
    predictions = model.predict(df[features])
    probabilities = model.predict_proba(df[features])

    # Create results DataFrame
    results = pd.DataFrame(
        {
            "prediction": predictions,
            "probability_class_0": probabilities[:, 0],
            "probability_class_1": probabilities[:, 1],
        }
    )

    # Save results
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")
