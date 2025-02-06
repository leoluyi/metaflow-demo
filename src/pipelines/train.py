from pathlib import Path

import joblib

from src.data.loader import load_data
from src.models.model import create_model
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.metrics import calculate_classification_metrics, log_metrics


def train():
    data_config = load_config("configs/base/data.yaml")
    model_config = load_config("configs/base/model.yaml")

    logger = setup_logger("train", "logs/train.log")

    # Load data
    train_df, test_df = load_data(
        data_config["data"]["train_path"], data_config["data"]["test_path"]
    )

    # Prepare features and target
    features = data_config["data"]["features"]
    target = data_config["data"]["target"]

    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]

    # Create and train model
    model = create_model(model_config["model"]["params"])
    model.fit(X_train, y_train)

    # Calculate metrics
    train_pred = model.predict(X_train)
    train_prob = model.predict_proba(X_train)[:, 1]
    train_metrics = calculate_classification_metrics(y_train, train_pred, train_prob)

    test_pred = model.predict(X_test)
    test_prob = model.predict_proba(X_test)[:, 1]
    test_metrics = calculate_classification_metrics(y_test, test_pred, test_prob)

    # Log metrics
    logger.info("Training Metrics:")
    log_metrics(logger, train_metrics, prefix="train_")

    logger.info("\nTest Metrics:")
    log_metrics(logger, test_metrics, prefix="test_")

    # Save model
    model_path = data_config["paths"]["model"]
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    logger.info(f"\nModel saved to {model_path}")
