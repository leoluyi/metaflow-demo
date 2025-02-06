from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None
) -> Dict[str, Any]:
    """Calculate classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities for positive class

    Returns:
        Dictionary containing metrics
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
    }

    if y_prob is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))

    conf_matrix = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = conf_matrix.tolist()

    return metrics


def calculate_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """Calculate regression metrics.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        Dictionary containing metrics
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def log_metrics(logger, metrics: Dict[str, Any], prefix: str = "") -> None:
    """Log metrics using the provided logger.

    Args:
        logger: Logger instance
        metrics: Dictionary of metrics
        prefix: Optional prefix for metric names
    """
    for name, value in metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"{prefix}{name}: {value:.4f}")
        elif name == "confusion_matrix":
            logger.info(f"{prefix}Confusion Matrix:\n{np.array(value)}")
        else:
            logger.info(f"{prefix}{name}: {value}")
