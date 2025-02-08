from typing import Dict

import pandas as pd


def log_data_drift(
    reference_data: pd.DataFrame, current_data: pd.DataFrame, feature_columns: list
) -> Dict[str, float]:
    drift_metrics = {}
    for col in feature_columns:
        drift_metrics[f"{col}_drift"] = (
            current_data[col].mean() - reference_data[col].mean()
        )
    return drift_metrics
