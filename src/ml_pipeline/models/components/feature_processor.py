# src/ml_pipeline/models/components/feature_processor.py
from typing import Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ml_pipeline.utils.errors import FeatureProcessorError


class FeatureProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self._fitted = False

    def fit_transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Fit scaler and transform data"""
        is_dataframe = isinstance(X, pd.DataFrame)
        columns = X.columns if is_dataframe else None

        X_scaled = self.scaler.fit_transform(X)
        self._fitted = True

        if is_dataframe:
            return pd.DataFrame(X_scaled, columns=columns, index=X.index)
        return X_scaled

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Transform data using fitted scaler"""
        if not self._fitted:
            raise FeatureProcessorError(
                "FeatureProcessor must be fitted before transform"
            )

        is_dataframe = isinstance(X, pd.DataFrame)
        columns = X.columns if is_dataframe else None

        X_scaled = self.scaler.transform(X)

        if is_dataframe:
            return pd.DataFrame(X_scaled, columns=columns, index=X.index)
        return X_scaled
