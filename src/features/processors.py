from sklearn.base import BaseEstimator, TransformerMixin


class FeatureProcessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_stats = {}

    def fit(self, X, y=None):
        """Calculate feature statistics."""
        for col in X.columns:
            self.feature_stats[col] = {
                "mean": X[col].mean(),
                "std": X[col].std(),
                "min": X[col].min(),
                "max": X[col].max(),
            }
        return self

    def transform(self, X):
        """Transform features."""
        X_transformed = X.copy()

        # Scale numerical features
        for col in X.columns:
            if col in self.feature_stats:
                X_transformed[col] = (
                    X[col] - self.feature_stats[col]["mean"]
                ) / self.feature_stats[col]["std"]

        # Add interaction features
        X_transformed["f1_f2"] = X_transformed["feature1"] * X_transformed["feature2"]
        X_transformed["f3_f4"] = X_transformed["feature3"] * X_transformed["feature4"]

        return X_transformed

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
