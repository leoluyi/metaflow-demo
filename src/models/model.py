# src/models/model.py
import joblib
from sklearn.ensemble import RandomForestClassifier


class Model:
    def __init__(self, **params):
        self.model = RandomForestClassifier(**params)

    def fit(self, X, y):
        """Train the model."""
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """Generate predictions."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Generate probability predictions."""
        return self.model.predict_proba(X)

    def save(self, path):
        """Save model to disk."""
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path):
        """Load model from disk."""
        model = cls()
        model.model = joblib.load(path)
        return model
