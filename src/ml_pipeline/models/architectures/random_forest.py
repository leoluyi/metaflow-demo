import logging

from sklearn.ensemble import RandomForestClassifier

from .base_model import BaseModel


class RFModel(BaseModel):
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state
        )

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
