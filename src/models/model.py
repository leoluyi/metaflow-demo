from typing import Any, Dict

from sklearn.ensemble import RandomForestClassifier


def create_model(params: Dict[str, Any]) -> RandomForestClassifier:
    return RandomForestClassifier(**params)
