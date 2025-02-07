import logging
from typing import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlertHandler:
    def __init__(self, thresholds: Dict[str, float]):
        self.thresholds = thresholds

    def check_metrics(self, metrics: Dict[str, float]):
        alerts = []
        for metric, value in metrics.items():
            if metric in self.thresholds:
                if value > self.thresholds[metric]:
                    alert = f"Alert: {metric} exceeded threshold: {value}"
                    alerts.append(alert)
                    logger.warning(alert)
        return alerts
