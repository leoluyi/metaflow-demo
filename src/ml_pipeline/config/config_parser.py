from configparser import ConfigParser

from ml_pipeline.utils.errors import ConfigError


class Config:
    def __init__(self, env: str, config_path: str = None):
        self.env = env
        self.config = ConfigParser()
        if config_path is None:
            import os

            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_dir, "config.ini")

        self.config.read(config_path)

        if env.upper() not in self.config.sections():
            raise ConfigError(f"Environment {env} not found in config")

    @property
    def paths(self):
        return self.config[self.env.upper()]

    @property
    def model_config(self):
        return self.config["MODEL"]

    @property
    def mlflow_config(self):
        return {
            "tracking_uri": self.config[self.env]["mlflow.tracking_uri"],
            "experiment_name": self.config["MLFLOW"]["experiment_name"],
        }
