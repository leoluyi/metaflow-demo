import os
from configparser import ConfigParser
from pathlib import Path


def load_config(config_path: str) -> dict:
    """Load INI configuration file."""
    config = ConfigParser()
    config.read(config_path)

    return {section: dict(config.items(section)) for section in config.sections()}


def save_config(config: dict, config_path: str):
    """Save configuration to INI file."""
    parser = ConfigParser()
    for section, values in config.items():
        parser[section] = values

    os.makedirs(Path(config_path).parent, exist_ok=True)
    with open(config_path, "w") as f:
        parser.write(f)
