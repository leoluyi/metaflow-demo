import logging
import os
from pathlib import Path

from .config.config_parser import Config
from .utils.logger import register_console_file_handler

# Get environment from ENV variable or default to 'dev'
env = os.getenv("PIPE_ENV", "dev")

# Initialize config with environment
config = Config(env)
LOG_FILE = Path(config.paths.get("log_dir", "logs")) / "syslog.log"

default_logger = logging.getLogger(__name__)
register_console_file_handler(default_logger, LOG_FILE)
