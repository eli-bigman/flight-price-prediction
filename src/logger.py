import logging
import os
import sys
from src.config import config

# Create logs directory if it doesn't exist
LOGS_DIR = config.PROJECT_ROOT / "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

LOG_FILE = LOGS_DIR / "app.log"

def get_logger(name):
    """
    Creates and configures a logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Check if logger already has handlers to prevent duplicate logs
    if not logger.handlers:
        # File Handler
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setLevel(logging.INFO)
        file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)

        # Stream Handler (Console)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        stream_handler.setFormatter(stream_format)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger
