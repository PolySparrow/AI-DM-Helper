import logging
from environment_vars import LOG_FILE, LOG_LEVEL
import os

def setup_logger(
    
    log_file=LOG_FILE,
    mode_type='w',
    level=LOG_LEVEL
):
    """Function to setup a logger with a single file handler."""
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(__name__)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Only add a file handler if not already present
    if not any(
        isinstance(h, logging.FileHandler) and h.baseFilename == log_file
        for h in logger.handlers
    ):
        handler = logging.FileHandler(log_file, mode=mode_type)
        handler.setFormatter(formatter)
        handler.setLevel(level)
        logger.addHandler(handler)

    return logger

