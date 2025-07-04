import logging
from environment_vars import LOG_FILE, LOG_LEVEL
import os

import logging
import logging.config
import os

# Example environment variables (replace with your actual import)

def setup_logger(
    log_file=LOG_FILE,
    level=LOG_LEVEL,
    app_name='AI-DM'
):
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_format = (
        "[%(asctime)s] {app} [%(levelname)s] [%(name)s] %(message)s "
        "[file=%(filename)s:%(lineno)d]"
    ).format(app=app_name)

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": log_format,
                "datefmt": "%Y-%m-%d %H:%M:%S"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": level,
                "formatter": "standard",
                "stream": "ext://sys.stdout"
            },
            "file": {
                "class": "logging.handlers.TimedRotatingFileHandler",
                "level": level,
                "formatter": "standard",
                "filename": log_file,
                "when": "midnight",
                "backupCount": 7,
                "encoding": "utf8",
                "utc": True
            }
        },
        "loggers": {
            "": {  # root logger
                "handlers": ["console", "file"],
                "level": level,
                "propagate": False
            }
        }
    }

    logging.config.dictConfig(logging_config)

# Usage example
if __name__ == "__main__":
    setup_logger(app_name="my-app")
    logger = logging.getLogger(__name__)

    logger.debug("This is a DEBUG message.")
    logger.info("This is an INFO message.")
    logger.warning("This is a WARNING message.")
    logger.error("This is an ERROR message.")
    logger.critical("This is a CRITICAL message.")

