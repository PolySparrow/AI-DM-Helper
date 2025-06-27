import logging
def setup_logger(
    name='gelatinous_cube',
    log_file='AI_DM.log',
    mode_type='a',
    level=logging.INFO
):
    """Function to setup a logger with a single file handler."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
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


