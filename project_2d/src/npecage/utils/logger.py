import logging


def setup_logger(level=logging.INFO):
    """
    Set up a logger with the specified level.

    Args:
        level (int): The logging level (default is logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger()
    logger.setLevel(level)

    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(ch)

    return logger
