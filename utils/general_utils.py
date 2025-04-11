# General-purpose utility functions (e.g., logging, error handling)

import logging

def setup_logger(log_file="app.log"):
    """
    Set up a logger for the application.
    Parameters
    ----------
    log_file : str
        Path to the log file.
    Returns
    -------
    logging.Logger
        Configured logger object.
    """
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger()

def log_and_raise_error(logger, message):
    """
    Log an error message and raise an exception.
    Parameters
    ----------
    logger : logging.Logger
        Logger object.
    message : str
        Error message.
    """
    logger.error(message)
    raise ValueError(message)