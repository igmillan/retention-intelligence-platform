"""
Logging utilities for Case 01.

This module provides a centralized logger configuration so that all pipeline
components use consistent formatting, levels, and file outputs.
"""

import logging
from pathlib import Path

from config.settings import LOGGING_CONFIG, LOGS_DIR


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Create or retrieve a configured logger instance.

    Parameters
    ----------
    name : str | None, optional
        Logger name. If None, the default project logger name is used.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger_name = name or LOGGING_CONFIG["logger_name"]
    logger = logging.getLogger(logger_name)

    if logger.handlers:
        return logger

    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    logger.setLevel(getattr(logging, LOGGING_CONFIG["log_level"]))
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_file_path = Path(LOGS_DIR) / LOGGING_CONFIG["log_filename"]

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger