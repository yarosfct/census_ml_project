"""
Logging utilities for the census ML project.

Provides a configured logger for consistent logging across the project.
"""

import logging
import sys

from census_ml.config import LOG_LEVEL


def get_logger(
    name: str,
    level: str | None = None,
    log_to_file: str | None = None,
) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: Name of the logger (typically __name__ from the calling module)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               If None, uses LOG_LEVEL from config.
        log_to_file: Optional file path to write logs to.

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Set level
    log_level = level or LOG_LEVEL
    logger.setLevel(getattr(logging, log_level.upper()))

    # Avoid adding handlers multiple times
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)

        # Formatter
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Optional file handler
        if log_to_file:
            file_handler = logging.FileHandler(log_to_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
