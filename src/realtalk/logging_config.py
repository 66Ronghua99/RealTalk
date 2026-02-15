"""Logging configuration for RealTalk."""
import logging
import sys
from typing import Optional


def setup_logger(
    name: str,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """Set up a logger with the specified configuration."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Disable propagation to prevent duplicate logs when child loggers have handlers
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

        if format_string is None:
            format_string = (
                "%(asctime)s - %(name)s - %(levelname)s - "
                "[%(filename)s:%(lineno)d] - %(message)s"
            )

        formatter = logging.Formatter(format_string)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


# Default logger
logger = setup_logger("realtalk")
