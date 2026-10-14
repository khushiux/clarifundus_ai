import logging
import os
import sys
from typing import Optional

def setup_logger(
    name: str = "retinatrust",
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """Configures a standardized logger with console and optional file handler.

    Args:
        name: Name of the logger.
        log_file: Optional file path to write log messages.
        level: Logging verbosity level.

    Returns:
        logging.Logger configured instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
