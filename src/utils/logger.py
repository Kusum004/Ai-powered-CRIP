# src/utils/logger.py
"""Centralized structured logger for Credit Risk Platform."""
import logging
import sys
import os

def get_logger(name: str = "CreditRiskPlatform", level: str = None) -> logging.Logger:
    """Returns a configured logger with consistent formatting."""
    log_level = level or os.getenv("LOG_LEVEL", "INFO").upper()
    numeric_level = getattr(logging, log_level, logging.INFO)
    
    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(numeric_level)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger
