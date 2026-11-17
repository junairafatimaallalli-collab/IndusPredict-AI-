"""
IndusPredict AI - Logging and Audit Trail System.
Records structured logs for industrial observability and database audit trails.
"""

from datetime import datetime
import logging
from pathlib import Path
import sys
from typing import Optional

from induspredict.core.config import APP_DATA_DIR

LOGS_DIR = APP_DATA_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
SYSTEM_LOG_FILE = LOGS_DIR / "induspredict.log"

# Configure root logger
logger = logging.getLogger("induspredict")
logger.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter(
    fmt="[%(asctime)s] [%(levelname)s] [%(name)s:%(module)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Stream handler
if not logger.handlers:
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # File handler
    file_handler = logging.FileHandler(SYSTEM_LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get child logger namespaced to induspredict."""
    if name:
        return logger.getChild(name)
    return logger
