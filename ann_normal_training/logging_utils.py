# Logging helpers for consistent training output.

from __future__ import annotations

import logging
import os
from datetime import datetime


def setup_logging(log_dir: str = "logs") -> logging.Logger:
    # Create a logger with file and console handlers.

    # Ensure the logging directory exists before creating a file handler.
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_log_{timestamp}.log")

    # Force reconfiguration so repeated runs in one session stay consistent.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return logging.getLogger(__name__)


def log_success(logger: logging.Logger, message: str) -> None:
    # Log a success event.

    logger.info(f"[SUCCESS] {message}")


def log_error(logger: logging.Logger, message: str) -> None:
    # Log an error event.

    logger.error(f"[ERROR] {message}")


def log_start(logger: logging.Logger, message: str) -> None:
    # Log a start event.

    logger.info(f"[START] {message}")


def log_target(logger: logging.Logger, message: str) -> None:
    # Log a target event.

    logger.info(f"[TARGET] {message}")


def log_save(logger: logging.Logger, message: str) -> None:
    # Log a save event.

    logger.info(f"[SAVE] {message}")


def log_stop(logger: logging.Logger, message: str) -> None:
    # Log a stop event.

    logger.info(f"[STOP] {message}")


def log_complete(logger: logging.Logger, message: str) -> None:
    # Log a completion event.

    logger.info(f"[COMPLETE] {message}")


def log_stats(logger: logging.Logger, message: str) -> None:
    # Log a statistics event.

    logger.info(f"[STATS] {message}")
