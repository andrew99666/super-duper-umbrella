"""Logging configuration helpers for the FastAPI application."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

DEFAULT_LOG_DIR = Path(os.getenv("APP_LOG_DIR", "logs"))
DEFAULT_LOG_FILE = os.getenv("APP_LOG_FILENAME", "latest-run.log")


def _normalise_level(level: Optional[str | int]) -> int:
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        value = level.upper()
        if value.isdigit():
            return int(value)
        if value in logging._nameToLevel:  # type: ignore[attr-defined]
            return logging._nameToLevel[value]  # pragma: no cover - stdlib mapping
    return logging.INFO


def configure_logging(level: Optional[str | int] = None) -> Path:
    """Configure root logging to stream to console and a fresh file.

    The log file is truncated on every call so that each application run starts
    with a clean slate. The path is returned to aid diagnostics and optional
    tooling.
    """

    log_level = _normalise_level(level)
    log_dir = DEFAULT_LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / DEFAULT_LOG_FILE

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        ],
    )
    logging.getLogger(__name__).info("Application logs initialised at %s", log_path)
    return log_path


if __name__ == "__main__":  # pragma: no cover - manual script usage
    path = configure_logging(os.getenv("LOG_LEVEL"))
    print(f"Log file created at {path}")
