"""Logging setup for SEA."""

from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Configure structured logging for the SEA platform."""
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
        force=True,
    )

    # Reduce noise from libraries
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
