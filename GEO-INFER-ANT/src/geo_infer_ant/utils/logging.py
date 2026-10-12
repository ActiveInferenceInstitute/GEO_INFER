"""Logging configuration helpers for GEO-INFER-ANT entry points."""

from __future__ import annotations

import logging


def setup_logging(
    level: str = "INFO", logger_name: str | None = None
) -> logging.Logger:
    """Configure and return an ANT logger without duplicating handlers."""
    logger = logging.getLogger(logger_name or "geo_infer_ant")
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Unknown logging level: {level}")
    logger.setLevel(numeric_level)
    return logger


__all__ = ["setup_logging"]
