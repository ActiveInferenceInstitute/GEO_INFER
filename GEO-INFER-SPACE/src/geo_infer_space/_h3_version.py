"""Shared, dependency-free parsing of the supported h3-py version range."""

from typing import cast

MIN_H3_VERSION = (4, 5, 0)
MAX_H3_MAJOR = 5


def parse_h3_version(version: str) -> tuple[int, int, int] | None:
    """Parse the h3-py version; callers own their availability/error behavior."""
    try:
        parts = version.lstrip("v").split(".")
        return cast(
            tuple[int, int, int],
            tuple(int(part.split("+")[0].split("-")[0]) for part in parts[:3])
            + (0,) * max(0, 3 - len(parts)),
        )
    except (AttributeError, TypeError, ValueError):
        return None
