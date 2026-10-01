"""Small, dependency-free helpers for ANT spatial inputs.

The ANT algorithms accept spatial configuration from several GEO-INFER
modules.  Keeping the normalization here prevents each algorithm from
silently accepting a different H3 or bounds representation.
"""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any
from collections.abc import Mapping


def parse_h3_resolution(value: Any) -> int:
    """Return an integer H3 resolution from common ANT representations.

    ``SpatialIndexingInterface`` expects an integer, while ANT configuration
    historically used values such as ``"h3_r8"``.  Both forms remain
    supported, but malformed values fail early with a useful error.
    """

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized.startswith("h3_r"):
            normalized = normalized[4:]
        try:
            value = int(normalized)
        except ValueError as exc:
            raise ValueError(f"Invalid H3 resolution: {value!r}") from exc
    elif isinstance(value, Integral):
        value = int(value)
    else:
        raise TypeError("H3 resolution must be an integer or 'h3_rN' string")

    if not 0 <= value <= 15:
        raise ValueError("H3 resolution must be between 0 and 15")
    return int(value)


def validate_bounds(bounds: Mapping[str, Any]) -> dict[str, float]:
    """Validate and return geographic bounds as floats."""

    required = ("min_lat", "max_lat", "min_lng", "max_lng")
    missing = [key for key in required if key not in bounds]
    if missing:
        raise ValueError(f"Missing spatial bounds: {', '.join(missing)}")

    normalized = {}
    for key in required:
        value = bounds[key]
        if not isinstance(value, Real):
            raise TypeError(f"Spatial bound {key!r} must be numeric")
        normalized[key] = float(value)

    if not normalized["min_lat"] < normalized["max_lat"]:
        raise ValueError("min_lat must be less than max_lat")
    if not normalized["min_lng"] < normalized["max_lng"]:
        raise ValueError("min_lng must be less than max_lng")
    if not -90 <= normalized["min_lat"] <= 90 or not -90 <= normalized["max_lat"] <= 90:
        raise ValueError("Latitude bounds must be within [-90, 90]")
    if (
        not -180 <= normalized["min_lng"] <= 180
        or not -180 <= normalized["max_lng"] <= 180
    ):
        raise ValueError("Longitude bounds must be within [-180, 180]")
    return normalized


def validate_numeric_matrix(matrix: Any, size: int, name: str) -> Any:
    """Validate a finite square matrix used by an optimizer."""

    import numpy as np

    array = np.asarray(matrix, dtype=float)
    if array.shape != (size, size):
        raise ValueError(f"{name} must have shape ({size}, {size})")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(array < 0):
        raise ValueError(f"{name} must not contain negative values")
    return array
