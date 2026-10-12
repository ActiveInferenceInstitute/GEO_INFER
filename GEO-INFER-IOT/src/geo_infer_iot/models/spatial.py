"""Validated sensor coordinates and H3 resolution at IoT boundaries."""

import math

import h3


def sensor_cell(latitude: float, longitude: float, resolution: int) -> str:
    """Return the H3 cell for finite WGS84 coordinates, including resolution 0."""
    if (
        isinstance(resolution, bool)
        or not isinstance(resolution, int)
        or not 0 <= resolution <= 15
    ):
        raise ValueError("h3_resolution must be an integer from 0 to 15")
    if not math.isfinite(latitude) or not -90 <= latitude <= 90:
        raise ValueError("latitude must be finite and between -90 and 90")
    if not math.isfinite(longitude) or not -180 <= longitude <= 180:
        raise ValueError("longitude must be finite and between -180 and 180")
    return h3.latlng_to_cell(latitude, longitude, resolution)
