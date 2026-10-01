"""
Conversion Utilities Module

This module provides functions for converting between different data formats,
coordinate systems, and units commonly used in geospatial analysis.
"""

import numpy as np
from typing import Any, cast
import logging

logger = logging.getLogger(__name__)

# Unit conversion constants
METER_TO_FEET = 3.28084
FEET_TO_METER = 1.0 / METER_TO_FEET

METER_TO_YARD = 1.09361
YARD_TO_METER = 1.0 / METER_TO_YARD

METER_TO_MILE = 0.000621371
MILE_TO_METER = 1.0 / METER_TO_MILE

METER_TO_KILOMETER = 0.001
KILOMETER_TO_METER = 1000.0

# Area conversion constants
SQUARE_METER_TO_SQUARE_FEET = METER_TO_FEET**2
SQUARE_FEET_TO_SQUARE_METER = 1.0 / SQUARE_METER_TO_SQUARE_FEET

SQUARE_METER_TO_ACRE = 0.000247105
ACRE_TO_SQUARE_METER = 4046.86

SQUARE_METER_TO_HECTARE = 0.0001
HECTARE_TO_SQUARE_METER = 10000.0

# Temperature conversion constants
CELSIUS_TO_FAHRENHEIT_OFFSET = 32.0
CELSIUS_TO_FAHRENHEIT_FACTOR = 9.0 / 5.0


def degrees_to_radians(degrees: float | np.ndarray) -> float | np.ndarray:
    """
    Convert degrees to radians.

    Args:
        degrees: Angle in degrees

    Returns:
        Angle in radians
    """
    return np.radians(degrees)


def radians_to_degrees(radians: float | np.ndarray) -> float | np.ndarray:
    """
    Convert radians to degrees.

    Args:
        radians: Angle in radians

    Returns:
        Angle in degrees
    """
    return np.degrees(radians)


def celsius_to_fahrenheit(
    celsius: float | np.ndarray,
) -> float | np.ndarray:
    """
    Convert Celsius to Fahrenheit.

    Args:
        celsius: Temperature in Celsius

    Returns:
        Temperature in Fahrenheit
    """
    return celsius * CELSIUS_TO_FAHRENHEIT_FACTOR + CELSIUS_TO_FAHRENHEIT_OFFSET


def fahrenheit_to_celsius(
    fahrenheit: float | np.ndarray,
) -> float | np.ndarray:
    """
    Convert Fahrenheit to Celsius.

    Args:
        fahrenheit: Temperature in Fahrenheit

    Returns:
        Temperature in Celsius
    """
    return (fahrenheit - CELSIUS_TO_FAHRENHEIT_OFFSET) / CELSIUS_TO_FAHRENHEIT_FACTOR


def kelvin_to_celsius(kelvin: float | np.ndarray) -> float | np.ndarray:
    """
    Convert Kelvin to Celsius.

    Args:
        kelvin: Temperature in Kelvin

    Returns:
        Temperature in Celsius
    """
    return kelvin - 273.15


def celsius_to_kelvin(celsius: float | np.ndarray) -> float | np.ndarray:
    """
    Convert Celsius to Kelvin.

    Args:
        celsius: Temperature in Celsius

    Returns:
        Temperature in Kelvin
    """
    return celsius + 273.15


def meters_to_feet(meters: float | np.ndarray) -> float | np.ndarray:
    """
    Convert meters to feet.

    Args:
        meters: Distance in meters

    Returns:
        Distance in feet
    """
    return meters * METER_TO_FEET


def feet_to_meters(feet: float | np.ndarray) -> float | np.ndarray:
    """
    Convert feet to meters.

    Args:
        feet: Distance in feet

    Returns:
        Distance in meters
    """
    return feet * FEET_TO_METER


def meters_to_miles(meters: float | np.ndarray) -> float | np.ndarray:
    """
    Convert meters to miles.

    Args:
        meters: Distance in meters

    Returns:
        Distance in miles
    """
    return meters * METER_TO_MILE


def miles_to_meters(miles: float | np.ndarray) -> float | np.ndarray:
    """
    Convert miles to meters.

    Args:
        miles: Distance in miles

    Returns:
        Distance in meters
    """
    return miles * MILE_TO_METER


def meters_to_kilometers(meters: float | np.ndarray) -> float | np.ndarray:
    """
    Convert meters to kilometers.

    Args:
        meters: Distance in meters

    Returns:
        Distance in kilometers
    """
    return meters * METER_TO_KILOMETER


def kilometers_to_meters(
    kilometers: float | np.ndarray,
) -> float | np.ndarray:
    """
    Convert kilometers to meters.

    Args:
        kilometers: Distance in kilometers

    Returns:
        Distance in meters
    """
    return kilometers * KILOMETER_TO_METER


def square_meters_to_square_feet(
    sq_meters: float | np.ndarray,
) -> float | np.ndarray:
    """
    Convert square meters to square feet.

    Args:
        sq_meters: Area in square meters

    Returns:
        Area in square feet
    """
    return sq_meters * SQUARE_METER_TO_SQUARE_FEET


def square_feet_to_square_meters(
    sq_feet: float | np.ndarray,
) -> float | np.ndarray:
    """
    Convert square feet to square meters.

    Args:
        sq_feet: Area in square feet

    Returns:
        Area in square meters
    """
    return sq_feet * SQUARE_FEET_TO_SQUARE_METER


def square_meters_to_acres(
    sq_meters: float | np.ndarray,
) -> float | np.ndarray:
    """
    Convert square meters to acres.

    Args:
        sq_meters: Area in square meters

    Returns:
        Area in acres
    """
    return sq_meters * SQUARE_METER_TO_ACRE


def acres_to_square_meters(acres: float | np.ndarray) -> float | np.ndarray:
    """
    Convert acres to square meters.

    Args:
        acres: Area in acres

    Returns:
        Area in square meters
    """
    return acres * ACRE_TO_SQUARE_METER


def square_meters_to_hectares(
    sq_meters: float | np.ndarray,
) -> float | np.ndarray:
    """
    Convert square meters to hectares.

    Args:
        sq_meters: Area in square meters

    Returns:
        Area in hectares
    """
    return sq_meters * SQUARE_METER_TO_HECTARE


def hectares_to_square_meters(
    hectares: float | np.ndarray,
) -> float | np.ndarray:
    """
    Convert hectares to square meters.

    Args:
        hectares: Area in hectares

    Returns:
        Area in square meters
    """
    return hectares * HECTARE_TO_SQUARE_METER


def cartesian_to_polar(
    x: float | np.ndarray, y: float | np.ndarray
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """
    Convert Cartesian coordinates to polar coordinates.

    Args:
        x: X coordinate(s)
        y: Y coordinate(s)

    Returns:
        Tuple of (radius, angle) in radians
    """
    radius = np.sqrt(x**2 + y**2)
    angle = np.arctan2(y, x)

    return radius, angle


def polar_to_cartesian(
    radius: float | np.ndarray, angle: float | np.ndarray
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """
    Convert polar coordinates to Cartesian coordinates.

    Args:
        radius: Radius
        angle: Angle in radians

    Returns:
        Tuple of (x, y) coordinates
    """
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)

    return x, y


def spherical_to_cartesian(
    radius: float | np.ndarray,
    theta: float | np.ndarray,
    phi: float | np.ndarray,
) -> tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray]:
    """
    Convert spherical coordinates to Cartesian coordinates.

    Args:
        radius: Radius
        theta: Azimuthal angle in radians (longitude)
        phi: Polar angle in radians (latitude from z-axis)

    Returns:
        Tuple of (x, y, z) coordinates
    """
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)

    return x, y, z


def cartesian_to_spherical(
    x: float | np.ndarray,
    y: float | np.ndarray,
    z: float | np.ndarray,
) -> tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray]:
    """
    Convert Cartesian coordinates to spherical coordinates.

    Args:
        x: X coordinate
        y: Y coordinate
        z: Z coordinate

    Returns:
        Tuple of (radius, theta, phi) where theta is azimuthal angle and phi is polar angle
    """
    radius = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)
    phi = np.arctan2(np.sqrt(x**2 + y**2), z)

    return radius, theta, phi


def normalize_array(
    array: np.ndarray,
    method: str = "minmax",
    feature_range: tuple[float, float] = (0, 1),
) -> np.ndarray:
    """
    Normalize array using specified method.

    Args:
        array: Input array
        method: Normalization method ('minmax', 'zscore', 'robust')
        feature_range: Feature range for minmax normalization

    Returns:
        Normalized array
    """
    if method == "minmax":
        min_val = np.min(array)
        max_val = np.max(array)

        if max_val == min_val:
            return np.full_like(array, feature_range[0])

        normalized = (array - min_val) / (max_val - min_val)
        return cast(
            np.ndarray,
            normalized * (feature_range[1] - feature_range[0]) + feature_range[0],
        )

    elif method == "zscore":
        mean_val = np.mean(array)
        std_val = np.std(array)

        if std_val == 0:
            return np.zeros_like(array)

        return cast(np.ndarray, (array - mean_val) / std_val)

    elif method == "robust":
        median_val = np.median(array)
        mad = np.median(np.abs(array - median_val))

        if mad == 0:
            return np.zeros_like(array)

        return cast(np.ndarray, (array - median_val) / mad)

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def standardize_array(
    array: np.ndarray, center: bool = True, scale: bool = True
) -> np.ndarray:
    """
    Standardize array (center and/or scale).

    Args:
        array: Input array
        center: Whether to center (subtract mean)
        scale: Whether to scale (divide by standard deviation)

    Returns:
        Standardized array
    """
    result = array.copy()

    if center:
        result = result - np.mean(result)

    if scale:
        std_val = np.std(result)
        if std_val != 0:
            result = result / std_val

    return result


def convert_data_types(data: Any, target_type: type) -> Any:
    """
    Convert data to target type with appropriate handling.

    Args:
        data: Input data
        target_type: Target data type

    Returns:
        Converted data
    """
    if target_type == np.ndarray:
        if isinstance(data, list):
            return np.array(data)
        elif isinstance(data, (int, float)):
            return np.array([data])
        elif isinstance(data, np.ndarray):
            return data.copy()
        else:
            try:
                return np.array(data)
            except Exception:
                raise ValueError(f"Cannot convert {type(data)} to numpy array")

    elif target_type is list:
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (int, float)):
            return [data]
        else:
            return list(data)

    elif target_type in [int, float]:
        if isinstance(data, np.ndarray) and data.size == 1:
            return target_type(data.item())
        elif isinstance(data, (list, tuple)) and len(data) == 1:
            return target_type(data[0])
        else:
            return target_type(data)

    else:
        return target_type(data)


def format_coordinate_string(
    lat: float, lon: float, format_type: str = "decimal"
) -> str:
    """
    Format coordinates as a string.

    Args:
        lat: Latitude
        lon: Longitude
        format_type: Format type ('decimal', 'dms', 'dm')

    Returns:
        Formatted coordinate string
    """
    if format_type == "decimal":
        return f"{lat:.6f}, {lon:.6f}"

    elif format_type == "dms":

        def decimal_to_dms(decimal: float, is_latitude: bool = True) -> str:
            abs_decimal = abs(decimal)
            degrees = int(abs_decimal)
            minutes = int((abs_decimal - degrees) * 60)
            seconds = (abs_decimal - degrees - minutes / 60) * 3600

            direction = (
                "N"
                if decimal >= 0 and is_latitude
                else "S"
                if is_latitude
                else "E"
                if decimal >= 0
                else "W"
            )

            return f"{degrees}°{minutes:02d}'{seconds:04.1f}\"{direction}"

        lat_dms = decimal_to_dms(lat, True)
        lon_dms = decimal_to_dms(lon, False)

        return f"{lat_dms}, {lon_dms}"

    elif format_type == "dm":

        def decimal_to_dm(decimal: float, is_latitude: bool = True) -> str:
            abs_decimal = abs(decimal)
            degrees = int(abs_decimal)
            minutes = (abs_decimal - degrees) * 60

            direction = (
                "N"
                if decimal >= 0 and is_latitude
                else "S"
                if is_latitude
                else "E"
                if decimal >= 0
                else "W"
            )

            return f"{degrees}°{minutes:05.2f}'{direction}"

        lat_dm = decimal_to_dm(lat, True)
        lon_dm = decimal_to_dm(lon, False)

        return f"{lat_dm}, {lon_dm}"

    else:
        raise ValueError(f"Unknown format type: {format_type}")


def parse_coordinate_string(coord_string: str) -> tuple[float, float]:
    """
    Parse coordinate string to decimal degrees.

    Args:
        coord_string: Coordinate string (e.g., "40.7128, -74.0060")

    Returns:
        Tuple of (latitude, longitude) in decimal degrees
    """
    try:
        parts = coord_string.replace(" ", "").split(",")
        if len(parts) != 2:
            raise ValueError(
                "Coordinate string must contain exactly two values separated by comma"
            )

        lat = float(parts[0])
        lon = float(parts[1])

        if not (-90 <= lat <= 90):
            raise ValueError("Latitude must be between -90 and 90 degrees")

        if not (-180 <= lon <= 180):
            raise ValueError("Longitude must be between -180 and 180 degrees")

        return lat, lon

    except (ValueError, IndexError) as e:
        raise ValueError(
            f"Invalid coordinate string format: {coord_string}. Expected 'lat, lon'"
        ) from e


__all__ = [
    "degrees_to_radians",
    "radians_to_degrees",
    "celsius_to_fahrenheit",
    "fahrenheit_to_celsius",
    "kelvin_to_celsius",
    "celsius_to_kelvin",
    "meters_to_feet",
    "feet_to_meters",
    "meters_to_miles",
    "miles_to_meters",
    "meters_to_kilometers",
    "kilometers_to_meters",
    "square_meters_to_square_feet",
    "square_feet_to_square_meters",
    "square_meters_to_acres",
    "acres_to_square_meters",
    "square_meters_to_hectares",
    "hectares_to_square_meters",
    "cartesian_to_polar",
    "polar_to_cartesian",
    "spherical_to_cartesian",
    "cartesian_to_spherical",
    "normalize_array",
    "standardize_array",
    "convert_data_types",
    "format_coordinate_string",
    "parse_coordinate_string",
]
