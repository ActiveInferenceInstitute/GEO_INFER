"""
Constants Module

This module contains mathematical, physical, and geospatial constants
used throughout the GEO-INFER-MATH library.
"""

import numpy as np
from typing import Any

# Earth parameters (WGS84)
EARTH_RADIUS_EQUATORIAL = 6378137.0  # Equatorial radius in meters
EARTH_RADIUS_POLAR = 6356752.314245  # Polar radius in meters
EARTH_RADIUS_MEAN = 6371008.7714  # Mean radius in meters
EARTH_FLATTENING = 1 / 298.257223563  # Flattening
EARTH_ECCENTRICITY = np.sqrt(2 * EARTH_FLATTENING - EARTH_FLATTENING**2)  # Eccentricity

# Earth's gravity parameters
EARTH_GRAVITY_EQUATORIAL = 9.780327  # Gravity at equator (m/s²)
EARTH_GRAVITY_POLES = 9.832186  # Gravity at poles (m/s²)
EARTH_GRAVITY_MEAN = 9.80665  # Standard gravity (m/s²)

# Mathematical constants
PI = np.pi
EULER_GAMMA = 0.5772156649015329  # Euler-Mascheroni constant
GOLDEN_RATIO = (1 + np.sqrt(5)) / 2

# Spatial analysis constants
DEFAULT_SPATIAL_WEIGHTS_K = 5  # Default number of nearest neighbors
DEFAULT_VARIANCE_THRESHOLD = 1e-10  # Threshold for variance calculations
DEFAULT_CONVERGENCE_TOLERANCE = 1e-6  # Default convergence tolerance
DEFAULT_MAX_ITERATIONS = 1000  # Default maximum iterations

# Interpolation constants
DEFAULT_IDW_POWER = 2.0  # Default inverse distance weighting power
DEFAULT_KRIGING_RANGE = 1.0  # Default kriging range parameter
DEFAULT_KRIGING_SILL = 1.0  # Default kriging sill parameter
DEFAULT_KRIGING_NUGGET = 0.1  # Default kriging nugget parameter

# Statistical constants
DEFAULT_CONFIDENCE_LEVEL = 0.95  # Default confidence level
DEFAULT_SIGNIFICANCE_LEVEL = 0.05  # Default significance level
DEFAULT_Z_SCORE_THRESHOLD = 1.96  # Default z-score threshold for 95% confidence

# Coordinate system constants
WGS84_EPSG_CODE = 4326
WEB_MERCATOR_EPSG_CODE = 3857
UTM_ZONE_WIDTH_DEGREES = 6  # Degrees per UTM zone
UTM_CENTRAL_MERIDIAN_OFFSET = 183  # For zone 1: 180 + 3 = 183

# Time constants (in seconds)
SECOND = 1
MINUTE = 60
HOUR = 3600
DAY = 86400
WEEK = 604800
MONTH = 2629746  # Average month (30.44 days)
YEAR = 31556952  # Tropical year

# Unit conversion constants
METER_TO_FEET = 3.28084
FEET_TO_METER = 1.0 / METER_TO_FEET

METER_TO_YARD = 1.09361
YARD_TO_METER = 1.0 / METER_TO_YARD

METER_TO_MILE = 0.000621371
MILE_TO_METER = 1.0 / METER_TO_MILE

METER_TO_KILOMETER = 0.001
KILOMETER_TO_METER = 1000.0

METER_TO_NAUTICAL_MILE = 0.000539957
NAUTICAL_MILE_TO_METER = 1852.0

# Area conversion constants
SQUARE_METER_TO_SQUARE_FEET = METER_TO_FEET**2
SQUARE_FEET_TO_SQUARE_METER = 1.0 / SQUARE_METER_TO_SQUARE_FEET

SQUARE_METER_TO_SQUARE_YARD = METER_TO_YARD**2
SQUARE_YARD_TO_SQUARE_METER = 1.0 / SQUARE_METER_TO_SQUARE_YARD

SQUARE_METER_TO_ACRE = 0.000247105
ACRE_TO_SQUARE_METER = 4046.86

SQUARE_METER_TO_HECTARE = 0.0001
HECTARE_TO_SQUARE_METER = 10000.0

SQUARE_METER_TO_SQUARE_KILOMETER = 1e-6
SQUARE_KILOMETER_TO_SQUARE_METER = 1e6

# Volume conversion constants
CUBIC_METER_TO_LITER = 1000.0
LITER_TO_CUBIC_METER = 0.001

CUBIC_METER_TO_CUBIC_FEET = METER_TO_FEET**3
CUBIC_FEET_TO_CUBIC_METER = 1.0 / CUBIC_METER_TO_CUBIC_FEET

# Temperature conversion constants
CELSIUS_TO_FAHRENHEIT_OFFSET = 32.0
CELSIUS_TO_FAHRENHEIT_FACTOR = 9.0 / 5.0
FAHRENHEIT_TO_CELSIUS_FACTOR = 5.0 / 9.0

KELVIN_TO_CELSIUS_OFFSET = 273.15
CELSIUS_TO_KELVIN_OFFSET = -KELVIN_TO_CELSIUS_OFFSET

# Pressure conversion constants (to Pascals)
ATMOSPHERE_TO_PASCAL = 101325.0
PASCAL_TO_ATMOSPHERE = 1.0 / ATMOSPHERE_TO_PASCAL

BAR_TO_PASCAL = 100000.0
PASCAL_TO_BAR = 1.0 / BAR_TO_PASCAL

# Speed conversion constants (to m/s)
KILOMETER_PER_HOUR_TO_METER_PER_SECOND = 1.0 / 3.6
METER_PER_SECOND_TO_KILOMETER_PER_HOUR = 3.6

MILE_PER_HOUR_TO_METER_PER_SECOND = 0.44704
METER_PER_SECOND_TO_MILE_PER_HOUR = 2.23694

KNOT_TO_METER_PER_SECOND = 0.514444
METER_PER_SECOND_TO_KNOT = 1.94384

# Angle conversion constants
DEGREE_TO_RADIAN = np.pi / 180.0
RADIAN_TO_DEGREE = 180.0 / np.pi

ARC_MINUTE_TO_DEGREE = 1.0 / 60.0
DEGREE_TO_ARC_MINUTE = 60.0

ARC_SECOND_TO_DEGREE = 1.0 / 3600.0
DEGREE_TO_ARC_SECOND = 3600.0

# Frequency conversion constants
HERTZ_TO_KILOHERTZ = 0.001
KILOHERTZ_TO_HERTZ = 1000.0

HERTZ_TO_MEGAHERTZ = 1e-6
MEGAHERTZ_TO_HERTZ = 1e6

# Quality control constants
MAX_COORDINATE_VALUE = 1e9  # Maximum reasonable coordinate value
MIN_COORDINATE_VALUE = -1e9  # Minimum reasonable coordinate value

MAX_DISTANCE_VALUE = 1e7  # Maximum reasonable distance in meters
MIN_DISTANCE_VALUE = 0.0  # Minimum reasonable distance

MAX_VALUE_RATIO = 1e12  # Maximum ratio between max and min values
MIN_NONZERO_VALUE = 1e-15  # Minimum non-zero value to avoid underflow

# Algorithm constants
DEFAULT_RANDOM_SEED = 42
MAX_MATRIX_SIZE = 10000  # Maximum matrix size for memory-intensive operations
DEFAULT_CHUNK_SIZE = 1000  # Default chunk size for processing large datasets

# Statistical distribution constants
NORMAL_DISTRIBUTION_Z_SCORES = {
    0.80: 1.282,  # 80% confidence
    0.85: 1.440,  # 85% confidence
    0.90: 1.645,  # 90% confidence
    0.95: 1.960,  # 95% confidence
    0.99: 2.576,  # 99% confidence
    0.999: 3.291,  # 99.9% confidence
}

# Geospatial analysis constants
DEFAULT_BUFFER_DISTANCE = 1000.0  # Default buffer distance in meters
DEFAULT_GRID_RESOLUTION = 100.0  # Default grid resolution in meters
DEFAULT_SPATIAL_TOLERANCE = 1e-6  # Default spatial tolerance

# Machine learning constants
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 100
DEFAULT_VALIDATION_SPLIT = 0.2

# Optimization constants
DEFAULT_OPTIMIZATION_TOLERANCE = 1e-6
DEFAULT_OPTIMIZATION_MAX_ITER = 1000
DEFAULT_OPTIMIZATION_POPULATION_SIZE = 50

# Constants dictionary for easy access
CONSTANTS = {
    # Earth parameters
    "earth": {
        "radius_equatorial": EARTH_RADIUS_EQUATORIAL,
        "radius_polar": EARTH_RADIUS_POLAR,
        "radius_mean": EARTH_RADIUS_MEAN,
        "flattening": EARTH_FLATTENING,
        "eccentricity": EARTH_ECCENTRICITY,
        "gravity_equatorial": EARTH_GRAVITY_EQUATORIAL,
        "gravity_poles": EARTH_GRAVITY_POLES,
        "gravity_mean": EARTH_GRAVITY_MEAN,
    },
    # Unit conversions
    "unit_conversions": {
        "length": {
            "meter_to_feet": METER_TO_FEET,
            "feet_to_meter": FEET_TO_METER,
            "meter_to_yard": METER_TO_YARD,
            "yard_to_meter": YARD_TO_METER,
            "meter_to_mile": METER_TO_MILE,
            "mile_to_meter": MILE_TO_METER,
            "meter_to_kilometer": METER_TO_KILOMETER,
            "kilometer_to_meter": KILOMETER_TO_METER,
            "meter_to_nautical_mile": METER_TO_NAUTICAL_MILE,
            "nautical_mile_to_meter": NAUTICAL_MILE_TO_METER,
        },
        "area": {
            "square_meter_to_square_feet": SQUARE_METER_TO_SQUARE_FEET,
            "square_feet_to_square_meter": SQUARE_FEET_TO_SQUARE_METER,
            "square_meter_to_acre": SQUARE_METER_TO_ACRE,
            "acre_to_square_meter": ACRE_TO_SQUARE_METER,
            "square_meter_to_hectare": SQUARE_METER_TO_HECTARE,
            "hectare_to_square_meter": HECTARE_TO_SQUARE_METER,
        },
        "temperature": {
            "celsius_to_fahrenheit_offset": CELSIUS_TO_FAHRENHEIT_OFFSET,
            "celsius_to_fahrenheit_factor": CELSIUS_TO_FAHRENHEIT_FACTOR,
            "fahrenheit_to_celsius_factor": FAHRENHEIT_TO_CELSIUS_FACTOR,
            "kelvin_to_celsius_offset": KELVIN_TO_CELSIUS_OFFSET,
            "celsius_to_kelvin_offset": CELSIUS_TO_KELVIN_OFFSET,
        },
    },
    # Algorithm parameters
    "algorithms": {
        "spatial_analysis": {
            "default_weights_k": DEFAULT_SPATIAL_WEIGHTS_K,
            "variance_threshold": DEFAULT_VARIANCE_THRESHOLD,
            "convergence_tolerance": DEFAULT_CONVERGENCE_TOLERANCE,
            "max_iterations": DEFAULT_MAX_ITERATIONS,
        },
        "interpolation": {
            "default_idw_power": DEFAULT_IDW_POWER,
            "default_kriging_range": DEFAULT_KRIGING_RANGE,
            "default_kriging_sill": DEFAULT_KRIGING_SILL,
            "default_kriging_nugget": DEFAULT_KRIGING_NUGGET,
        },
        "statistics": {
            "confidence_level": DEFAULT_CONFIDENCE_LEVEL,
            "significance_level": DEFAULT_SIGNIFICANCE_LEVEL,
            "z_score_threshold": DEFAULT_Z_SCORE_THRESHOLD,
        },
        "optimization": {
            "tolerance": DEFAULT_OPTIMIZATION_TOLERANCE,
            "max_iter": DEFAULT_OPTIMIZATION_MAX_ITER,
            "population_size": DEFAULT_OPTIMIZATION_POPULATION_SIZE,
        },
    },
    # Coordinate systems
    "coordinate_systems": {
        "wgs84_epsg": WGS84_EPSG_CODE,
        "web_mercator_epsg": WEB_MERCATOR_EPSG_CODE,
        "utm_zone_width": UTM_ZONE_WIDTH_DEGREES,
        "utm_central_meridian_offset": UTM_CENTRAL_MERIDIAN_OFFSET,
    },
    # Quality control
    "quality_control": {
        "max_coordinate_value": MAX_COORDINATE_VALUE,
        "min_coordinate_value": MIN_COORDINATE_VALUE,
        "max_distance_value": MAX_DISTANCE_VALUE,
        "min_distance_value": MIN_DISTANCE_VALUE,
        "max_value_ratio": MAX_VALUE_RATIO,
        "min_nonzero_value": MIN_NONZERO_VALUE,
    },
    # Statistical distributions
    "distributions": {"normal_z_scores": NORMAL_DISTRIBUTION_Z_SCORES},
}


def get_constant(category: str, name: str) -> Any:
    """
    Get a constant value by category and name.

    Args:
        category: Constant category
        name: Constant name

    Returns:
        Constant value

    Raises:
        KeyError: If category or name not found
    """
    if category not in CONSTANTS:
        raise KeyError(f"Constant category '{category}' not found")
    if name not in CONSTANTS[category]:
        raise KeyError(f"Constant '{name}' not found in category '{category}'")

    return CONSTANTS[category][name]


def list_constants(
    category: str | None = None,
) -> dict[str, Any] | dict[str, dict[str, Any]]:
    """
    List available constants.

    Args:
        category: Specific category to list (optional)

    Returns:
        Dictionary of constants
    """
    if category is None:
        return CONSTANTS
    else:
        if category not in CONSTANTS:
            raise KeyError(f"Constant category '{category}' not found")
        return CONSTANTS[category]


__all__ = [
    # Earth parameters
    "EARTH_RADIUS_EQUATORIAL",
    "EARTH_RADIUS_POLAR",
    "EARTH_RADIUS_MEAN",
    "EARTH_FLATTENING",
    "EARTH_ECCENTRICITY",
    "EARTH_GRAVITY_EQUATORIAL",
    "EARTH_GRAVITY_POLES",
    "EARTH_GRAVITY_MEAN",
    # Mathematical constants
    "PI",
    "EULER_GAMMA",
    "GOLDEN_RATIO",
    # Spatial analysis constants
    "DEFAULT_SPATIAL_WEIGHTS_K",
    "DEFAULT_VARIANCE_THRESHOLD",
    "DEFAULT_CONVERGENCE_TOLERANCE",
    "DEFAULT_MAX_ITERATIONS",
    # Interpolation constants
    "DEFAULT_IDW_POWER",
    "DEFAULT_KRIGING_RANGE",
    "DEFAULT_KRIGING_SILL",
    "DEFAULT_KRIGING_NUGGET",
    # Statistical constants
    "DEFAULT_CONFIDENCE_LEVEL",
    "DEFAULT_SIGNIFICANCE_LEVEL",
    "DEFAULT_Z_SCORE_THRESHOLD",
    # Coordinate system constants
    "WGS84_EPSG_CODE",
    "WEB_MERCATOR_EPSG_CODE",
    "UTM_ZONE_WIDTH_DEGREES",
    "UTM_CENTRAL_MERIDIAN_OFFSET",
    # Time constants
    "SECOND",
    "MINUTE",
    "HOUR",
    "DAY",
    "WEEK",
    "MONTH",
    "YEAR",
    # Unit conversion constants
    "METER_TO_FEET",
    "FEET_TO_METER",
    "METER_TO_YARD",
    "YARD_TO_METER",
    "METER_TO_MILE",
    "MILE_TO_METER",
    "METER_TO_KILOMETER",
    "KILOMETER_TO_METER",
    "METER_TO_NAUTICAL_MILE",
    "NAUTICAL_MILE_TO_METER",
    "SQUARE_METER_TO_SQUARE_FEET",
    "SQUARE_FEET_TO_SQUARE_METER",
    "SQUARE_METER_TO_ACRE",
    "ACRE_TO_SQUARE_METER",
    "SQUARE_METER_TO_HECTARE",
    "HECTARE_TO_SQUARE_METER",
    "CELSIUS_TO_FAHRENHEIT_OFFSET",
    "CELSIUS_TO_FAHRENHEIT_FACTOR",
    # Constants dictionary and functions
    "CONSTANTS",
    "get_constant",
    "list_constants",
]
