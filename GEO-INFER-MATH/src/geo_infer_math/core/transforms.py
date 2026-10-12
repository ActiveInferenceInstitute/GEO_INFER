"""
Coordinate Systems and Transformations Module

This module provides functions and classes for transforming coordinates between
various geographic and projected coordinate reference systems (CRS).
"""

import numpy as np
import warnings
from typing import Any, cast
from collections.abc import Callable
from dataclasses import dataclass, field
from math import pi, sin, cos, tan, sqrt, atan2, degrees, radians

# Earth parameters
EARTH_RADIUS_EQUATORIAL = 6378137.0  # WGS84 equatorial radius in meters
EARTH_RADIUS_POLAR = 6356752.314245  # WGS84 polar radius in meters
EARTH_FLATTENING = 1 / 298.257223563  # WGS84 flattening
EARTH_ECCENTRICITY = sqrt(2 * EARTH_FLATTENING - EARTH_FLATTENING**2)


@dataclass
class CRSDefinition:
    """Definition of a Coordinate Reference System."""

    name: str
    epsg_code: int | None = None
    proj_string: str | None = None
    parameters: dict[str, float] = field(default_factory=dict)


class CoordinateTransformer:
    """
    Class for transforming coordinates between different CRS.

    Supports common transformations including:
    - WGS84 ↔ UTM
    - Geographic ↔ Projected coordinates
    - Datum transformations
    """

    def __init__(
        self,
        from_crs: str | int | CRSDefinition,
        to_crs: str | int | CRSDefinition,
    ):
        """
        Initialize coordinate transformer.

        Args:
            from_crs: Source coordinate reference system
            to_crs: Target coordinate reference system
        """
        self.from_crs = self._parse_crs(from_crs)
        self.to_crs = self._parse_crs(to_crs)
        self.transformation_chain = self._build_transformation_chain()

    def _parse_crs(self, crs: str | int | CRSDefinition) -> CRSDefinition:
        """Parse CRS specification into CRSDefinition object."""
        if isinstance(crs, CRSDefinition):
            return crs
        elif isinstance(crs, int):
            # EPSG code
            return CRSDefinition(name=f"EPSG:{crs}", epsg_code=crs)
        elif isinstance(crs, str):
            # String specification
            crs_upper = crs.upper()
            if crs_upper.startswith("EPSG:"):
                try:
                    epsg_code = int(crs.split(":")[1])
                    return CRSDefinition(name=crs, epsg_code=epsg_code)
                except (ValueError, IndexError) as exc:
                    raise ValueError(f"Invalid EPSG code specification: {crs}") from exc
            elif crs_upper in ["WGS84", "EPSG:4326"]:
                return CRSDefinition(name=crs, epsg_code=4326)
            elif crs_upper in ["UTM"]:
                return CRSDefinition(name=crs)
            elif crs_upper in ["EPSG:3857", "WEBMERCATOR"]:
                return CRSDefinition(name=crs, epsg_code=3857)
            else:
                # For unknown string specifications, check if they might be valid
                # For now, we'll be permissive but could add more validation
                return CRSDefinition(name=crs, proj_string=crs)
        else:
            raise ValueError(f"Unsupported CRS specification type: {type(crs)}")

    def _build_transformation_chain(self) -> list[Callable[..., Any]]:
        """Build the transformation chain."""
        # For now, implement basic transformations
        # In a full implementation, this would use PROJ or similar libraries
        chain: list[Callable[..., tuple[float, float, float | None]]] = []

        # WGS84 to UTM
        if (
            self.from_crs.name.upper() in ["EPSG:4326", "WGS84"]
            and "UTM" in self.to_crs.name.upper()
        ):
            chain.append(self._wgs84_to_utm)

        # UTM to WGS84
        elif "UTM" in self.from_crs.name.upper() and self.to_crs.name.upper() in [
            "EPSG:4326",
            "WGS84",
        ]:
            chain.append(self._utm_to_wgs84)

        # Geographic to Web Mercator
        elif self.from_crs.name.upper() in [
            "EPSG:4326",
            "WGS84",
        ] and self.to_crs.name.upper() in ["EPSG:3857", "WEBMERCATOR"]:
            chain.append(self._geographic_to_web_mercator)

        # Web Mercator to Geographic
        elif self.from_crs.name.upper() in [
            "EPSG:3857",
            "WEBMERCATOR",
        ] and self.to_crs.name.upper() in ["EPSG:4326", "WGS84"]:
            chain.append(self._web_mercator_to_geographic)

        elif self.from_crs.name.upper() == self.to_crs.name.upper() or (
            self.from_crs.epsg_code == self.to_crs.epsg_code
            and self.from_crs.epsg_code is not None
        ):
            # Same CRS - no transformation needed
            chain.append(lambda x, y, z=None: (x, y, z))

        else:
            # Check if this is a truly unsupported transformation
            supported_from = ["EPSG:4326", "WGS84", "UTM", "EPSG:3857", "WEBMERCATOR"]
            supported_to = ["EPSG:4326", "WGS84", "UTM", "EPSG:3857", "WEBMERCATOR"]

            from_supported = any(
                s in self.from_crs.name.upper() for s in supported_from
            )
            to_supported = any(s in self.to_crs.name.upper() for s in supported_to)

            if not from_supported or not to_supported:
                raise ValueError(
                    f"Unsupported coordinate transformation from {self.from_crs.name} to {self.to_crs.name}"
                )

            # Fallback: pass-through with warning for unsupported CRS pairs
            chain.append(self._generic_transformation)

        return chain

    def transform_point(
        self, point: tuple[float, float, float | None]
    ) -> tuple[float, float, float | None]:
        """
        Transform a single point.

        Args:
            point: (x, y, z) coordinates

        Returns:
            Transformed coordinates
        """
        x, y, z = point
        result = (x, y, z)

        for transformation in self.transformation_chain:
            result = transformation(*result)

        return result

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Transform multiple points.

        Args:
            points: Array of shape (n_points, 2) or (n_points, 3)

        Returns:
            Array of transformed coordinates
        """
        if points.shape[1] == 2:
            # Add z=0 for 2D points
            points_3d = np.column_stack([points, np.zeros(len(points))])
        else:
            points_3d = points.copy()

        transformed = np.zeros_like(points_3d)

        for i, point in enumerate(points_3d):
            transformed[i] = self.transform_point(tuple(point))

        return transformed[:, : points.shape[1]]  # Return same dimensionality as input

    def _wgs84_to_utm(
        self, lon: float, lat: float, z: float | None = None
    ) -> tuple[float, float, float | None]:
        """Transform WGS84 geographic coordinates to UTM."""
        # Determine UTM zone
        zone = int((lon + 180) / 6) + 1

        # Special cases for Norway and Svalbard
        if lat >= 56.0 and lat < 64.0 and lon >= 3.0 and lon < 12.0:
            zone = 32
        elif lat >= 72.0 and lat < 84.0:
            if lon >= 0.0 and lon < 9.0:
                zone = 31
            elif lon >= 9.0 and lon < 21.0:
                zone = 33
            elif lon >= 21.0 and lon < 33.0:
                zone = 35
            elif lon >= 33.0 and lon < 42.0:
                zone = 37

        # Convert degrees to radians
        lat_rad = radians(lat)
        lon_rad = radians(lon)
        lon0_rad = radians((zone - 1) * 6 - 180 + 3)  # Central meridian

        # UTM projection parameters
        a = EARTH_RADIUS_EQUATORIAL
        e = sqrt(EARTH_FLATTENING * (2 - EARTH_FLATTENING))  # Eccentricity
        k0 = 0.9996

        # Calculate M (meridional arc)
        M = self._meridional_arc(lat_rad, a, e)

        # Calculate intermediate values
        cos_lat = cos(lat_rad)
        sin_lat = sin(lat_rad)
        tan_lat = tan(lat_rad)

        e2 = e**2
        e4 = e2**2
        e6 = e4 * e2
        e8 = e6 * e2

        nu = a / sqrt(1 - e2 * sin_lat**2)
        rho = a * (1 - e2) / (1 - e2 * sin_lat**2) ** (3 / 2)
        psi = nu / rho

        # Calculate easting
        dlon = lon_rad - lon0_rad
        A2 = 3 / 8 * (e2 + e4 / 4 + 15 * e6 / 128 - 455 * e8 / 4096)
        A4 = 15 / 256 * (e4 + 3 * e6 / 4 - 77 * e8 / 128)
        A6 = 35 / 3072 * (e6 - 41 * e8 / 32)
        A8 = -315 / 131072 * e8

        cos_dlon = cos(dlon)
        cos_2dlon = cos(2 * dlon)
        cos_4dlon = cos(4 * dlon)
        cos_6dlon = cos(6 * dlon)
        cos_8dlon = cos(8 * dlon)

        easting = (
            k0
            * nu
            * dlon
            * cos_lat
            * (1 + A2 * cos_2dlon + A4 * cos_4dlon + A6 * cos_6dlon + A8 * cos_8dlon)
            + 500000
        )

        # Calculate northing

        northing = k0 * (
            M
            + nu
            * sin_lat
            * cos_dlon
            * dlon
            * (
                1 / 2
                + (1 / 24) * (5 - tan_lat**2 + 9 * psi**2 + 4 * psi**4) * cos_2dlon
                + (1 / 720)
                * (
                    61
                    - 58 * tan_lat**2
                    + tan_lat**4
                    + 270 * psi**2
                    - 330 * psi**2 * tan_lat**2
                )
                * cos_4dlon
                + (1 / 40320)
                * (1385 - 3111 * tan_lat**2 + 543 * tan_lat**4 - tan_lat**6)
                * cos_6dlon
            )
        )

        # Add false northing for southern hemisphere
        if lat < 0:
            northing += 10000000

        return easting, northing, z

    def _meridional_arc(self, lat_rad: float, a: float, e: float) -> float:
        """Calculate meridional arc length."""
        e2 = e**2
        e4 = e2**2
        e6 = e4 * e2
        e8 = e6 * e2

        # Coefficients for meridional arc calculation
        M0 = 1 - e2 / 4 - 3 * e4 / 64 - 5 * e6 / 256 - 175 * e8 / 16384
        M2 = 3 / 8 * (e2 + e4 / 4 + 15 * e6 / 128 - 455 * e8 / 4096)
        M4 = 15 / 256 * (e4 + 3 * e6 / 4 - 77 * e8 / 128)
        M6 = 35 / 3072 * (e6 - 41 * e8 / 32)
        M8 = -315 / 131072 * e8

        sin_2lat = sin(2 * lat_rad)
        sin_4lat = sin(4 * lat_rad)
        sin_6lat = sin(6 * lat_rad)
        sin_8lat = sin(8 * lat_rad)

        return a * (
            M0 * lat_rad - M2 * sin_2lat + M4 * sin_4lat - M6 * sin_6lat + M8 * sin_8lat
        )

    def _utm_to_wgs84(
        self, easting: float, northing: float, z: float | None = None
    ) -> tuple[float, float, float | None]:
        """Transform UTM coordinates to WGS84 geographic."""
        # Remove false easting and northing
        x = easting - 500000
        y = northing

        # Determine if southern hemisphere
        southern = y < 10000000
        if southern:
            y -= 10000000

        # UTM projection parameters
        a = EARTH_RADIUS_EQUATORIAL
        f = EARTH_FLATTENING
        k0 = 0.9996

        # Calculate footprint latitude
        n = f / (2 - f)
        A = a / (1 + n) * (1 + n**2 / 4 + n**4 / 64)

        # Iterative calculation of latitude
        xi = x / (k0 * A)
        eta = y / (k0 * A)

        lat = xi
        for _ in range(10):  # Iterative refinement
            lat = (
                xi
                - n * sin(2 * lat) * cosh(eta)
                + n**2 * sin(4 * lat) * cosh(2 * eta)
                - n**3 * sin(6 * lat) * cosh(3 * eta)
            )

        # Calculate longitude
        lon = atan2(sinh(eta), cos(lat))

        # Add central meridian
        zone = int((easting - 500000) / 100000) + 30  # Approximation for zone
        lon += radians((zone - 1) * 6 - 180 + 3)

        return degrees(lon), degrees(lat), z

    def _geographic_to_web_mercator(
        self, lon: float, lat: float, z: float | None = None
    ) -> tuple[float, float, float | None]:
        """Transform geographic coordinates to Web Mercator."""
        # Web Mercator projection
        x = lon * 20037508.34 / 180
        y = degrees(atan2(sin(radians(lat)), cos(radians(lat)) - 1)) * 20037508.34 / 180

        return x, y, z

    def _web_mercator_to_geographic(
        self, x: float, y: float, z: float | None = None
    ) -> tuple[float, float, float | None]:
        """Transform Web Mercator coordinates to geographic."""
        # Inverse Web Mercator projection
        lon = x * 180 / 20037508.34
        lat = degrees(atan2(sinh(y * pi / 20037508.34), 1))

        return lon, lat, z

    def _generic_transformation(
        self, x: float, y: float, z: float | None = None
    ) -> tuple[float, float, float | None]:
        """Pass-through fallback for unrecognised CRS pairs."""
        warnings.warn(
            f"Generic transformation from {self.from_crs.name} to {self.to_crs.name} is unsupported. "
            "Returning original coordinates.",
            stacklevel=2,
        )
        return x, y, z


def geographic_to_projected(
    lon: float, lat: float, projection: str = "utm"
) -> tuple[float, float]:
    """
    Transform geographic coordinates to projected coordinates.

    Args:
        lon: Longitude in degrees
        lat: Latitude in degrees
        projection: Target projection ('utm', 'mercator', etc.)

    Returns:
        Projected coordinates (x, y)
    """
    if projection.lower() == "utm":
        transformer = CoordinateTransformer("EPSG:4326", "UTM")
        return transformer.transform_point((lon, lat, None))[:2]
    elif projection.lower() in ["mercator", "webmercator"]:
        transformer = CoordinateTransformer("EPSG:4326", "EPSG:3857")
        return transformer.transform_point((lon, lat, None))[:2]
    else:
        raise ValueError(f"Unsupported projection: {projection}")


def projected_to_geographic(
    x: float, y: float, projection: str = "utm"
) -> tuple[float, float]:
    """
    Transform projected coordinates to geographic coordinates.

    Args:
        x: Projected x coordinate
        y: Projected y coordinate
        projection: Source projection ('utm', 'mercator', etc.)

    Returns:
        Geographic coordinates (longitude, latitude)
    """
    if projection.lower() == "utm":
        transformer = CoordinateTransformer("UTM", "EPSG:4326")
        return transformer.transform_point((x, y, None))[:2]
    elif projection.lower() in ["mercator", "webmercator"]:
        transformer = CoordinateTransformer("EPSG:3857", "EPSG:4326")
        return transformer.transform_point((x, y, None))[:2]
    else:
        raise ValueError(f"Unsupported projection: {projection}")


def utm_zone_from_lon_lat(lon: float, lat: float) -> tuple[int, str]:
    """
    Determine UTM zone from longitude and latitude.

    Args:
        lon: Longitude in degrees
        lat: Latitude in degrees

    Returns:
        Tuple of (zone_number, hemisphere)
    """
    # Calculate basic zone from longitude
    zone = int((lon + 180) / 6) + 1

    # Special cases for Norway and Svalbard
    if lat >= 56.0 and lat < 64.0 and lon >= 3.0 and lon < 12.0:
        zone = 32
    elif lat >= 72.0 and lat < 84.0:
        if lon >= 0.0 and lon < 9.0:
            zone = 31
        elif lon >= 9.0 and lon < 21.0:
            zone = 33
        elif lon >= 21.0 and lon < 33.0:
            zone = 35
        elif lon >= 33.0 and lon < 42.0:
            zone = 37

    hemisphere = "N" if lat >= 0 else "S"
    return zone, hemisphere


def utm_central_meridian(zone: int) -> float:
    """
    Calculate the central meridian for a UTM zone.

    Args:
        zone: UTM zone number

    Returns:
        Central meridian in degrees
    """
    return (zone - 1) * 6 - 180 + 3


def datum_transformation(
    x: float, y: float, z: float, from_datum: str = "WGS84", to_datum: str = "NAD83"
) -> tuple[float, float, float]:
    """
    Transform coordinates between different datums.

    Args:
        x, y, z: Source coordinates
        from_datum: Source datum
        to_datum: Target datum

    Returns:
        Transformed coordinates
    """
    # Simplified datum transformation
    # In practice, this would use complex transformation parameters

    if from_datum.upper() == "WGS84" and to_datum.upper() == "NAD83":
        # WGS84 to NAD83 transformation (simplified)
        dx, dy, dz = -0.991, 1.907, 0.512  # meters (approximate)
        return x + dx, y + dy, z + dz

    elif from_datum.upper() == "NAD83" and to_datum.upper() == "WGS84":
        # NAD83 to WGS84 transformation (simplified)
        dx, dy, dz = 0.991, -1.907, -0.512  # meters (approximate)
        return x + dx, y + dy, z + dz

    else:
        warnings.warn(
            f"Datum transformation from {from_datum} to {to_datum} is unsupported. "
            "Returning original coordinates.",
            stacklevel=2,
        )
        return x, y, z


def affine_transformation(
    points: np.ndarray, matrix: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    """
    Apply affine transformation to points.

    Args:
        points: Array of points (n_points, 2) or (n_points, 3)
        matrix: Transformation matrix
        translation: Translation vector

    Returns:
        Transformed points
    """
    if points.shape[1] == 2:
        # 2D transformation
        transformed = np.dot(points, matrix.T) + translation[:2]
    else:
        # 3D transformation
        transformed = np.dot(points, matrix.T) + translation

    return cast(np.ndarray, transformed)


def rotation_matrix_2d(angle: float) -> np.ndarray:
    """
    Create 2D rotation matrix.

    Args:
        angle: Rotation angle in radians

    Returns:
        2x2 rotation matrix
    """
    cos_a = cos(angle)
    sin_a = sin(angle)
    return np.array([[cos_a, -sin_a], [sin_a, cos_a]])


def rotation_matrix_3d(axis: str, angle: float) -> np.ndarray:
    """
    Create 3D rotation matrix around specified axis.

    Args:
        axis: Rotation axis ('x', 'y', or 'z')
        angle: Rotation angle in radians

    Returns:
        3x3 rotation matrix
    """
    cos_a = cos(angle)
    sin_a = sin(angle)

    if axis.lower() == "x":
        return np.array([[1, 0, 0], [0, cos_a, -sin_a], [0, sin_a, cos_a]])
    elif axis.lower() == "y":
        return np.array([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]])
    elif axis.lower() == "z":
        return np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
    else:
        raise ValueError(f"Invalid axis: {axis}. Must be 'x', 'y', or 'z'.")


def scale_matrix_2d(sx: float, sy: float) -> np.ndarray:
    """
    Create 2D scaling matrix.

    Args:
        sx: Scale factor in x direction
        sy: Scale factor in y direction

    Returns:
        2x2 scaling matrix
    """
    return np.array([[sx, 0], [0, sy]])


def scale_matrix_3d(sx: float, sy: float, sz: float) -> np.ndarray:
    """
    Create 3D scaling matrix.

    Args:
        sx: Scale factor in x direction
        sy: Scale factor in y direction
        sz: Scale factor in z direction

    Returns:
        3x3 scaling matrix
    """
    return np.array([[sx, 0, 0], [0, sy, 0], [0, 0, sz]])


def shear_matrix_2d(shx: float, shy: float) -> np.ndarray:
    """
    Create 2D shear matrix.

    Args:
        shx: Shear factor in x direction
        shy: Shear factor in y direction

    Returns:
        2x2 shear matrix
    """
    return np.array([[1, shx], [shy, 1]])


# Utility functions for hyperbolic functions (used in UTM transformations)
def sinh(x: float) -> float:
    """Hyperbolic sine function."""
    return (exp(x) - exp(-x)) / 2


def cosh(x: float) -> float:
    """Hyperbolic cosine function."""
    return (exp(x) + exp(-x)) / 2


def tanh(x: float) -> float:
    """Hyperbolic tangent function."""
    return sinh(x) / cosh(x)


def atanh(x: float) -> float:
    """Inverse hyperbolic tangent function."""
    return 0.5 * log((1 + x) / (1 - x))


def exp(x: float) -> float:
    """Exponential function."""
    import math

    return float(math.exp(x))


def log(x: float) -> float:
    """Natural logarithm function."""
    if x <= 0:
        raise ValueError("Logarithm undefined for non-positive values")
    import math

    return float(math.log(x))


__all__ = [
    "CRSDefinition",
    "CoordinateTransformer",
    "geographic_to_projected",
    "projected_to_geographic",
    "utm_zone_from_lon_lat",
    "utm_central_meridian",
    "datum_transformation",
    "affine_transformation",
    "rotation_matrix_2d",
    "rotation_matrix_3d",
    "scale_matrix_2d",
    "scale_matrix_3d",
    "shear_matrix_2d",
    "EARTH_RADIUS_EQUATORIAL",
    "EARTH_RADIUS_POLAR",
    "EARTH_FLATTENING",
    "EARTH_ECCENTRICITY",
]
