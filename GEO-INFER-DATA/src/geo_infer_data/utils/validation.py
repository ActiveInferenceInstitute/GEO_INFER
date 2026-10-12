"""
Data validation utilities for GEO-INFER-DATA.

This module provides comprehensive validation utilities for geospatial data
including format validation, schema validation, and data integrity checks.
"""

import logging
from typing import Any
from collections.abc import Sequence
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import StrEnum

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.validation import explain_validity

from ..models.schemas import (
    DataQualityReport,
    DatasetMetadata,
    QualityCheck,
    QualityStatus,
)


logger = logging.getLogger(__name__)


def _now_for_series(series: pd.Series) -> pd.Timestamp:
    """Return a comparison timestamp matching a datetime series timezone."""
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        return pd.Timestamp.now(tz=series.dt.tz)
    return pd.Timestamp.now()


def _coerce_for_series(value: datetime, series: pd.Series) -> pd.Timestamp:
    """Coerce metadata timestamps to the timezone of a datetime series."""
    timestamp = pd.Timestamp(value)
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        if timestamp.tzinfo is None:
            return timestamp.tz_localize(series.dt.tz)
        return timestamp.tz_convert(series.dt.tz)
    if timestamp.tzinfo is not None:
        return timestamp.tz_convert(None)
    return timestamp


def scan_geometry_validity(geometries: Any) -> list[tuple[Any, str, Any]]:
    """Scan geometry values for validity.

    Shared primitive used by both :class:`GeospatialValidator` implementations
    (utils and core). Accepts any iterable of ``(index, geometry)`` pairs, e.g.
    ``GeoDataFrame.geometry.items()``.

    Returns:
        List of ``(index, reason, geometry)`` tuples, one per input geometry,
        where reason is ``"null"``, ``"invalid"``, or ``"ok"``.
    """
    issues: list[tuple[Any, str, Any]] = []
    for idx, geom in geometries:
        if geom is None:
            issues.append((idx, "null", geom))
        elif not geom.is_valid:
            issues.append((idx, "invalid", geom))
        else:
            issues.append((idx, "ok", geom))
    return issues


def wgs84_bounds_issues(bounds: Sequence[float]) -> list[str]:
    """Return the WGS84 bound-check types violated by ``[lon, lat, lon, lat]`` bounds."""
    min_lon, min_lat, max_lon, max_lat = bounds[:4]
    issues: list[str] = []
    if not (-180 <= min_lon <= 180) or not (-180 <= max_lon <= 180):
        issues.append("invalid_longitude_bounds")
    if not (-90 <= min_lat <= 90) or not (-90 <= max_lat <= 90):
        issues.append("invalid_latitude_bounds")
    return issues


def count_out_of_range(values: Any, low: float, high: float) -> int:
    """Count entries of a numeric series outside ``[low, high]``."""
    return int(sum(1 for value in values.dropna() if not (low <= value <= high)))


def count_future_dates(series: pd.Series) -> int:
    """Count timestamps strictly later than now (timezone-aware safe)."""
    return int((series > _now_for_series(series)).sum())


def has_mixed_types(series: pd.Series) -> bool:
    """Return True when a column contains more than one Python value type."""
    return len(series.dropna().apply(type).unique()) > 1


class ValidationRule(StrEnum):
    """Available validation rules."""

    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    VALIDITY = "validity"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    FORMAT = "format"
    SCHEMA = "schema"


@dataclass
class ValidationConfig:
    """Configuration for data validation."""

    validation_rules: list[str] | None = field(default=None)
    quality_threshold: float = 0.8
    strict_mode: bool = False
    real_time_monitoring: bool = True
    custom_rules: dict[str, Any] | None = field(default=None)

    def __post_init__(self) -> None:
        if self.validation_rules is None:
            self.validation_rules = [rule.value for rule in ValidationRule]


class GeospatialValidator:
    """
    Comprehensive geospatial data validation.

    This class provides validation utilities for geospatial data including
    format validation, schema validation, geometric validation, and
    data integrity checks.

    Examples:
        >>> validator = GeospatialValidator()
        >>>
        >>> # Validate geospatial data
        >>> result = await validator.validate_data(geodataframe)
        >>> print(f"Validation score: {result.score:.2f}")
        >>>
        >>> # Validate specific aspect
        >>> geometry_check = validator.validate_geometries(geodataframe)
        >>> print(f"Geometry validation: {geometry_check.status}")
    """

    def __init__(self, config: ValidationConfig | None = None) -> None:
        self.config = config if config is not None else ValidationConfig()
        self.validation_rules = {
            "geometry": self._validate_geometry,
            "coordinates": self._validate_coordinates,
            "attributes": self._validate_attributes,
            "metadata": self._validate_metadata,
            "temporal": self._validate_temporal,
            "spatial_reference": self._validate_spatial_reference,
        }
        self._report_rules = {
            "completeness": self._check_completeness,
            "accuracy": self._check_accuracy,
            "consistency": self._check_consistency,
            "validity": self._check_validity,
            "temporal": self._check_temporal,
            "spatial": self._check_spatial,
            "format": self._check_format,
            "schema": self._check_schema,
        }

        logger.info("Initialized GeospatialValidator")

    async def validate_data(self, data: Any) -> QualityCheck:
        """
        Validate geospatial data comprehensively.

        Args:
            data: Data to validate

        Returns:
            Quality check result
        """
        logger.debug("Starting comprehensive data validation")

        issues = []
        score = 1.0

        # Run all validation checks
        for validation_name, validation_func in self.validation_rules.items():
            try:
                check_result = await validation_func(data)
                if not check_result["valid"]:
                    issues.extend(check_result["issues"])
                    score -= check_result["penalty"]
            except Exception as e:
                logger.error(f"Validation {validation_name} failed: {e}")
                issues.append(
                    {
                        "type": "validation_error",
                        "message": f"{validation_name} validation failed: {e}",
                        "severity": "high",
                    }
                )
                score -= 0.2

        # Determine status
        if score >= 0.8:
            status = QualityStatus.PASS
        elif score >= 0.5:
            status = QualityStatus.WARNING
        else:
            status = QualityStatus.FAIL

        return QualityCheck(score=max(0.0, score), status=status, issues=issues)

    async def _validate_geometry(self, data: Any) -> dict[str, Any]:
        """Validate geometry data."""
        issues = []
        penalty = 0.0

        if isinstance(data, gpd.GeoDataFrame) and "geometry" in data.columns:
            total_count = len(data)
            scanned = scan_geometry_validity(data.geometry.items())
            problem_count = 0
            for idx, reason, geom in scanned:
                if reason == "ok":
                    continue
                problem_count += 1
                if reason == "null":
                    issues.append(
                        {
                            "type": "null_geometry",
                            "message": f"Null geometry at index {idx}",
                            "severity": "high",
                        }
                    )
                else:
                    issues.append(
                        {
                            "type": "invalid_geometry",
                            "message": f"Invalid geometry at index {idx}: {explain_validity(geom)}",
                            "severity": "high",
                        }
                    )

            if problem_count:
                invalid_percent = problem_count / total_count
                penalty = min(0.5, invalid_percent * 2)  # Up to 50% penalty

        return {"valid": penalty < 0.1, "issues": issues, "penalty": penalty}

    async def _validate_coordinates(self, data: Any) -> dict[str, Any]:
        """Validate coordinate data."""
        issues = []
        penalty = 0.0

        if isinstance(data, pd.DataFrame):
            # Check for latitude/longitude columns
            lat_cols = [col for col in data.columns if "lat" in col.lower()]
            lon_cols = [col for col in data.columns if "lon" in col.lower()]

            for lat_col in lat_cols:
                if count_out_of_range(data[lat_col], -90, 90) > 0:
                    issues.append(
                        {
                            "type": "invalid_latitude",
                            "message": f"Invalid latitude values in {lat_col}",
                            "severity": "high",
                        }
                    )
                    penalty += 0.2

            for lon_col in lon_cols:
                if count_out_of_range(data[lon_col], -180, 180) > 0:
                    issues.append(
                        {
                            "type": "invalid_longitude",
                            "message": f"Invalid longitude values in {lon_col}",
                            "severity": "high",
                        }
                    )
                    penalty += 0.2

        elif isinstance(data, gpd.GeoDataFrame):
            # Check geometry bounds
            if not data.empty:
                bounds = data.total_bounds
                if len(bounds) == 4:
                    for issue_type in wgs84_bounds_issues(bounds):
                        issues.append(
                            {
                                "type": issue_type,
                                "message": (
                                    "Invalid longitude bounds in geometry"
                                    if issue_type == "invalid_longitude_bounds"
                                    else "Invalid latitude bounds in geometry"
                                ),
                                "severity": "high",
                            }
                        )
                        penalty += 0.2

        return {"valid": penalty < 0.1, "issues": issues, "penalty": penalty}

    async def _validate_attributes(self, data: Any) -> dict[str, Any]:
        """Validate data attributes."""
        issues = []
        penalty = 0.0

        if isinstance(data, (pd.DataFrame, gpd.GeoDataFrame)):
            # Check for required columns
            if len(data.columns) == 0:
                issues.append(
                    {
                        "type": "no_columns",
                        "message": "Dataset has no columns",
                        "severity": "critical",
                    }
                )
                penalty += 0.5

            # Check for data types
            for col in data.columns:
                if data[col].dtype == "object" and has_mixed_types(data[col]):
                    issues.append(
                        {
                            "type": "mixed_types",
                            "message": f"Mixed data types in column {col}",
                            "severity": "low",
                        }
                    )
                    penalty += 0.1

        return {"valid": penalty < 0.1, "issues": issues, "penalty": penalty}

    async def _validate_metadata(self, data: Any) -> dict[str, Any]:
        """Validate metadata consistency."""
        # Metadata validation logic
        return {"valid": True, "issues": [], "penalty": 0.0}

    async def _validate_temporal(self, data: Any) -> dict[str, Any]:
        """Validate temporal data."""
        issues = []
        penalty = 0.0

        if isinstance(data, (pd.DataFrame, gpd.GeoDataFrame)):
            # Find datetime columns
            datetime_cols = data.select_dtypes(
                include=["datetime", "datetimetz"]
            ).columns

            for col in datetime_cols:
                # Check for future dates
                future_count = count_future_dates(data[col])
                if future_count:
                    issues.append(
                        {
                            "type": "future_dates",
                            "message": f"Future dates detected in {col}: {future_count} records",
                            "severity": "medium",
                        }
                    )
                    penalty += 0.1

                # Check for unreasonable date ranges
                if not data[col].empty:
                    date_range = data[col].max() - data[col].min()
                    if date_range.days > 365 * 50:  # More than 50 years
                        issues.append(
                            {
                                "type": "unreasonable_date_range",
                                "message": f"Unreasonable date range in {col}: {date_range.days} days",
                                "severity": "low",
                            }
                        )
                        penalty += 0.1

        return {"valid": penalty < 0.1, "issues": issues, "penalty": penalty}

    async def _validate_spatial_reference(self, data: Any) -> dict[str, Any]:
        """Validate spatial reference system."""
        issues = []
        penalty = 0.0

        if isinstance(data, gpd.GeoDataFrame):
            if data.crs is None:
                issues.append(
                    {
                        "type": "missing_crs",
                        "message": "Missing coordinate reference system",
                        "severity": "medium",
                    }
                )
                penalty += 0.2
            else:
                # Check CRS validity
                try:
                    crs_code = data.crs.to_epsg()
                    if crs_code is None:
                        issues.append(
                            {
                                "type": "invalid_crs",
                                "message": "Invalid coordinate reference system",
                                "severity": "medium",
                            }
                        )
                        penalty += 0.2
                except Exception as e:
                    issues.append(
                        {
                            "type": "crs_error",
                            "message": f"CRS validation error: {e}",
                            "severity": "medium",
                        }
                    )
                    penalty += 0.2

        return {"valid": penalty < 0.1, "issues": issues, "penalty": penalty}

    def validate_geometries(self, geodataframe: gpd.GeoDataFrame) -> QualityCheck:
        """
        Validate geometries in a GeoDataFrame.

        Args:
            geodataframe: GeoDataFrame to validate

        Returns:
            Geometry validation result
        """
        issues = []
        score = 1.0

        if (
            not isinstance(geodataframe, gpd.GeoDataFrame)
            or "geometry" not in geodataframe.columns
        ):
            return QualityCheck(
                score=0.0,
                status=QualityStatus.FAIL,
                issues=[
                    {"type": "missing_geometry", "message": "No geometry column found"}
                ],
            )
        # Check geometry validity
        total = max(len(geodataframe), 1)
        scanned = scan_geometry_validity(geodataframe.geometry.items())
        problem_count = 0
        invalid_coordinates = 0
        for idx, reason, geom in scanned:
            if reason in ("null", "invalid"):
                problem_count += 1
                if reason == "null":
                    issues.append(
                        {
                            "type": "null_geometry",
                            "message": f"Null geometry at index {idx}",
                            "severity": "high",
                        }
                    )
                else:
                    issues.append(
                        {
                            "type": "invalid_geometry",
                            "message": f"Invalid geometry at index {idx}",
                            "severity": "high",
                        }
                    )
            elif reason == "ok" and wgs84_bounds_issues(geom.bounds):
                invalid_coordinates += 1

        if problem_count:
            invalid_percent = problem_count / total
            score -= min(0.8, invalid_percent * 2)
        if invalid_coordinates:
            coord_percent = invalid_coordinates / total
            issues.append(
                {
                    "type": "invalid_coordinates",
                    "message": f"Coordinates outside WGS84 bounds: {coord_percent:.2%}",
                    "severity": "high",
                }
            )
            score -= min(0.9, coord_percent * 1.5)
        if geodataframe.crs is None:
            issues.append(
                {
                    "type": "missing_crs",
                    "message": "Missing coordinate reference system",
                }
            )
            score -= 0.2
        # Check geometry types
        geom_types = geodataframe.geometry.type.unique()
        if len(geom_types) > 5:  # Too many geometry types
            issues.append(
                {
                    "type": "mixed_geometry_types",
                    "message": f"Too many geometry types: {list(geom_types)}",
                    "severity": "low",
                }
            )
            score -= 0.1

        status = (
            QualityStatus.PASS
            if score >= 0.8
            else QualityStatus.WARNING
            if score >= 0.5
            else QualityStatus.FAIL
        )

        return QualityCheck(score=max(0.0, score), status=status, issues=issues)

    def validate_coordinates(
        self, data: pd.DataFrame | gpd.GeoDataFrame
    ) -> QualityCheck:
        """
        Validate coordinate data.

        Args:
            data: DataFrame or GeoDataFrame with coordinate data

        Returns:
            Coordinate validation result
        """
        issues = []
        score = 1.0

        if isinstance(data, gpd.GeoDataFrame):
            # Validate geometry bounds
            try:
                bounds = data.total_bounds
                if len(bounds) == 4:
                    for issue_type in wgs84_bounds_issues(bounds):
                        issues.append(
                            {
                                "type": issue_type,
                                "message": "Invalid "
                                + (
                                    "longitude"
                                    if issue_type == "invalid_longitude_bounds"
                                    else "latitude"
                                )
                                + " bounds",
                                "severity": "high",
                            }
                        )
                        score -= 0.3

            except Exception as e:
                issues.append(
                    {
                        "type": "bounds_calculation_error",
                        "message": f"Error calculating bounds: {e}",
                        "severity": "high",
                    }
                )
                score -= 0.2

        elif isinstance(data, pd.DataFrame):
            # Check for lat/lon columns
            lat_cols = [col for col in data.columns if "lat" in col.lower()]
            lon_cols = [col for col in data.columns if "lon" in col.lower()]

            if not lat_cols and not lon_cols:
                return QualityCheck(
                    score=0.0,
                    status=QualityStatus.FAIL,
                    issues=[
                        {
                            "type": "missing_coordinates",
                            "message": "latitude and longitude columns are required",
                        }
                    ],
                )

            for lat_col in lat_cols:
                invalid_count = count_out_of_range(data[lat_col], -90, 90)

                if invalid_count > 0:
                    invalid_percent = invalid_count / len(data)
                    issues.append(
                        {
                            "type": "invalid_latitude",
                            "message": f"Invalid latitude values in {lat_col}: {invalid_percent:.2%}",
                            "severity": "high",
                        }
                    )
                    score -= min(0.9, invalid_percent * 4.0)

            for lon_col in lon_cols:
                invalid_count = count_out_of_range(data[lon_col], -180, 180)

                if invalid_count > 0:
                    invalid_percent = invalid_count / len(data)
                    issues.append(
                        {
                            "type": "invalid_longitude",
                            "message": f"Invalid longitude values in {lon_col}: {invalid_percent:.2%}",
                            "severity": "high",
                        }
                    )
                    score -= min(0.9, invalid_percent * 4.0)

        status = (
            QualityStatus.PASS
            if score >= 0.8
            else QualityStatus.WARNING
            if score >= 0.5
            else QualityStatus.FAIL
        )

        return QualityCheck(score=max(0.0, score), status=status, issues=issues)

    def validate_temporal_data(
        self, data: pd.DataFrame | gpd.GeoDataFrame
    ) -> QualityCheck:
        """
        Validate temporal data.

        Args:
            data: DataFrame or GeoDataFrame with temporal data

        Returns:
            Temporal validation result
        """
        issues = []
        score = 1.0

        if isinstance(data, (pd.DataFrame, gpd.GeoDataFrame)):
            # Find datetime columns
            datetime_cols = data.select_dtypes(
                include=["datetime", "datetimetz"]
            ).columns

            for col in datetime_cols:
                # Check for future dates
                future_count = count_future_dates(data[col])
                if future_count:
                    future_percent = future_count / len(data)

                    issues.append(
                        {
                            "type": "future_dates",
                            "message": f"Future dates in {col}: {future_percent:.2%}",
                            "severity": "medium",
                        }
                    )
                    score -= 0.2

                # Check for chronological order
                if not data[col].is_monotonic_increasing:
                    issues.append(
                        {
                            "type": "non_chronological",
                            "message": f"Non-chronological order in {col}",
                            "severity": "low",
                        }
                    )
                    score -= 0.1

        status = (
            QualityStatus.PASS
            if score >= 0.8
            else QualityStatus.WARNING
            if score >= 0.5
            else QualityStatus.FAIL
        )

        return QualityCheck(score=max(0.0, score), status=status, issues=issues)

    async def build_quality_report(
        self, data: Any, metadata: DatasetMetadata | None = None
    ) -> DataQualityReport:
        """
        Validate geospatial data comprehensively and build a quality report.

        Args:
            data: Data to validate
            metadata: Dataset metadata

        Returns:
            Comprehensive quality report
        """
        logger.info("Starting comprehensive data validation")

        dataset_id = metadata.title if metadata else "unknown_dataset"
        checks = {}
        overall_score = 0.0

        # Run all configured validation checks
        configured_rules = self.config.validation_rules
        rules: list[str] = (
            configured_rules
            if configured_rules is not None
            else [rule.value for rule in ValidationRule]
        )
        for rule_name in rules:
            if rule_name in self._report_rules:
                try:
                    check_result = await self._report_rules[rule_name](data, metadata)
                    checks[rule_name] = check_result

                    # Update overall score
                    if check_result.status == QualityStatus.PASS:
                        overall_score += check_result.score
                    elif check_result.status == QualityStatus.WARNING:
                        overall_score += (
                            check_result.score * 0.8
                        )  # Weight warnings lower
                    else:
                        overall_score += (
                            check_result.score * 0.5
                        )  # Weight failures lower

                except Exception as e:
                    logger.error(f"Validation rule {rule_name} failed: {e}")
                    checks[rule_name] = QualityCheck(
                        score=0.0,
                        status=QualityStatus.FAIL,
                        issues=[{"type": "validation_error", "message": str(e)}],
                    )

        # Calculate overall score
        if checks:
            overall_score /= len(checks)
            overall_score = min(
                overall_score, min(check.score for check in checks.values())
            )

        # Generate recommendations
        recommendations = self._generate_recommendations(checks, overall_score)

        quality_report = DataQualityReport(
            dataset_id=dataset_id,
            overall_score=overall_score,
            checks=checks,
            recommendations=recommendations,
            assessment_method=rules,
            validation_rules=list(rules),
        )

        logger.info(
            f"Data validation completed with overall score: {overall_score:.2f}"
        )
        return quality_report

    async def _check_completeness(
        self, data: Any, metadata: DatasetMetadata | None
    ) -> QualityCheck:
        """Check data completeness."""
        issues = []
        score = 1.0

        if isinstance(data, (pd.DataFrame, gpd.GeoDataFrame)):
            if data.empty:
                issues.append(
                    {
                        "type": "empty_dataset",
                        "message": "Dataset is empty",
                        "severity": "critical",
                    }
                )
                return QualityCheck(score=0.0, status=QualityStatus.FAIL, issues=issues)

            # Check for missing values
            missing_percent = data.isnull().sum().sum() / (
                data.shape[0] * data.shape[1]
            )

            if missing_percent > 0.1:  # More than 10% missing
                issues.append(
                    {
                        "type": "high_missing_values",
                        "message": f"High percentage of missing values: {missing_percent:.2%}",
                        "severity": "high",
                    }
                )
                score -= min(0.9, missing_percent * 1.5)
            elif missing_percent > 0.05:  # More than 5% missing
                issues.append(
                    {
                        "type": "moderate_missing_values",
                        "message": f"Moderate percentage of missing values: {missing_percent:.2%}",
                        "severity": "medium",
                    }
                )
                score -= 0.1

        elif isinstance(data, np.ndarray):
            # Check for NaN values in arrays
            nan_count = np.isnan(data).sum()
            total_count = data.size

            if nan_count > 0:
                nan_percent = nan_count / total_count
                issues.append(
                    {
                        "type": "array_nan_values",
                        "message": f"Array contains NaN values: {nan_percent:.2%}",
                        "severity": "medium",
                    }
                )
                score -= 0.2

        status = (
            QualityStatus.PASS
            if score >= 0.8
            else QualityStatus.WARNING
            if score >= 0.5
            else QualityStatus.FAIL
        )

        return QualityCheck(score=max(0.0, score), status=status, issues=issues)

    async def _check_accuracy(
        self, data: Any, metadata: DatasetMetadata | None
    ) -> QualityCheck:
        """Check data accuracy."""
        issues = []
        score = 1.0

        if isinstance(data, (pd.DataFrame, gpd.GeoDataFrame)):
            # Check for outliers
            numeric_columns = data.select_dtypes(include=[np.number]).columns

            for col in numeric_columns:
                if col in data.columns:
                    values = data[col].dropna()
                    if len(values) > 0:
                        # Simple outlier detection using IQR
                        Q1 = values.quantile(0.25)
                        Q3 = values.quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR

                        outliers = values[
                            (values < lower_bound) | (values > upper_bound)
                        ]
                        outlier_percent = len(outliers) / len(values)

                        if outlier_percent > 0.1:  # More than 10% outliers
                            issues.append(
                                {
                                    "type": "high_outliers",
                                    "message": f"High percentage of outliers in {col}: {outlier_percent:.2%}",
                                    "severity": "medium",
                                }
                            )
                            score -= 0.2

            # Check coordinate accuracy if geospatial
            if isinstance(data, gpd.GeoDataFrame) and "geometry" in data.columns:
                # Check for invalid coordinates
                invalid_coords = sum(
                    1
                    for _idx, reason, _geom in scan_geometry_validity(
                        data.geometry.items()
                    )
                    if reason != "ok"
                )

                if invalid_coords > 0:
                    invalid_percent = invalid_coords / len(data)
                    issues.append(
                        {
                            "type": "invalid_geometries",
                            "message": f"Invalid geometries found: {invalid_percent:.2%}",
                            "severity": "high",
                        }
                    )
                    score -= 0.3

        status = (
            QualityStatus.PASS
            if score >= 0.8
            else QualityStatus.WARNING
            if score >= 0.5
            else QualityStatus.FAIL
        )

        return QualityCheck(score=max(0.0, score), status=status, issues=issues)

    async def _check_consistency(
        self, data: Any, metadata: DatasetMetadata | None
    ) -> QualityCheck:
        """Check data consistency."""
        issues = []
        score = 1.0

        if isinstance(data, (pd.DataFrame, gpd.GeoDataFrame)):
            # Check data types consistency
            for col in data.columns:
                if col == "geometry":
                    continue
                if has_mixed_types(data[col]):
                    issues.append(
                        {
                            "type": "mixed_data_types",
                            "message": f"Mixed data types in column {col}",
                            "severity": "low",
                        }
                    )
                    score -= 0.1

            # Check for duplicate records. Geometry objects can expose pandas
            # internals that are not duplicate-check friendly, so use scalar
            # attributes for this consistency rule.
            duplicate_data = data.drop(columns=["geometry"], errors="ignore")
            duplicates = duplicate_data.duplicated().sum()
            if duplicates > 0:
                duplicate_percent = duplicates / len(data)
                issues.append(
                    {
                        "type": "duplicate_records",
                        "message": f"Duplicate records found: {duplicate_percent:.2%}",
                        "severity": "medium",
                    }
                )
                score -= 0.2

            # Check temporal consistency if datetime columns exist
            datetime_cols = data.select_dtypes(
                include=["datetime", "datetimetz"]
            ).columns
            for col in datetime_cols:
                if col in data.columns:
                    # Check for chronological order
                    if not data[col].is_monotonic_increasing:
                        issues.append(
                            {
                                "type": "non_chronological",
                                "message": f"Non-chronological order in {col}",
                                "severity": "low",
                            }
                        )
                        score -= 0.1

        status = (
            QualityStatus.PASS
            if score >= 0.8
            else QualityStatus.WARNING
            if score >= 0.5
            else QualityStatus.FAIL
        )

        return QualityCheck(score=max(0.0, score), status=status, issues=issues)

    async def _check_validity(
        self, data: Any, metadata: DatasetMetadata | None
    ) -> QualityCheck:
        """Check data validity."""
        issues = []
        score = 1.0

        if isinstance(data, (pd.DataFrame, gpd.GeoDataFrame)):
            # Check for valid values in each column
            for col in data.columns:
                if data[col].dtype in ["object", "string"]:
                    # Check for suspicious string values
                    suspicious_values = (
                        data[col]
                        .dropna()
                        .astype(str)
                        .str.contains(r"[^\w\s\-.,()&]", regex=True)
                    )
                    if suspicious_values.any():
                        issues.append(
                            {
                                "type": "suspicious_characters",
                                "message": f"Suspicious characters in column {col}",
                                "severity": "low",
                            }
                        )
                        score -= 0.1

                elif pd.api.types.is_numeric_dtype(data[col]):
                    # Check for infinite values
                    if np.isinf(data[col]).any():
                        issues.append(
                            {
                                "type": "infinite_values",
                                "message": f"Infinite values in column {col}",
                                "severity": "high",
                            }
                        )
                        score -= 0.3

        elif isinstance(data, np.ndarray):
            # Check for valid array values
            if np.isinf(data).any():
                issues.append(
                    {
                        "type": "infinite_values",
                        "message": "Infinite values in array",
                        "severity": "high",
                    }
                )
                score -= 0.3

            if np.isnan(data).any():
                issues.append(
                    {
                        "type": "nan_values",
                        "message": "NaN values in array",
                        "severity": "medium",
                    }
                )
                score -= 0.2

        status = (
            QualityStatus.PASS
            if score >= 0.8
            else QualityStatus.WARNING
            if score >= 0.5
            else QualityStatus.FAIL
        )

        return QualityCheck(score=max(0.0, score), status=status, issues=issues)

    async def _check_temporal(
        self, data: Any, metadata: DatasetMetadata | None
    ) -> QualityCheck:
        """Check temporal validity."""
        issues = []
        score = 1.0

        if isinstance(data, (pd.DataFrame, gpd.GeoDataFrame)):
            # Find datetime columns
            datetime_cols = data.select_dtypes(
                include=["datetime", "datetimetz"]
            ).columns

            for col in datetime_cols:
                if col in data.columns:
                    # Check for future dates (might be data entry errors)
                    future_count = count_future_dates(data[col])
                    if future_count:
                        future_percent = future_count / len(data)
                        issues.append(
                            {
                                "type": "future_dates",
                                "message": f"Future dates in {col}: {future_percent:.2%}",
                                "severity": "medium",
                            }
                        )
                        score -= 0.2

                    # Check for unreasonable date ranges
                    date_range = data[col].max() - data[col].min()
                    if date_range > timedelta(days=365 * 100):  # More than 100 years
                        issues.append(
                            {
                                "type": "unreasonable_date_range",
                                "message": f"Unreasonable date range in {col}: {date_range.days} days",
                                "severity": "low",
                            }
                        )
                        score -= 0.1

        # Check temporal extent if metadata provided
        if metadata and metadata.temporal:
            # Validate temporal extent consistency
            if isinstance(data, (pd.DataFrame, gpd.GeoDataFrame)):
                datetime_cols = data.select_dtypes(
                    include=["datetime", "datetimetz"]
                ).columns
                if len(datetime_cols) > 0:
                    series = data[datetime_cols[0]]
                    data_min = series.min()
                    data_max = series.max()

                    if data_min < _coerce_for_series(
                        metadata.temporal.start, series
                    ) or data_max > _coerce_for_series(metadata.temporal.end, series):
                        issues.append(
                            {
                                "type": "temporal_extent_mismatch",
                                "message": "Data temporal extent exceeds metadata bounds",
                                "severity": "medium",
                            }
                        )
                        score -= 0.2

        status = (
            QualityStatus.PASS
            if score >= 0.8
            else QualityStatus.WARNING
            if score >= 0.5
            else QualityStatus.FAIL
        )

        return QualityCheck(score=max(0.0, score), status=status, issues=issues)

    async def _check_spatial(
        self, data: Any, metadata: DatasetMetadata | None
    ) -> QualityCheck:
        """Check spatial validity."""
        issues = []
        score = 1.0

        if isinstance(data, gpd.GeoDataFrame) and "geometry" in data.columns:
            # Check geometry validity
            invalid_geoms = 0
            invalid_coords = 0
            for geom in data.geometry:
                if geom is None or not geom.is_valid:
                    invalid_geoms += 1
                    continue
                min_lon, min_lat, max_lon, max_lat = geom.bounds
                if min_lon < -180 or max_lon > 180 or min_lat < -90 or max_lat > 90:
                    invalid_coords += 1

            if invalid_geoms > 0:
                invalid_percent = invalid_geoms / len(data)
                issues.append(
                    {
                        "type": "invalid_geometries",
                        "message": f"Invalid geometries: {invalid_percent:.2%}",
                        "severity": "high",
                    }
                )
                score -= min(0.9, invalid_percent * 4.0)

            if invalid_coords > 0:
                invalid_percent = invalid_coords / len(data)
                issues.append(
                    {
                        "type": "invalid_coordinates",
                        "message": f"Invalid coordinates: {invalid_percent:.2%}",
                        "severity": "high",
                    }
                )
                score -= min(0.9, invalid_percent * 4.0)

            # Check coordinate system consistency
            if data.crs is None:
                issues.append(
                    {
                        "type": "missing_crs",
                        "message": "Missing coordinate reference system",
                        "severity": "medium",
                    }
                )
                score -= 0.2

            # Check for geometries outside expected bounds
            if metadata and metadata.spatial:
                bounds = metadata.spatial.bbox
                if len(bounds) >= 4:
                    min_lon, min_lat, max_lon, max_lat = bounds[:4]

                    # Check if geometries are within bounds
                    out_of_bounds = 0
                    for geom in data.geometry:
                        if geom and geom.bounds:
                            geom_min_lon, geom_min_lat, geom_max_lon, geom_max_lat = (
                                geom.bounds
                            )
                            if (
                                geom_max_lon < min_lon
                                or geom_min_lon > max_lon
                                or geom_max_lat < min_lat
                                or geom_min_lat > max_lat
                            ):
                                out_of_bounds += 1

                    if out_of_bounds > 0:
                        oob_percent = out_of_bounds / len(data)
                        issues.append(
                            {
                                "type": "out_of_bounds",
                                "message": f"Geometries outside bounds: {oob_percent:.2%}",
                                "severity": "medium",
                            }
                        )
                        score -= 0.2

        elif (
            isinstance(data, (pd.DataFrame,))
            and "latitude" in data.columns
            and "longitude" in data.columns
        ):
            # Check coordinate validity for lat/lon data
            invalid_coords = 0

            for _, row in data.iterrows():
                lat = row.get("latitude")
                lon = row.get("longitude")

                if lat is not None and lon is not None:
                    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                        invalid_coords += 1
                else:
                    invalid_coords += 1

            if invalid_coords > 0:
                invalid_percent = invalid_coords / len(data)
                issues.append(
                    {
                        "type": "invalid_coordinates",
                        "message": f"Invalid coordinates: {invalid_percent:.2%}",
                        "severity": "high",
                    }
                )
                score -= min(0.9, invalid_percent * 4.0)

        status = (
            QualityStatus.PASS
            if score >= 0.8
            else QualityStatus.WARNING
            if score >= 0.5
            else QualityStatus.FAIL
        )

        return QualityCheck(score=max(0.0, score), status=status, issues=issues)

    async def _check_format(
        self, data: Any, metadata: DatasetMetadata | None
    ) -> QualityCheck:
        """Check data format validity."""
        issues = []
        score = 1.0

        # Check data type consistency
        if hasattr(data, "dtypes"):
            # Pandas-like data
            for col, dtype in data.dtypes.items():
                if pd.api.types.is_object_dtype(dtype):
                    # Check for mixed types in object columns
                    unique_types = data[col].dropna().apply(type).unique()
                    if len(unique_types) > 3:  # Arbitrary threshold
                        issues.append(
                            {
                                "type": "mixed_types",
                                "message": f"Multiple types in column {col}",
                                "severity": "low",
                            }
                        )
                        score -= 0.1

        status = (
            QualityStatus.PASS
            if score >= 0.8
            else QualityStatus.WARNING
            if score >= 0.5
            else QualityStatus.FAIL
        )

        return QualityCheck(score=max(0.0, score), status=status, issues=issues)

    async def _check_schema(
        self, data: Any, metadata: DatasetMetadata | None
    ) -> QualityCheck:
        """Check schema validity."""
        issues: list[dict[str, Any]] = []
        score = 1.0

        # Schema validation logic
        # This would check against expected schema definitions

        status = (
            QualityStatus.PASS
            if score >= 0.8
            else QualityStatus.WARNING
            if score >= 0.5
            else QualityStatus.FAIL
        )

        return QualityCheck(score=max(0.0, score), status=status, issues=issues)

    def _generate_recommendations(
        self, checks: dict[str, QualityCheck], overall_score: float
    ) -> list[str]:
        """Generate improvement recommendations."""
        recommendations = []

        if overall_score < 0.8:
            recommendations.append("Overall data quality is below acceptable threshold")

        for check_name, check in checks.items():
            if check.status == QualityStatus.FAIL:
                recommendations.append(f"Address {check_name} validation failures")
            elif check.status == QualityStatus.WARNING:
                recommendations.append(f"Review {check_name} warnings")

        return recommendations
