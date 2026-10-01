"""
Validators for GEO-INFER data quality and system health.
"""

import time
import logging
import statistics
import numpy as np
import pandas as pd
from datetime import datetime, UTC
from typing import Any
from abc import ABC, abstractmethod

from ..models.types import ValidationRule

# Optional dependencies
try:
    import h3

    HAS_H3 = True
except ImportError:
    HAS_H3 = False


class BaseValidator(ABC):
    """Base class for all validators."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)
        self.validation_rules: list[ValidationRule] = []
        self.setup_rules()

    @abstractmethod
    def setup_rules(self) -> None:
        """Setup validation rules specific to this validator."""
        pass

    @abstractmethod
    def validate(self, data: Any) -> dict[str, Any]:
        """Validate data and return results."""
        pass


class DataQualityValidator(BaseValidator):
    """Validator for general data quality."""

    def setup_rules(self) -> None:
        """Setup data quality validation rules."""
        self.validation_rules = [
            ValidationRule(
                name="no_nulls_in_required_fields",
                field="*",
                rule_type="custom",
                parameters={"required_fields": ["timestamp", "value"]},
                description="Required fields must not contain null values",
            ),
            ValidationRule(
                name="timestamp_format",
                field="timestamp",
                rule_type="format",
                parameters={"format": "iso"},
                description="Timestamps must be in ISO format",
            ),
            ValidationRule(
                name="numeric_values",
                field="value",
                rule_type="custom",
                parameters={"check": "numeric"},
                description="Values must be numeric",
            ),
        ]

    def validate(self, data: pd.DataFrame | list[dict] | dict) -> dict[str, Any]:
        """Validate data quality."""
        start_time = time.time()

        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame.from_records(data)
        else:
            df = data.copy()

        errors_out: list[dict[str, Any]] = []
        warnings_out: list[dict[str, Any]] = []
        field_quality_out: dict[str, float] = {}
        validation_results: dict[str, Any] = {
            "total_records": len(df),
            "valid_records": 0,
            "validation_errors": errors_out,
            "warnings": warnings_out,
            "quality_score": 0.0,
            "field_quality": field_quality_out,
            "validation_time": 0.0,
        }

        if len(df) == 0:
            validation_results["quality_score"] = 1.0
            validation_results["validation_time"] = time.time() - start_time
            return validation_results

        error_count = 0
        warning_count = 0

        # Apply validation rules
        for rule in self.validation_rules:
            try:
                if rule.rule_type == "range":
                    result = self._validate_range(df, rule)
                elif rule.rule_type == "format":
                    result = self._validate_format(df, rule)
                elif rule.rule_type == "custom":
                    result = self._validate_custom(df, rule)
                else:
                    continue

                if not result["passed"]:
                    if rule.severity == "error":
                        errors_out.append(
                            {
                                "rule": rule.name,
                                "field": rule.field,
                                "message": result["message"],
                                "affected_records": result.get("affected_records", 0),
                            }
                        )
                        error_count += result.get("affected_records", 0)
                    elif rule.severity == "warning":
                        warnings_out.append(
                            {
                                "rule": rule.name,
                                "field": rule.field,
                                "message": result["message"],
                                "affected_records": result.get("affected_records", 0),
                            }
                        )
                        warning_count += result.get("affected_records", 0)

                # Track field-level quality
                if rule.field != "*":
                    field_quality_out[rule.field] = result.get("quality_score", 1.0)

            except Exception as e:
                self.logger.error(f"Error applying validation rule {rule.name}: {e}")

        # Calculate overall quality score
        total_issues = (
            error_count + warning_count * 0.5
        )  # Warnings count as half errors
        validation_results["quality_score"] = max(0.0, 1.0 - (total_issues / len(df)))
        validation_results["valid_records"] = len(df) - error_count
        validation_results["validation_time"] = time.time() - start_time

        self.logger.info(
            f"Data quality validation complete: {validation_results['quality_score']:.2f} score"
        )

        return validation_results

    def _validate_range(self, df: pd.DataFrame, rule: ValidationRule) -> dict[str, Any]:
        """Validate value ranges."""
        field = rule.field
        if field not in df.columns:
            return {"passed": True, "message": f"Field {field} not found"}

        min_val = rule.parameters.get("min")
        max_val = rule.parameters.get("max")

        mask = pd.Series([True] * len(df))
        if min_val is not None:
            mask &= df[field] >= min_val
        if max_val is not None:
            mask &= df[field] <= max_val

        failed_count = (~mask).sum()

        return {
            "passed": failed_count == 0,
            "message": f"{failed_count} records in {field} outside range [{min_val}, {max_val}]",
            "affected_records": failed_count,
            "quality_score": mask.sum() / len(df),
        }

    def _validate_format(
        self, df: pd.DataFrame, rule: ValidationRule
    ) -> dict[str, Any]:
        """Validate data formats."""
        field = rule.field
        if field not in df.columns:
            return {"passed": True, "message": f"Field {field} not found"}

        format_type = rule.parameters.get("format")
        failed_count = 0

        if format_type == "iso":
            # Validate ISO timestamp format
            try:
                pd.to_datetime(df[field], errors="raise")
                failed_count = 0
            except Exception:
                failed_count = len(df)

        return {
            "passed": failed_count == 0,
            "message": f"{failed_count} records in {field} with invalid {format_type} format",
            "affected_records": failed_count,
            "quality_score": (len(df) - failed_count) / len(df),
        }

    def _validate_custom(
        self, df: pd.DataFrame, rule: ValidationRule
    ) -> dict[str, Any]:
        """Apply custom validation logic."""
        if rule.name == "no_nulls_in_required_fields":
            required_fields = rule.parameters.get("required_fields", [])
            failed_count = 0

            for field in required_fields:
                if field in df.columns:
                    failed_count += df[field].isnull().sum()

            return {
                "passed": failed_count == 0,
                "message": f"{failed_count} null values in required fields",
                "affected_records": failed_count,
                "quality_score": (len(df) * len(required_fields) - failed_count)
                / (len(df) * len(required_fields)),
            }

        elif rule.name == "numeric_values":
            field = rule.field
            if field not in df.columns:
                return {"passed": True, "message": f"Field {field} not found"}

            try:
                pd.to_numeric(df[field], errors="raise")
                failed_count = 0
            except Exception:
                failed_count = (
                    len(df) - pd.to_numeric(df[field], errors="coerce").notna().sum()
                )

            return {
                "passed": failed_count == 0,
                "message": f"{failed_count} non-numeric values in {field}",
                "affected_records": failed_count,
                "quality_score": (len(df) - failed_count) / len(df),
            }

        return {"passed": True, "message": "Unknown custom rule"}


class SpatialValidator(BaseValidator):
    """Validator for spatial data quality."""

    def setup_rules(self) -> None:
        """Setup spatial validation rules."""
        self.validation_rules = [
            ValidationRule(
                name="coordinate_bounds",
                field="coordinates",
                rule_type="range",
                parameters={"lat_range": [-90, 90], "lon_range": [-180, 180]},
                description="Coordinates must be within valid Earth bounds",
            ),
            ValidationRule(
                name="h3_index_validity",
                field="h3_index",
                rule_type="custom",
                parameters={"check": "h3_valid"},
                description="H3 indices must be valid",
            ),
        ]

    def validate(self, data: pd.DataFrame | list[dict]) -> dict[str, Any]:
        """Validate spatial data."""
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()

        start_time = time.time()

        spatial_validation_out: dict[str, Any] = {
            "coordinate_validity": {},
            "h3_validation": {},
            "spatial_distribution": {},
        }
        results: dict[str, Any] = {
            "total_records": len(df),
            "spatial_validation": spatial_validation_out,
            "validation_time": 0.0,
        }

        # Validate coordinates
        if all(col in df.columns for col in ["latitude", "longitude"]):
            coord_results = self._validate_coordinates(df)
            spatial_validation_out["coordinate_validity"] = coord_results

        # Validate H3 indices
        if "h3_index" in df.columns and HAS_H3:
            h3_results = self._validate_h3_indices(df)
            spatial_validation_out["h3_validation"] = h3_results

        # Analyze spatial distribution
        if all(col in df.columns for col in ["latitude", "longitude"]):
            distribution_results = self._analyze_spatial_distribution(df)
            spatial_validation_out["spatial_distribution"] = distribution_results

        results["validation_time"] = time.time() - start_time

        return results

    def _validate_coordinates(self, df: pd.DataFrame) -> dict[str, Any]:
        """Validate coordinate values."""
        # Ensure numeric types
        try:
            df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
            df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        except (TypeError, ValueError) as exc:
            # M9-05: a coercion failure must be observable, not silently
            # swallowed back into stale non-numeric columns. Record the
            # unknown status with its reason and stop before interpreting
            # uncoerced values (mirrors _analyze_spatial_distribution).
            return {
                "status": "unknown",
                "reason": f"coordinate numeric coercion failed: {exc}",
            }

        lat_valid = (
            (df["latitude"] >= -90) & (df["latitude"] <= 90) & df["latitude"].notna()
        )
        lon_valid = (
            (df["longitude"] >= -180)
            & (df["longitude"] <= 180)
            & df["longitude"].notna()
        )

        both_valid = lat_valid & lon_valid

        return {
            "valid_coordinates": int(both_valid.sum()),
            "invalid_coordinates": int((~both_valid).sum()),
            "invalid_latitude": int((~lat_valid).sum()),
            "invalid_longitude": int((~lon_valid).sum()),
            "coordinate_quality_score": (
                float(both_valid.sum() / len(df)) if len(df) > 0 else 0.0
            ),
        }

    def _validate_h3_indices(self, df: pd.DataFrame) -> dict[str, Any]:
        """Validate H3 index values."""
        if not HAS_H3:
            return {"error": "H3 library not available"}

        valid_count = 0
        invalid_indices = []

        for idx, h3_index in enumerate(df["h3_index"]):
            try:
                # Handle non-string inputs gracefully
                if not isinstance(h3_index, str):
                    invalid_indices.append(idx)
                    continue

                if h3.is_valid_cell(h3_index):
                    valid_count += 1
                else:
                    invalid_indices.append(idx)
            except Exception:
                invalid_indices.append(idx)

        return {
            "valid_h3_indices": valid_count,
            "invalid_h3_indices": len(invalid_indices),
            "invalid_index_positions": invalid_indices[:10],
            "h3_quality_score": valid_count / len(df) if len(df) > 0 else 0.0,
        }

    def _analyze_spatial_distribution(self, df: pd.DataFrame) -> dict[str, Any]:
        """Analyze spatial distribution of data points."""
        try:
            lats = pd.to_numeric(df["latitude"], errors="coerce").values.astype(
                np.float64
            )
            lons = pd.to_numeric(df["longitude"], errors="coerce").values.astype(
                np.float64
            )
        except Exception:
            return {"error": "Could not coerce coordinates to numeric"}

        return {
            "latitude_stats": {
                "min": float(np.min(lats)),
                "max": float(np.max(lats)),
                "mean": float(np.mean(lats)),
                "std": float(np.std(lats)),
            },
            "longitude_stats": {
                "min": float(np.min(lons)),
                "max": float(np.max(lons)),
                "mean": float(np.mean(lons)),
                "std": float(np.std(lons)),
            },
            "spatial_extent": {
                "lat_range": float(np.max(lats) - np.min(lats)),
                "lon_range": float(np.max(lons) - np.min(lons)),
            },
        }


class IoTValidator(BaseValidator):
    """Validator for IoT sensor data."""

    def setup_rules(self) -> None:
        """Setup IoT validation rules."""
        self.validation_rules = [
            ValidationRule(
                name="sensor_id_format",
                field="sensor_id",
                rule_type="format",
                parameters={"pattern": r"^sensor_\d+$"},
                description="Sensor IDs must follow naming convention",
            ),
            ValidationRule(
                name="radiation_range",
                field="radiation_level",
                rule_type="range",
                parameters={"min": 0.0, "max": 100.0},
                description="Radiation levels must be within expected range",
            ),
            ValidationRule(
                name="temporal_consistency",
                field="timestamp",
                rule_type="custom",
                parameters={"max_age_hours": 24},
                description="Measurements should not be too old",
            ),
        ]

    def validate(self, data: pd.DataFrame | list[dict]) -> dict[str, Any]:
        """Validate IoT sensor data."""
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()

        start_time = time.time()

        sensor_validation_out: dict[str, Any] = {
            "data_quality": {},
            "temporal_analysis": {},
            "anomaly_detection": {},
        }
        results: dict[str, Any] = {
            "total_sensors": len(df),
            "sensor_validation": sensor_validation_out,
            "validation_time": 0.0,
        }

        # Apply data quality validation
        data_quality = DataQualityValidator(self.config, self.logger)
        sensor_validation_out["data_quality"] = data_quality.validate(df)

        # Temporal analysis
        if "timestamp" in df.columns:
            temporal_results = self._analyze_temporal_patterns(df)
            sensor_validation_out["temporal_analysis"] = temporal_results

        # Anomaly detection for radiation levels
        if "radiation_level" in df.columns:
            anomaly_results = self._detect_radiation_anomalies(df)
            sensor_validation_out["anomaly_detection"] = anomaly_results

        results["validation_time"] = time.time() - start_time

        return results

    def _analyze_temporal_patterns(self, df: pd.DataFrame) -> dict[str, Any]:
        """Analyze temporal patterns in sensor data."""
        try:
            timestamps = pd.to_datetime(df["timestamp"])
            now = datetime.now(UTC)

            # Calculate time differences
            # Handle potential tz-naive timestamps by ensuring both are aware or naive?
            # Prefer aware. If timestamps are naive, assume UTC.
            if timestamps.dt.tz is None:
                timestamps = timestamps.dt.tz_localize(UTC)

            time_diffs = [
                (now - ts).total_seconds() / 3600 for ts in timestamps
            ]  # Hours

            return {
                "newest_measurement_hours_ago": min(time_diffs) if time_diffs else 0,
                "oldest_measurement_hours_ago": max(time_diffs) if time_diffs else 0,
                "mean_age_hours": statistics.mean(time_diffs) if time_diffs else 0,
                "measurements_last_hour": sum(1 for td in time_diffs if td <= 1),
                "measurements_last_day": sum(1 for td in time_diffs if td <= 24),
                "temporal_coverage": {
                    "start": (
                        min(timestamps).isoformat() if not timestamps.empty else ""
                    ),
                    "end": max(timestamps).isoformat() if not timestamps.empty else "",
                    "span_hours": (
                        (max(timestamps) - min(timestamps)).total_seconds() / 3600
                        if not timestamps.empty
                        else 0
                    ),
                },
            }
        except Exception as e:
            return {"error": f"Temporal analysis failed: {str(e)}"}

    def _detect_radiation_anomalies(self, df: pd.DataFrame) -> dict[str, Any]:
        """Detect anomalies in radiation measurements."""
        # Ensure numeric types
        try:
            df["radiation_level"] = pd.to_numeric(
                df["radiation_level"], errors="coerce"
            )
        except Exception:
            return {"error": "Could not parse radiation levels"}

        # Drop NaNs for calculation
        valid_rad = df["radiation_level"].dropna()
        radiation_values = valid_rad.to_numpy(dtype=float)
        radiation_values = radiation_values[np.isfinite(radiation_values)]

        if len(radiation_values) < 2:
            return {"error": "Not enough valid data for anomaly detection"}

        # Statistical anomaly detection
        scale = max(float(np.max(np.abs(radiation_values))), 1.0)
        scaled_values = radiation_values / scale
        mean_scaled = float(np.mean(scaled_values, dtype=np.float64))
        std_scaled = float(np.std(scaled_values, dtype=np.float64))
        mean_rad = mean_scaled * scale
        std_rad = std_scaled * scale

        if std_rad == 0:
            z_scores = np.zeros_like(radiation_values)
        else:
            z_scores = np.abs((scaled_values - mean_scaled) / std_scaled)

        mild_anomalies = np.sum(z_scores >= 2.0)
        severe_anomalies = np.sum(z_scores >= 3.0)
        critical_anomalies = np.sum(z_scores >= 5.0)

        # Find specific anomalous measurements
        anomalous_indices = np.where(z_scores >= 2.0)[0]
        anomalous_measurements = []

        for idx in anomalous_indices[:10]:  # Limit to first 10
            anomalous_measurements.append(
                {
                    "index": int(idx),
                    "sensor_id": (
                        df.iloc[idx]["sensor_id"]
                        if "sensor_id" in df.columns
                        else f"sensor_{idx}"
                    ),
                    "radiation_level": float(radiation_values[idx]),
                    "z_score": float(z_scores[idx]),
                    "anomaly_severity": (
                        "critical"
                        if z_scores[idx] >= 5.0
                        else ("severe" if z_scores[idx] >= 3.0 else "mild")
                    ),
                }
            )

        return {
            "total_measurements": len(radiation_values),
            "anomaly_counts": {
                "mild": int(mild_anomalies),
                "severe": int(severe_anomalies),
                "critical": int(critical_anomalies),
            },
            "anomaly_rate": (
                float(mild_anomalies / len(radiation_values))
                if len(radiation_values) > 0
                else 0
            ),
            "radiation_statistics": {
                "mean": float(mean_rad),
                "std": float(std_rad),
                "min": float(np.min(radiation_values)),
                "max": float(np.max(radiation_values)),
            },
            "anomalous_measurements": anomalous_measurements,
        }


class BayesianValidator(BaseValidator):
    """Validator for Bayesian inference results."""

    def setup_rules(self) -> None:
        """Setup Bayesian validation rules."""
        self.validation_rules = [
            ValidationRule(
                name="convergence_check",
                field="convergence",
                rule_type="custom",
                parameters={"required": True},
                description="Bayesian inference must converge",
            ),
            ValidationRule(
                name="uncertainty_bounds",
                field="uncertainty",
                rule_type="range",
                parameters={"min": 0.0, "max": 10.0},
                description="Uncertainty values must be reasonable",
            ),
        ]

    def validate(self, inference_results: dict[str, Any]) -> dict[str, Any]:
        """Validate Bayesian inference results."""
        start_time = time.time()

        inference_validation_out: dict[str, Any] = {
            "convergence": False,
            "prediction_quality": {},
            "uncertainty_analysis": {},
            "model_diagnostics": {},
        }
        validation_results: dict[str, Any] = {
            "inference_validation": inference_validation_out,
            "validation_time": 0.0,
            "overall_quality": "unknown",
        }

        # Check convergence
        convergence_result = self._check_convergence(inference_results)
        inference_validation_out["convergence"] = convergence_result

        # Validate predictions
        if "predictions" in inference_results:
            prediction_quality = self._validate_predictions(
                inference_results["predictions"]
            )
            inference_validation_out["prediction_quality"] = prediction_quality

        # Analyze uncertainty
        if "uncertainty" in inference_results:
            uncertainty_analysis = self._analyze_uncertainty(
                inference_results["uncertainty"]
            )
            inference_validation_out["uncertainty_analysis"] = uncertainty_analysis

        # Model diagnostics
        model_diagnostics = self._model_diagnostics(inference_results)
        inference_validation_out["model_diagnostics"] = model_diagnostics

        # Overall quality assessment
        validation_results["overall_quality"] = self._assess_overall_quality(
            inference_validation_out
        )
        validation_results["validation_time"] = time.time() - start_time

        return validation_results

    def _check_convergence(self, results: dict[str, Any]) -> bool:
        """Check if Bayesian inference converged."""
        return bool(results.get("converged", False))

    def _validate_predictions(self, predictions: Any) -> dict[str, Any]:
        """Validate prediction values."""
        # Handle non-sequence inputs
        if not isinstance(predictions, (list, tuple, np.ndarray)):
            return {"error": "Predictions must be a list or array", "quality": "poor"}

        pred_array = np.asarray(predictions, dtype=float)

        if pred_array.ndim == 0:
            pred_array = pred_array.reshape(1)

        if len(pred_array) == 0:
            return {}

        finite_values = pred_array[np.isfinite(pred_array)]
        if finite_values.size:
            with np.errstate(over="ignore", invalid="ignore"):
                mean = float(np.mean(finite_values, dtype=np.float64))
                std = float(np.std(finite_values, dtype=np.float64))
                minimum = float(np.min(finite_values))
                maximum = float(np.max(finite_values))
        else:
            mean = std = minimum = maximum = 0.0

        return {
            "total_predictions": len(pred_array),
            "finite_predictions": int(np.sum(np.isfinite(pred_array))),
            "nan_predictions": int(np.sum(np.isnan(pred_array))),
            "infinite_predictions": int(np.sum(np.isinf(pred_array))),
            "negative_predictions": int(np.sum(pred_array < 0)),
            "prediction_stats": {
                "mean": mean,
                "std": std,
                "min": minimum,
                "max": maximum,
            },
        }

    def _analyze_uncertainty(self, uncertainty: Any) -> dict[str, Any]:
        """Analyze uncertainty estimates."""
        # Handle non-sequence inputs
        if not isinstance(uncertainty, (list, tuple, np.ndarray)):
            return {"error": "Uncertainty must be a list or array", "quality": "poor"}

        unc_array = np.asarray(uncertainty, dtype=float)

        if unc_array.ndim == 0:
            unc_array = unc_array.reshape(1)

        if len(unc_array) == 0:
            return {}

        finite_values = unc_array[np.isfinite(unc_array)]
        if finite_values.size:
            with np.errstate(over="ignore", invalid="ignore"):
                mean = float(np.mean(finite_values, dtype=np.float64))
                std = float(np.std(finite_values, dtype=np.float64))
                minimum = float(np.min(finite_values))
                maximum = float(np.max(finite_values))
        else:
            mean = std = minimum = maximum = 0.0

        return {
            "total_uncertainty_estimates": len(unc_array),
            "finite_uncertainty": int(np.sum(np.isfinite(unc_array))),
            "uncertainty_stats": {
                "mean": mean,
                "std": std,
                "min": minimum,
                "max": maximum,
            },
            "uncertainty_quality": (
                "good"
                if np.all(unc_array >= 0) and np.all(np.isfinite(unc_array))
                else "poor"
            ),
        }

    def _model_diagnostics(self, results: dict[str, Any]) -> dict[str, Any]:
        """Perform model diagnostics."""
        diagnostics = {
            "has_predictions": "predictions" in results,
            "has_uncertainty": "uncertainty" in results,
            "has_confidence_intervals": "confidence_intervals" in results,
            "processing_time": results.get("processing_time", 0),
            "prior_mean": results.get("prior_mean"),
            "length_scale": results.get("length_scale"),
        }

        return diagnostics

    def _assess_overall_quality(self, validation_results: dict[str, Any]) -> str:
        """Assess overall quality of Bayesian inference."""
        convergence = validation_results.get("convergence", False)

        prediction_quality = validation_results.get("prediction_quality", {})
        finite_predictions = prediction_quality.get("finite_predictions", 0)
        total_predictions = prediction_quality.get("total_predictions", 1)

        uncertainty_quality = validation_results.get("uncertainty_analysis", {}).get(
            "uncertainty_quality", "poor"
        )

        # Quality assessment logic
        if (
            convergence
            and finite_predictions / total_predictions > 0.95
            and uncertainty_quality == "good"
        ):
            return "excellent"
        elif convergence and finite_predictions / total_predictions > 0.9:
            return "good"
        elif convergence:
            return "acceptable"
        else:
            return "poor"


class PerformanceValidator:
    """Validator for system performance metrics."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.thresholds = self.config.get("validation", {})

    def validate_performance(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Validate system performance metrics."""
        performance_out: dict[str, Any] = {
            "timing_checks": {},
            "throughput_checks": {},
            "resource_checks": {},
            "overall_performance": "unknown",
        }
        results: dict[str, Any] = {
            "performance_validation": performance_out,
        }

        # Check timing metrics
        max_inference_time = self.thresholds.get("max_inference_time", "30s")
        max_time_seconds = self._parse_time_string(str(max_inference_time))

        actual_times = []
        for key, value in metrics.items():
            if "time" in key.lower() and isinstance(value, (int, float)):
                actual_times.append(value)

        if actual_times:
            max_actual_time = max(actual_times)
            performance_out["timing_checks"] = {
                "max_time_threshold": max_time_seconds,
                "max_actual_time": max_actual_time,
                "timing_acceptable": max_actual_time <= max_time_seconds,
            }

        # Check throughput
        min_accuracy = self.thresholds.get("min_prediction_accuracy", 0.85)
        if "accuracy" in metrics:
            performance_out["throughput_checks"] = {
                "accuracy_threshold": min_accuracy,
                "actual_accuracy": metrics["accuracy"],
                "accuracy_acceptable": metrics["accuracy"] >= min_accuracy,
            }

        # Resource usage checks
        max_memory = self.thresholds.get("max_memory_usage", "4GB")
        max_memory_bytes = self._parse_memory_string(str(max_memory))

        if "memory_usage" in metrics:
            memory_acceptable = metrics["memory_usage"] <= max_memory_bytes
            performance_out["resource_checks"] = {
                "memory_threshold": max_memory_bytes,
                "actual_memory": metrics["memory_usage"],
                "memory_acceptable": memory_acceptable,
            }

        # Overall assessment
        timing_ok = performance_out["timing_checks"].get("timing_acceptable", True)
        accuracy_ok = performance_out["throughput_checks"].get(
            "accuracy_acceptable", True
        )
        memory_ok = performance_out["resource_checks"].get("memory_acceptable", True)

        if timing_ok and accuracy_ok and memory_ok:
            performance_out["overall_performance"] = "acceptable"
        else:
            performance_out["overall_performance"] = "unacceptable"

        return results

    def _parse_time_string(self, time_str: Any) -> float:
        """Parse time string like '30s' to seconds."""
        if isinstance(time_str, (int, float)):
            return float(time_str)
        if isinstance(time_str, str):
            if time_str.endswith("s"):
                return float(time_str[:-1])
            elif time_str.endswith("m"):
                return float(time_str[:-1]) * 60
            elif time_str.endswith("h"):
                return float(time_str[:-1]) * 3600
            else:
                return float(time_str)
        return float(time_str)

    def _parse_memory_string(self, memory_str: Any) -> float:
        """Parse memory string like '4GB' to bytes."""
        if isinstance(memory_str, (int, float)):
            return float(memory_str)
        if isinstance(memory_str, str):
            if memory_str.endswith("GB"):
                return float(memory_str[:-2]) * 1024**3
            elif memory_str.endswith("MB"):
                return float(memory_str[:-2]) * 1024**2
            elif memory_str.endswith("KB"):
                return float(memory_str[:-2]) * 1024
            else:
                return float(memory_str)
        return float(memory_str)


class QualityController:
    """Main quality control system that coordinates all validators."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)

        # Initialize validators
        self.data_validator = DataQualityValidator(config, logger)
        self.spatial_validator = SpatialValidator(config, logger)
        self.iot_validator = IoTValidator(config, logger)
        self.bayesian_validator = BayesianValidator(config, logger)
        self.performance_validator = PerformanceValidator(config)

    def run_comprehensive_validation(
        self,
        sensor_data: pd.DataFrame | None = None,
        spatial_results: dict[str, Any] | None = None,
        inference_results: dict[str, Any] | None = None,
        performance_metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run comprehensive validation across all components."""
        start_time = time.time()

        components_out: list[str] = []
        validation_summary: dict[str, Any] = {
            "validation_timestamp": datetime.now(UTC).isoformat(),
            "components_validated": components_out,
            "overall_results": {},
            "total_validation_time": 0.0,
        }

        # Validate sensor data
        if sensor_data is not None:
            self.logger.info("Running IoT sensor data validation")
            iot_results = self.iot_validator.validate(sensor_data)
            validation_summary["iot_validation"] = iot_results
            components_out.append("iot_data")

        # Validate spatial results
        if spatial_results is not None:
            self.logger.info("Running spatial validation")
            spatial_val_results = self.spatial_validator.validate(
                spatial_results.get("h3_aggregated_data", pd.DataFrame())
            )
            validation_summary["spatial_validation"] = spatial_val_results
            components_out.append("spatial_data")

        # Validate Bayesian inference
        if inference_results is not None:
            self.logger.info("Running Bayesian inference validation")
            bayes_results = self.bayesian_validator.validate(inference_results)
            validation_summary["bayesian_validation"] = bayes_results
            components_out.append("bayesian_inference")

        # Validate performance
        if performance_metrics is not None:
            self.logger.info("Running performance validation")
            perf_results = self.performance_validator.validate_performance(
                performance_metrics
            )
            validation_summary["performance_validation"] = perf_results
            components_out.append("performance")

        # Overall assessment
        validation_summary["overall_results"] = self._assess_overall_system_quality(
            validation_summary
        )
        validation_summary["total_validation_time"] = time.time() - start_time

        overall_res = validation_summary["overall_results"]
        self.logger.info(
            f"Comprehensive validation complete: {overall_res.get('system_quality') if isinstance(overall_res, dict) else 'unknown'}"
        )

        return validation_summary

    def _assess_overall_system_quality(
        self, validation_results: dict[str, Any]
    ) -> dict[str, Any]:
        """Assess overall system quality based on all validation results."""
        quality_scores = []

        # Extract quality scores from different validations
        if "iot_validation" in validation_results:
            iot_quality = validation_results["iot_validation"]["sensor_validation"][
                "data_quality"
            ]["quality_score"]
            quality_scores.append(iot_quality)

        if "bayesian_validation" in validation_results:
            bayes_quality = validation_results["bayesian_validation"]["overall_quality"]
            # Convert quality to numeric score
            quality_map = {
                "excellent": 1.0,
                "good": 0.8,
                "acceptable": 0.6,
                "poor": 0.3,
                "unknown": 0.5,
            }
            quality_scores.append(quality_map.get(bayes_quality, 0.5))

        if "performance_validation" in validation_results:
            perf_quality = validation_results["performance_validation"][
                "performance_validation"
            ]["overall_performance"]
            perf_score = 1.0 if perf_quality == "acceptable" else 0.3
            quality_scores.append(perf_score)

        # Calculate overall system quality
        if quality_scores:
            overall_score = statistics.mean(quality_scores)
            if overall_score >= 0.9:
                system_quality = "excellent"
            elif overall_score >= 0.7:
                system_quality = "good"
            elif overall_score >= 0.5:
                system_quality = "acceptable"
            else:
                system_quality = "poor"
        else:
            overall_score = 0.0
            system_quality = "unknown"

        return {
            "system_quality": system_quality,
            "overall_score": overall_score,
            "component_scores": quality_scores,
            "components_tested": len(quality_scores),
            "recommendation": self._get_quality_recommendation(system_quality),
        }

    def _get_quality_recommendation(self, quality: str) -> str:
        """Get recommendation based on system quality."""
        recommendations = {
            "excellent": "System performing optimally. Continue monitoring.",
            "good": "System performing well. Minor optimizations may be beneficial.",
            "acceptable": "System functional but improvements recommended.",
            "poor": "System requires immediate attention and fixes.",
            "unknown": "Insufficient data for quality assessment.",
        }
        return recommendations.get(quality, "No recommendation available.")


def run_full_system_test(
    sensor_data: pd.DataFrame,
    spatial_results: dict[str, Any],
    inference_results: dict[str, Any],
    performance_metrics: dict[str, Any],
    config: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Run a complete system test with all validators."""
    qc = QualityController(config, logger)
    return qc.run_comprehensive_validation(
        sensor_data=sensor_data,
        spatial_results=spatial_results,
        inference_results=inference_results,
        performance_metrics=performance_metrics,
    )
