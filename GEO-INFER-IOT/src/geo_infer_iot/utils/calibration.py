"""
Sensor Calibration Utilities

This module provides utilities for sensor calibration, drift detection,
and calibration management for IoT sensor networks.
"""

import logging
from typing import Any
from datetime import datetime, timedelta, UTC
import numpy as np
from dataclasses import dataclass
from geo_infer_time.core.timestamps import normalize_timestamp

from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Result of a calibration operation."""

    success: bool
    calibration_parameters: dict[str, Any]
    calibration_error: float
    timestamp: datetime
    method: str
    notes: str = ""

    def __post_init__(self) -> None:
        self.timestamp = normalize_timestamp(self.timestamp)


class SensorCalibration:
    """
    Sensor calibration and drift detection utilities.

    Provides methods for:
    - Sensor calibration using reference measurements
    - Drift detection and correction
    - Calibration schedule management
    - Multi-point calibration procedures
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.calibration_history: list[dict[str, Any]] = []
        self.drift_models: dict[str, Any] = {}

        # Default calibration parameters
        self.default_params = {
            "calibration_methods": ["linear", "polynomial", "lookup_table"],
            "drift_detection_threshold": 0.05,  # 5% deviation
            "calibration_interval_days": 90,
            "reference_tolerance": 0.02,  # 2% tolerance for reference values
        }

        logger.info("SensorCalibration initialized")

    def calibrate_sensor(
        self,
        sensor_id: str,
        reference_data: list[dict],
        calibration_method: str = "linear",
    ) -> CalibrationResult:
        """
        Calibrate a sensor using reference measurements.

        Args:
            sensor_id: ID of the sensor to calibrate
            reference_data: List of reference measurements with known true values
            calibration_method: Calibration method to use

        Returns:
            CalibrationResult with calibration parameters and quality metrics
        """
        try:
            if len(reference_data) < 2:
                return CalibrationResult(
                    success=False,
                    calibration_parameters={},
                    calibration_error=float("inf"),
                    timestamp=datetime.now(UTC),
                    method=calibration_method,
                    notes="Insufficient reference data",
                )

            # Extract sensor readings and reference values
            sensor_values = [d["sensor_value"] for d in reference_data]
            reference_values = [d["reference_value"] for d in reference_data]

            # Perform calibration based on method
            if calibration_method == "linear":
                calibration_params = self._linear_calibration(
                    sensor_values, reference_values
                )
            elif calibration_method == "polynomial":
                calibration_params = self._polynomial_calibration(
                    sensor_values, reference_values
                )
            else:
                # Default to linear if method not supported
                calibration_params = self._linear_calibration(
                    sensor_values, reference_values
                )

            # Calculate calibration error
            calibrated_values = self._apply_calibration(
                sensor_values, calibration_params, calibration_method
            )
            calibration_error = np.mean(
                [
                    abs(calibrated - ref) / ref if ref != 0 else abs(calibrated - ref)
                    for calibrated, ref in zip(calibrated_values, reference_values)
                ]
            )

            # Store calibration result
            result = CalibrationResult(
                success=True,
                calibration_parameters=calibration_params,
                calibration_error=calibration_error,
                timestamp=datetime.now(UTC),
                method=calibration_method,
                notes=f"Calibration completed with {len(reference_data)} reference points",
            )

            self.calibration_history.append(
                {
                    "sensor_id": sensor_id,
                    "timestamp": result.timestamp,
                    "result": result,
                }
            )

            return result

        except Exception as e:
            logger.error(f"Error calibrating sensor {sensor_id}: {e}")
            return CalibrationResult(
                success=False,
                calibration_parameters={},
                calibration_error=float("inf"),
                timestamp=datetime.now(UTC),
                method=calibration_method,
                notes=f"Calibration failed: {str(e)}",
            )

    def _linear_calibration(
        self, sensor_values: list[float], reference_values: list[float]
    ) -> dict:
        """Perform linear calibration: reference = slope * sensor + offset."""
        if len(sensor_values) < 2:
            return {"slope": 1.0, "offset": 0.0}

        # Linear regression
        slope, offset, r_value, p_value, std_err = stats.linregress(
            sensor_values, reference_values
        )

        return {
            "slope": slope,
            "offset": offset,
            "r_squared": r_value**2,
            "std_error": std_err,
        }

    def _polynomial_calibration(
        self, sensor_values: list[float], reference_values: list[float]
    ) -> dict:
        """Perform polynomial calibration (2nd order)."""
        if len(sensor_values) < 3:
            # A quadratic fit needs at least three points; fall back to linear
            return self._linear_calibration(sensor_values, reference_values)

        try:
            # Fit 2nd order polynomial
            coeffs = np.polyfit(sensor_values, reference_values, 2)
            ref_arr = np.array(reference_values)

            # Calculate R-squared
            predicted = np.polyval(coeffs, sensor_values)
            ss_res = np.sum((ref_arr - predicted) ** 2)
            ss_tot = np.sum((ref_arr - np.mean(ref_arr)) ** 2)
            r_squared = float(1 - (ss_res / ss_tot) if ss_tot != 0 else 0)

            return {
                "coefficients": coeffs.tolist(),
                "degree": 2,
                "r_squared": r_squared,
            }
        except Exception as e:
            logger.warning(f"Polynomial calibration failed, using linear: {e}")
            return self._linear_calibration(sensor_values, reference_values)

    def _apply_calibration(
        self, sensor_values: list[float], params: dict, method: str
    ) -> list[float]:
        """Apply calibration parameters to sensor values."""
        if method == "linear":
            slope = params.get("slope", 1.0)
            offset = params.get("offset", 0.0)
            return [slope * val + offset for val in sensor_values]
        elif method == "polynomial" and "coefficients" in params:
            coeffs = params["coefficients"]
            return [float(np.polyval(coeffs, val)) for val in sensor_values]
        else:
            return sensor_values  # No calibration

    def detect_drift(
        self, sensor_id: str, recent_measurements: list[dict], reference_baseline: dict
    ) -> dict:
        """
        Detect sensor drift using recent measurements and baseline.

        Args:
            sensor_id: ID of the sensor
            recent_measurements: Recent measurements for drift analysis
            reference_baseline: Baseline measurements for comparison

        Returns:
            Dictionary with drift analysis results
        """
        try:
            if len(recent_measurements) < 5 or len(reference_baseline) < 5:
                return {
                    "drift_detected": False,
                    "drift_score": 0.0,
                    "confidence": 0.0,
                    "notes": "Insufficient data for drift analysis",
                }

            # Extract values
            recent_values = [m["value"] for m in recent_measurements]
            baseline_values = [m["value"] for m in reference_baseline]

            # Statistical comparison
            recent_mean = np.mean(recent_values)
            baseline_mean = np.mean(baseline_values)
            recent_std = np.std(recent_values)
            baseline_std = np.std(baseline_values)

            # Calculate drift metrics
            mean_drift = (
                abs(recent_mean - baseline_mean) / baseline_mean
                if baseline_mean != 0
                else 0
            )
            std_drift = (
                abs(recent_std - baseline_std) / baseline_std
                if baseline_std != 0
                else 0
            )

            # Combined drift score
            drift_score = (mean_drift + std_drift) / 2

            # Statistical significance test
            t_stat, p_value = stats.ttest_ind(recent_values, baseline_values)
            confidence = 1 - p_value if p_value < 0.05 else 0.5

            # Determine if drift is significant
            threshold = self.config.get("drift_threshold", 0.05)
            drift_detected = bool(drift_score > threshold)

            return {
                "drift_detected": drift_detected,
                "drift_score": drift_score,
                "confidence": confidence,
                "mean_drift": mean_drift,
                "std_drift": std_drift,
                "recent_mean": recent_mean,
                "baseline_mean": baseline_mean,
                "recent_std": recent_std,
                "baseline_std": baseline_std,
                "threshold": threshold,
                "analysis_timestamp": datetime.now(UTC).isoformat(),
            }

        except Exception as e:
            logger.error(f"Error detecting drift for sensor {sensor_id}: {e}")
            return {
                "drift_detected": False,
                "drift_score": 0.0,
                "confidence": 0.0,
                "error": str(e),
            }

    def generate_calibration_schedule(self, sensor_inventory: list[dict]) -> list[dict]:
        """
        Generate calibration schedule for sensor inventory.

        Args:
            sensor_inventory: List of sensor information with calibration history

        Returns:
            List of calibration schedule entries
        """
        schedule = []
        calibration_interval = self.config.get("calibration_interval_days", 90)

        for sensor in sensor_inventory:
            sensor_id = sensor.get("sensor_id")

            # Get last calibration date
            last_calibration = None
            for record in self.calibration_history:
                if record["sensor_id"] == sensor_id:
                    if (
                        last_calibration is None
                        or record["timestamp"] > last_calibration
                    ):
                        last_calibration = record["timestamp"]

            # Calculate next calibration date
            if last_calibration:
                next_calibration = last_calibration + timedelta(
                    days=calibration_interval
                )
            else:
                # First calibration - schedule immediately
                next_calibration = datetime.now(UTC)

            # Determine priority based on time since last calibration
            days_since_calibration = (
                (datetime.now(UTC) - last_calibration).days
                if last_calibration
                else float("inf")
            )
            priority = self._calculate_calibration_priority(
                days_since_calibration, calibration_interval
            )

            schedule.append(
                {
                    "sensor_id": sensor_id,
                    "sensor_type": sensor.get("sensor_type", "unknown"),
                    "last_calibration": last_calibration.isoformat()
                    if last_calibration
                    else None,
                    "next_calibration": next_calibration.isoformat(),
                    "days_until_due": max(
                        0, (next_calibration - datetime.now(UTC)).days
                    ),
                    "priority": priority,
                    "calibration_method": sensor.get("calibration_method", "linear"),
                }
            )

        # Sort by priority and due date
        schedule.sort(key=lambda x: (x["priority"], x["days_until_due"]))

        return schedule

    def _calculate_calibration_priority(
        self, days_since_calibration: float, interval_days: int
    ) -> str:
        """Calculate calibration priority based on time since last calibration."""
        if days_since_calibration > interval_days * 1.2:  # 20% overdue
            return "critical"
        elif days_since_calibration > interval_days:
            return "high"
        elif days_since_calibration > interval_days * 0.8:  # 80% of interval
            return "medium"
        else:
            return "low"

    def validate_calibration_data(self, calibration_data: list[dict]) -> dict:
        """
        Validate calibration data quality and completeness.

        Args:
            calibration_data: List of calibration measurements

        Returns:
            Validation results and quality metrics
        """
        if len(calibration_data) < 3:
            return {
                "valid": False,
                "issues": ["Insufficient calibration points (minimum 3 required)"],
                "quality_score": 0.0,
            }

        issues = []
        quality_score = 1.0

        # Check for duplicate sensor values
        sensor_values = [d["sensor_value"] for d in calibration_data]
        if len(set(sensor_values)) < len(sensor_values) * 0.8:  # Less than 80% unique
            issues.append("Too many duplicate sensor values")
            quality_score *= 0.8

        # Check value range
        value_range = max(sensor_values) - min(sensor_values)
        if value_range == 0:
            issues.append("All sensor values are identical")
            quality_score *= 0.5
        elif value_range < 0.1 * np.mean(sensor_values):  # Range too small
            issues.append(
                "Sensor value range may be too small for reliable calibration"
            )
            quality_score *= 0.9

        # Check for outliers in reference values
        reference_values = [d["reference_value"] for d in calibration_data]
        z_scores = np.abs(stats.zscore(reference_values))
        outlier_count = np.sum(z_scores > 3)

        if outlier_count > 0:
            issues.append(f"{outlier_count} potential outliers in reference values")
            quality_score *= max(0.5, 1.0 - 0.1 * outlier_count)

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "quality_score": quality_score,
            "calibration_points": len(calibration_data),
            "value_range": value_range,
            "unique_sensor_values": len(set(sensor_values)),
        }

    def get_calibration_report(
        self, sensor_id: str | None = None, time_window_days: int = 30
    ) -> dict:
        """
        Generate calibration report for sensor(s).

        Args:
            sensor_id: Specific sensor ID, or None for all sensors
            time_window_days: Time window for the report

        Returns:
            Comprehensive calibration report
        """
        cutoff_date = datetime.now(UTC) - timedelta(days=time_window_days)

        # Filter calibration history
        relevant_calibrations = []
        for record in self.calibration_history:
            if record["timestamp"] >= cutoff_date:
                if sensor_id is None or record["sensor_id"] == sensor_id:
                    relevant_calibrations.append(record)

        if not relevant_calibrations:
            return {
                "report_period_days": time_window_days,
                "total_calibrations": 0,
                "sensors_calibrated": 0,
                "message": "No calibrations found in the specified time window",
            }

        # Analyze calibration results
        successful_calibrations = [
            r for r in relevant_calibrations if r["result"].success
        ]
        failed_calibrations = [
            r for r in relevant_calibrations if not r["result"].success
        ]

        success_rate = len(successful_calibrations) / len(relevant_calibrations)

        # Average calibration error
        errors = [r["result"].calibration_error for r in successful_calibrations]
        avg_error = np.mean(errors) if errors else 0.0

        # Sensors calibrated
        calibrated_sensors = list(set(r["sensor_id"] for r in relevant_calibrations))

        return {
            "report_period_days": time_window_days,
            "total_calibrations": len(relevant_calibrations),
            "successful_calibrations": len(successful_calibrations),
            "failed_calibrations": len(failed_calibrations),
            "success_rate": success_rate,
            "average_calibration_error": avg_error,
            "sensors_calibrated": len(calibrated_sensors),
            "calibration_methods_used": list(
                set(r["result"].method for r in successful_calibrations)
            ),
            "generated_at": datetime.now(UTC).isoformat(),
        }
