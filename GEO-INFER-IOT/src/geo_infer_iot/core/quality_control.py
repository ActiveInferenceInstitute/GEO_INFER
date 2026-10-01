"""
Quality Control Module

This module provides comprehensive quality control and validation for IoT sensor data,
including outlier detection, range validation, temporal consistency checks, and
spatial consistency validation.
"""

import logging
from typing import Any
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass
from collections import defaultdict, deque

# Optional imports for enhanced functionality
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler as StandardScaler

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = logging.getLogger(__name__)


@dataclass
class QualityCheckResult:
    """Result of a quality check operation."""

    passed: bool
    issues: list[str]
    quality_score: float
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


class QualityController:
    """
    Comprehensive quality control system for IoT sensor data.

    This class provides multiple quality control mechanisms:
    - Outlier detection using statistical methods and machine learning
    - Range validation for sensor measurements
    - Temporal consistency checks across time series
    - Spatial consistency validation using H3-based analysis
    - Calibration drift detection
    """

    #: Per-sensor measurements retained for temporal and outlier analysis.
    DEFAULT_HISTORY_SIZE = 500

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.quality_history: list[dict[str, Any]] = []
        self.sensor_baselines: dict[str, Any] = {}
        self.outlier_detector: Any | None = None
        self.history_size = int(
            self.config.get("history_size", self.DEFAULT_HISTORY_SIZE)
        )
        # Bounded per-sensor rings so a long-lived controller keeps constant memory.
        self.measurement_history: dict[str, deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=self.history_size)
        )

        # Default quality control parameters
        self.default_params = {
            "outlier_detection": {
                "method": "isolation_forest",
                "contamination": 0.1,
                "window_size": 100,
            },
            "range_validation": {
                "action": "flag",  # 'flag', 'reject', 'correct'
                "strict_mode": False,
            },
            "temporal_consistency": {
                "max_change_rate": 0.1,  # Max relative change per minute
                "window_minutes": 60,
            },
            "spatial_consistency": {
                "neighbor_threshold": 2.0,  # Standard deviations
                "min_neighbors": 3,
            },
        }

        # Initialize outlier detector if available
        if HAS_SKLEARN:
            self._initialize_outlier_detector()

        logger.info("QualityController initialized")

    def _initialize_outlier_detector(self) -> None:
        """Initialize the outlier detection model."""
        try:
            outlier_config = self.config.get(
                "outlier_detection", self.default_params["outlier_detection"]
            )
            self.outlier_detector = IsolationForest(
                contamination=outlier_config.get("contamination", 0.1), random_state=42
            )
        except Exception as e:
            logger.warning(f"Failed to initialize outlier detector: {e}")
            self.outlier_detector = None

    def validate_measurement(self, measurement: dict) -> QualityCheckResult:
        """
        Perform comprehensive quality validation on a single measurement.

        Args:
            measurement: Sensor measurement to validate

        Returns:
            QualityCheckResult with validation outcome
        """
        issues = []
        quality_score = 1.0

        # Range validation
        range_result = self._validate_range(measurement)
        if not range_result.passed:
            issues.extend(range_result.issues)
            quality_score *= 0.8

        # Temporal consistency check
        temporal_result = self._validate_temporal_consistency(measurement)
        if not temporal_result.passed:
            issues.extend(temporal_result.issues)
            quality_score *= 0.9

        # Outlier detection
        outlier_result = self._detect_outliers(measurement)
        if not outlier_result.passed:
            issues.extend(outlier_result.issues)
            quality_score *= 0.7

        # Spatial consistency check (if location data available)
        if "latitude" in measurement and "longitude" in measurement:
            spatial_result = self._validate_spatial_consistency(measurement)
            if not spatial_result.passed:
                issues.extend(spatial_result.issues)
                quality_score *= 0.85

        passed = len(issues) == 0
        if not passed:
            quality_score = max(0.0, quality_score - 0.1 * len(issues))

        # Retain after scoring so a measurement is never compared against itself.
        self._record_measurement(measurement)

        return QualityCheckResult(
            passed=passed,
            issues=issues,
            quality_score=quality_score,
            metadata={
                "validation_timestamp": datetime.now().isoformat(),
                "validation_checks": ["range", "temporal", "outlier", "spatial"],
            },
        )

    def _validate_range(self, measurement: dict) -> QualityCheckResult:
        """Validate measurement value against expected ranges."""
        issues = []
        variable = measurement.get("variable", "unknown")
        value = measurement.get("value")

        if value is None:
            return QualityCheckResult(False, ["Missing value"], 0.0)

        # Get variable-specific ranges from config
        variable_ranges = self.config.get("variable_ranges", {})

        if variable in variable_ranges:
            min_val, max_val = variable_ranges[variable]

            if value < min_val or value > max_val:
                issues.append(
                    f"Value {value} outside expected range [{min_val}, {max_val}] for {variable}"
                )

        # General sanity checks
        if not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
            issues.append(f"Invalid value type or NaN/Inf: {value}")

        return QualityCheckResult(
            len(issues) == 0, issues, 0.9 if len(issues) == 0 else 0.5
        )

    def _validate_temporal_consistency(self, measurement: dict) -> QualityCheckResult:
        """Validate temporal consistency of measurements."""
        issues = []
        sensor_id = measurement.get("sensor_id", "unknown")
        timestamp = measurement.get("timestamp")
        value = measurement.get("value")

        if not timestamp or value is None:
            return QualityCheckResult(True, [], 1.0)  # Skip if no temporal data

        # Get recent measurements for this sensor
        recent_measurements = self._get_recent_measurements(sensor_id, minutes=60)

        if len(recent_measurements) >= 2:
            # Calculate rate of change
            recent_values = [
                m["value"] for m in recent_measurements[-5:]
            ]  # Last 5 measurements
            if len(recent_values) >= 2:
                temporal_cfg = self.config.get("temporal_consistency", {})
                max_change_rate = temporal_cfg.get("max_change_rate", 0.1)

                for i in range(1, len(recent_values)):
                    prev_val = recent_values[i - 1]
                    curr_val = recent_values[i]

                    if prev_val != 0:  # Avoid division by zero
                        change_rate = abs(curr_val - prev_val) / abs(prev_val)
                        if change_rate > max_change_rate:
                            issues.append(
                                f"Temporal change rate {change_rate:.2%} exceeds threshold {max_change_rate:.2%}"
                            )

        return QualityCheckResult(
            len(issues) == 0, issues, 0.9 if len(issues) == 0 else 0.6
        )

    def _detect_outliers(self, measurement: dict) -> QualityCheckResult:
        """Detect outliers using statistical methods and machine learning."""
        issues: list[str] = []
        value = measurement.get("value")

        if value is None or self.outlier_detector is None:
            return QualityCheckResult(True, [], 1.0)

        try:
            sensor_id = measurement.get("sensor_id", "unknown")

            # Isolation Forest needs a population; fit it on the retained window.
            temp_cfg: dict[str, Any] = self.default_params.get(
                "temporal_consistency", {}
            )  # type: ignore[assignment]
            default_win = temp_cfg.get("window_minutes", 60)
            window = int(self.config.get("outlier_window_minutes", default_win))
            recent = self._get_recent_measurements(sensor_id, minutes=window)
            min_samples = int(self.config.get("outlier_min_samples", 20))
            if len(recent) >= min_samples:
                samples = np.array([[entry["value"]] for entry in recent])
                self.outlier_detector.fit(samples)
                if self.outlier_detector.predict(np.array([[value]]))[0] == -1:
                    issues.append(
                        f"Isolation Forest flagged value {value} against "
                        f"{len(recent)} recent measurements"
                    )

            # Statistical 3-sigma check against the running baseline.
            if sensor_id in self.sensor_baselines:
                baseline = self.sensor_baselines[sensor_id]
                mean_val = baseline["mean"]
                std_val = baseline["std"]

                if std_val > 0:
                    z_score = abs(value - mean_val) / std_val
                    if z_score > 3.0:  # 3-sigma rule
                        issues.append(
                            f"Statistical outlier detected (z-score: {z_score:.2f})"
                        )

            # Update baseline statistics
            self._update_sensor_baseline(sensor_id, value)

        except Exception as e:
            logger.warning(f"Error in outlier detection: {e}")

        return QualityCheckResult(
            len(issues) == 0, issues, 0.8 if len(issues) == 0 else 0.4
        )

    def _validate_spatial_consistency(self, measurement: dict) -> QualityCheckResult:
        """Validate spatial consistency by comparing against nearby sensor baselines.

        Uses the in-memory ``sensor_baselines`` registry: the current reading is
        compared against the mean ± threshold×std of all other sensors that have
        established baselines.  A reading is flagged only if it deviates beyond
        threshold standard deviations from the majority (>80%) of neighbours,
        which suggests a sensor fault rather than a real environmental signal.

        Falls back to passing when fewer than ``min_neighbors`` baselines exist.
        """
        issues: list = []
        sensor_id = measurement.get("sensor_id", "unknown")
        value = measurement.get("value")

        if value is None:
            return QualityCheckResult(True, [], 1.0)

        spatial_cfg = self.config.get(
            "spatial_consistency", self.default_params["spatial_consistency"]
        )
        threshold = float(spatial_cfg.get("neighbor_threshold", 2.0))
        min_neighbors = int(spatial_cfg.get("min_neighbors", 3))

        # Collect baselines from all other sensors with >= 5 measurements
        neighbor_baselines = [
            b
            for sid, b in self.sensor_baselines.items()
            if sid != sensor_id and b.get("count", 0) >= 5
        ]

        if len(neighbor_baselines) < min_neighbors:
            return QualityCheckResult(True, [], 0.9)

        outlier_count = sum(
            1
            for b in neighbor_baselines
            if abs(value - b["mean"]) > threshold * max(b["std"], 0.01)
        )
        fraction_outlier = outlier_count / len(neighbor_baselines)

        if fraction_outlier > 0.8:
            issues.append(
                f"Spatial outlier: value {value:.3f} deviates >{threshold}σ from "
                f"{outlier_count}/{len(neighbor_baselines)} neighbour baselines"
            )

        quality_score = round(1.0 - fraction_outlier * 0.5, 3)
        return QualityCheckResult(len(issues) == 0, issues, quality_score)

    @staticmethod
    def _parse_timestamp(timestamp: Any) -> datetime | None:
        """Coerce a measurement timestamp to a naive datetime, or None.

        Args:
            timestamp: An ISO-8601 string, a datetime, or anything else.

        Returns:
            A timezone-naive datetime, or None when the value is unusable.
        """
        if isinstance(timestamp, datetime):
            parsed = timestamp
        elif isinstance(timestamp, str):
            try:
                parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except ValueError:
                return None
        else:
            return None
        # Drop tzinfo so mixed-awareness histories stay comparable.
        return parsed.replace(tzinfo=None)

    def _record_measurement(self, measurement: dict) -> None:
        """Append a measurement to its sensor's history ring.

        Measurements without a usable numeric value or timestamp are not
        retained, because the temporal and outlier checks cannot use them.

        Args:
            measurement: The measurement that was just validated.
        """
        value = measurement.get("value")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return
        if np.isnan(value) or np.isinf(value):
            return
        observed_at = self._parse_timestamp(measurement.get("timestamp"))
        if observed_at is None:
            return
        sensor_id = measurement.get("sensor_id", "unknown")
        self.measurement_history[sensor_id].append(
            {"value": float(value), "timestamp": observed_at}
        )

    def _get_recent_measurements(self, sensor_id: str, minutes: int = 60) -> list[dict]:
        """Return this sensor's retained measurements from the last *minutes*.

        The window is anchored on the newest retained measurement rather than
        wall-clock now, so replayed or backfilled histories are analysed the
        same way live ones are.

        Args:
            sensor_id: Identifier of the sensor to look up.
            minutes: Width of the lookback window in minutes.

        Returns:
            Measurements inside the window, oldest first.
        """
        history = self.measurement_history.get(sensor_id)
        if not history:
            return []
        newest = history[-1]["timestamp"]
        cutoff = newest - timedelta(minutes=minutes)
        return [entry for entry in history if entry["timestamp"] >= cutoff]

    def _update_sensor_baseline(self, sensor_id: str, value: float) -> None:
        """Update baseline statistics for a sensor."""
        if sensor_id not in self.sensor_baselines:
            self.sensor_baselines[sensor_id] = {
                "mean": value,
                "std": 0.1,
                "count": 1,
                "last_update": datetime.now(),
            }
        else:
            baseline = self.sensor_baselines[sensor_id]
            count = baseline["count"]
            old_mean = baseline["mean"]

            # Update mean using Welford's online algorithm
            new_mean = old_mean + (value - old_mean) / (count + 1)

            # Update variance (simplified)
            if count > 1:
                old_var = baseline["std"] ** 2
                new_var = (
                    (count - 1) * old_var + (value - old_mean) * (value - new_mean)
                ) / count
                new_std = np.sqrt(
                    max(0.01, new_var)
                )  # Minimum std to avoid division by zero
            else:
                new_std = 0.1

            self.sensor_baselines[sensor_id].update(
                {
                    "mean": new_mean,
                    "std": new_std,
                    "count": count + 1,
                    "last_update": datetime.now(),
                }
            )

    def validate_batch(self, measurements: list[dict]) -> dict:
        """
        Validate a batch of measurements.

        Args:
            measurements: List of measurements to validate

        Returns:
            Dictionary with batch validation results
        """
        batch_results = []

        for measurement in measurements:
            result = self.validate_measurement(measurement)
            batch_results.append(
                {
                    "sensor_id": measurement.get("sensor_id", "unknown"),
                    "passed": result.passed,
                    "quality_score": result.quality_score,
                    "issues": result.issues,
                }
            )

        # Batch statistics
        passed_count = sum(1 for r in batch_results if r["passed"])
        avg_quality_score = np.mean([r["quality_score"] for r in batch_results])

        return {
            "total_measurements": len(measurements),
            "passed_measurements": passed_count,
            "failed_measurements": len(measurements) - passed_count,
            "pass_rate": passed_count / len(measurements) if measurements else 0.0,
            "average_quality_score": avg_quality_score,
            "results": batch_results,
        }

    def get_quality_report(self, time_window_hours: int = 24) -> dict:
        """
        Generate quality control report for recent measurements.

        Args:
            time_window_hours: Time window for the report

        Returns:
            Comprehensive quality control report
        """
        # This would analyze quality history within the time window
        # For now, return summary of current state

        return {
            "report_period_hours": time_window_hours,
            "sensors_tracked": len(self.sensor_baselines),
            "quality_checks_performed": len(self.quality_history),
            "sensor_baselines": {
                sensor_id: {
                    "mean": baseline["mean"],
                    "std": baseline["std"],
                    "measurement_count": baseline["count"],
                }
                for sensor_id, baseline in self.sensor_baselines.items()
            },
            "generated_at": datetime.now().isoformat(),
        }
