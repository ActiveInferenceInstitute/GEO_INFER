"""
Event detection for GEO-INFER-TIME.

This module provides event detection capabilities for identifying
anomalies, changepoints, and significant events in time series data.
"""

import logging
import math
from typing import Any
import numpy as np

from ..models.timeseries import TimeSeries

logger = logging.getLogger(__name__)


class EventDetector:
    """
    Event detector for time series data.

    Provides methods for detecting anomalies, changepoints, and
    significant events in temporal data.
    """

    def __init__(
        self,
        threshold_multiplier: float = 3.0,
        window_size: int = 10,
    ) -> None:
        """
        Initialize the event detector.

        Args:
            threshold_multiplier: Multiplier for standard deviation threshold
            window_size: Window size for moving statistics
        """
        if isinstance(threshold_multiplier, bool) or not isinstance(
            threshold_multiplier, (int, float)
        ):
            raise TypeError("threshold_multiplier must be a number")
        if not math.isfinite(threshold_multiplier) or threshold_multiplier < 0:
            raise ValueError("threshold_multiplier must be finite and non-negative")
        if isinstance(window_size, bool) or not isinstance(window_size, int):
            raise TypeError("window_size must be an integer")
        if window_size <= 0:
            raise ValueError("window_size must be greater than zero")

        self.threshold_multiplier = float(threshold_multiplier)
        self.window_size = window_size

    def detect_anomalies(
        self, timeseries: TimeSeries, method: str = "z_score"
    ) -> dict[str, Any]:
        """
        Detect anomalies in time series.

        Args:
            timeseries: TimeSeries object
            method: Detection method ('z_score', 'iqr', 'isolation_forest')

        Returns:
            Dictionary with detected anomalies
        """
        supported_methods = {"z_score", "iqr", "isolation_forest"}
        if method not in supported_methods:
            raise ValueError(f"Unknown anomaly detection method: {method}")

        data = timeseries.to_dataframe()
        if data.empty or data.shape[1] == 0:
            return {"method": method, "anomalies": [], "count": 0}
        values = data.iloc[:, 0].dropna().values
        timestamps = data.index[~data.iloc[:, 0].isna()]

        if len(values) == 0:
            return {"method": method, "anomalies": [], "count": 0}

        anomalies = []

        if method == "z_score":
            mean = np.mean(values)
            std = np.std(values)

            for val, ts in zip(values, timestamps):
                z_score = abs(val - mean) / std if std > 0 else 0
                if z_score > self.threshold_multiplier:
                    anomalies.append(
                        {
                            "timestamp": ts.isoformat(),
                            "value": float(val),
                            "z_score": float(z_score),
                            "type": "outlier",
                        }
                    )

        elif method == "iqr":
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            lower_bound = q1 - self.threshold_multiplier * iqr
            upper_bound = q3 + self.threshold_multiplier * iqr

            for val, ts in zip(values, timestamps):
                if val < lower_bound or val > upper_bound:
                    anomalies.append(
                        {
                            "timestamp": ts.isoformat(),
                            "value": float(val),
                            "type": "outlier",
                        }
                    )

        elif method == "isolation_forest":
            from sklearn.ensemble import IsolationForest

            clf = IsolationForest(
                contamination=0.05,
                random_state=42,
            )
            predictions = clf.fit_predict(values.reshape(-1, 1))
            scores = clf.decision_function(values.reshape(-1, 1))

            for i, (pred, val, score) in enumerate(zip(predictions, values, scores)):
                if pred == -1:
                    anomalies.append(
                        {
                            "timestamp": timestamps[i].isoformat(),
                            "value": float(val),
                            "score": float(score),
                            "type": "outlier",
                        }
                    )

        return {
            "method": method,
            "anomalies": anomalies,
            "count": len(anomalies),
        }

    def detect_changepoints(
        self, timeseries: TimeSeries, sensitivity: float = 0.5
    ) -> dict[str, Any]:
        """
        Detect changepoints in time series.

        Args:
            timeseries: TimeSeries object
            sensitivity: Non-negative multiplier applied to local variability.

        Returns:
            Dictionary with detected changepoints
        """
        if isinstance(sensitivity, bool) or not isinstance(sensitivity, (int, float)):
            raise TypeError("sensitivity must be a number")
        if not math.isfinite(sensitivity) or sensitivity < 0:
            raise ValueError("sensitivity must be finite and non-negative")

        data = timeseries.to_dataframe()
        if data.empty or data.shape[1] == 0:
            return {"changepoints": [], "count": 0}
        values = data.iloc[:, 0].dropna().values
        timestamps = data.index[~data.iloc[:, 0].isna()]

        changepoints = []

        # Simple changepoint detection using moving window statistics
        for i in range(self.window_size, len(values) - self.window_size):
            window_before = values[i - self.window_size : i]
            window_after = values[i : i + self.window_size]

            mean_before = np.mean(window_before)
            mean_after = np.mean(window_after)
            std_before = np.std(window_before)
            std_after = np.std(window_after)

            # Detect significant change in mean
            mean_change = abs(mean_after - mean_before)
            threshold = sensitivity * (std_before + std_after) / 2

            if mean_change > threshold:
                changepoints.append(
                    {
                        "timestamp": timestamps[i].isoformat(),
                        "index": i,
                        "mean_change": float(mean_change),
                        "mean_before": float(mean_before),
                        "mean_after": float(mean_after),
                    }
                )

        return {
            "changepoints": changepoints,
            "count": len(changepoints),
        }
