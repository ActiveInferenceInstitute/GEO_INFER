"""
Module bridge: integrates GEO-INFER-DATA and GEO-INFER-TIME into PLACE workflows.

Provides PlaceDataManager (data acquisition + quality) and PlaceTemporalAnalyzer
(time-series trend detection on tide, seismic, and environmental data).

Both classes use graceful degradation - if the upstream module is not installed
the bridge still works with reduced functionality and logs a warning.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conditional imports with graceful fallback
# ---------------------------------------------------------------------------

_HAS_DATA = False
_HAS_TIME = False

try:
    from geo_infer_data import (
        DataQualityManager,
        DataQualityReport as DataQualityReport,
        MultiSourceDataIngestion,
    )

    _HAS_DATA = True
except ImportError:
    logger.info(
        "geo_infer_data not available; PlaceDataManager will use built-in methods"
    )

try:
    from geo_infer_time import (
        EventDetector,
        ForecastingEngine,
        TemporalAnalyzer,
        TimeSeries,
    )

    _HAS_TIME = True
except ImportError:
    logger.info(
        "geo_infer_time not available; PlaceTemporalAnalyzer will use built-in methods"
    )


# ---------------------------------------------------------------------------
# PlaceDataManager - wraps GEO-INFER-DATA for PLACE workflows
# ---------------------------------------------------------------------------


class PlaceDataManager:
    """Unified data acquisition and quality management for PLACE analyses.

    When ``geo_infer_data`` is installed this class delegates to its
    ``MultiSourceDataIngestion`` and ``DataQualityManager``.  When it is not
    installed the class provides lightweight equivalents so that PLACE can
    still run standalone.

    Example::

        mgr = PlaceDataManager()
        report = mgr.validate_dataset(my_geojson)
        mgr.log_provenance("calfire_perimeters", metadata)
    """

    def __init__(self, config_path: str | None = None) -> None:
        self._provenance: list[dict[str, Any]] = []
        self._ingestion = None
        self._quality = None

        if _HAS_DATA:
            try:
                self._ingestion = MultiSourceDataIngestion(data_sources=["local"])
                self._quality = DataQualityManager()
                logger.info("PlaceDataManager initialised with GEO-INFER-DATA backend")
            except Exception as exc:
                logger.warning("GEO-INFER-DATA init failed, using built-in: %s", exc)

    @property
    def has_data_module(self) -> bool:
        return _HAS_DATA and self._quality is not None

    # -- Quality validation -------------------------------------------------

    def validate_dataset(self, data: Any, name: str = "unnamed") -> dict[str, Any]:
        """Validate a dataset and return a quality report.

        Args:
            data: Raw dataset (dict, GeoJSON, DataFrame, etc.).
            name: Human-readable dataset name for reporting.

        Returns:
            Dict with keys ``valid``, ``completeness``, ``issues``.
        """
        if self._quality is not None:
            try:
                validator: Any = getattr(self._quality, "validator", self._quality)
                report = validator.validate_data(data)
                return {
                    "valid": report.is_valid if hasattr(report, "is_valid") else True,
                    "completeness": getattr(report, "completeness", 1.0),
                    "issues": getattr(report, "issues", []),
                    "backend": "geo_infer_data",
                }
            except Exception as exc:
                logger.warning("DATA quality check failed for %s: %s", name, exc)

        # Built-in lightweight validation
        issues = []
        completeness = 1.0

        if isinstance(data, dict):
            if data.get("type") == "FeatureCollection":
                features = data.get("features", [])
                if not features:
                    issues.append("FeatureCollection has no features")
                    completeness = 0.0
                else:
                    # Check for missing geometries
                    missing_geom = sum(1 for f in features if not f.get("geometry"))
                    if missing_geom:
                        issues.append(
                            f"{missing_geom}/{len(features)} features lack geometry"
                        )
                        completeness = 1.0 - missing_geom / len(features)
            elif "success" in data and not data["success"]:
                issues.append(
                    f"Data source returned error: {data.get('error', 'unknown')}"
                )
                completeness = 0.0

        return {
            "valid": len(issues) == 0,
            "completeness": completeness,
            "issues": issues,
            "backend": "built_in",
        }

    # -- Provenance tracking ------------------------------------------------

    def log_provenance(
        self,
        source_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record data provenance for audit trails.

        Args:
            source_name: Identifier of the data source (e.g. "noaa_tides").
            metadata: Arbitrary metadata dict.
        """
        entry = {
            "source": source_name,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        self._provenance.append(entry)
        logger.debug("Provenance logged for %s", source_name)

    def get_provenance(self) -> list[dict[str, Any]]:
        """Return the full provenance log."""
        return list(self._provenance)


# ---------------------------------------------------------------------------
# PlaceTemporalAnalyzer - wraps GEO-INFER-TIME for PLACE workflows
# ---------------------------------------------------------------------------


class PlaceTemporalAnalyzer:
    """Time-series analysis for PLACE environmental data.

    When ``geo_infer_time`` is installed this class uses its
    ``TemporalAnalyzer``, ``EventDetector``, and ``ForecastingEngine``.
    Otherwise it provides basic numpy-based trend detection and anomaly
    flagging.

    Typical use cases:
    - Detecting tide level anomalies (storm surge, tsunami precursors)
    - Identifying fire weather trend escalation
    - Tracking seismic rate changes over time
    - Forest health NDVI trend analysis
    """

    def __init__(self) -> None:
        self._analyzer = None
        self._detector = None
        self._forecaster = None

        if _HAS_TIME:
            try:
                self._analyzer = TemporalAnalyzer()
                self._detector = EventDetector()
                self._forecaster = ForecastingEngine()
                logger.info(
                    "PlaceTemporalAnalyzer initialised with GEO-INFER-TIME backend"
                )
            except Exception as exc:
                logger.warning("GEO-INFER-TIME init failed, using built-in: %s", exc)

    @property
    def has_time_module(self) -> bool:
        return _HAS_TIME and self._analyzer is not None

    # -- Trend analysis -----------------------------------------------------

    def detect_trend(
        self,
        values: list[float],
        timestamps: list[str] | None = None,
        label: str = "series",
    ) -> dict[str, Any]:
        """Detect linear trend in a time series.

        Args:
            values: Ordered numeric values.
            timestamps: Optional ISO-format timestamp strings.
            label: Series label for reporting.

        Returns:
            Dict with ``slope``, ``direction``, ``r_squared``, ``significant``.
        """
        import numpy as np

        if self._analyzer is not None:
            try:
                import pandas as pd

                time_series_cls = globals().get("TimeSeries")
                if time_series_cls is not None:
                    ts = time_series_cls(data=pd.Series(values))
                    res = self._analyzer.detect_trend(ts)
                else:
                    # TIME payload class unavailable despite a patched flag;
                    # pass raw values so the backend can still be exercised.
                    res = self._analyzer.detect_trend(values)
                if isinstance(res, dict):
                    slope_val = res.get("trend_strength", 0.0)
                    if res.get("trend_direction") == "decreasing":
                        slope_val = -slope_val
                    merged = {
                        "slope": slope_val,
                        "direction": res.get("trend_direction", "stable"),
                        **res,
                        "backend": "geo_infer_time",
                    }
                    # Report fit quality honestly: prefer a real r_squared
                    # field from the backend, otherwise fall back to the
                    # reported trend_strength. Never fabricate a perfect fit.
                    r_sq = float(
                        merged.get("r_squared", merged.get("trend_strength", 0.0))
                    )
                    merged["r_squared"] = r_sq
                    merged["significant"] = r_sq > 0.5
                    return merged
            except Exception as exc:
                logger.debug("TIME trend detection fallback for %s: %s", label, exc)

        # Built-in numpy linear regression
        arr = np.array(values, dtype=float)
        n = len(arr)
        if n < 3:
            return {
                "slope": 0.0,
                "direction": "insufficient_data",
                "r_squared": 0.0,
                "significant": False,
            }

        x = np.arange(n, dtype=float)
        slope, intercept = np.polyfit(x, arr, 1)
        predicted = slope * x + intercept
        ss_res = np.sum((arr - predicted) ** 2)
        ss_tot = np.sum((arr - np.mean(arr)) ** 2)
        r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        direction = (
            "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
        )
        significant = abs(r_sq) > 0.5 and n >= 5

        return {
            "slope": round(float(slope), 6),
            "direction": direction,
            "r_squared": round(float(r_sq), 4),
            "significant": significant,
            "n_observations": n,
            "backend": "built_in",
        }

    # -- Anomaly detection --------------------------------------------------

    def detect_anomalies(
        self,
        values: list[float],
        sigma_threshold: float = 2.0,
        label: str = "series",
    ) -> dict[str, Any]:
        """Detect anomalous values in a time series.

        Args:
            values: Ordered numeric values.
            sigma_threshold: Number of standard deviations for anomaly.
            label: Series label for reporting.

        Returns:
            Dict with ``anomalies`` list, ``mean``, ``std``.
        """
        import numpy as np

        if self._detector is not None:
            try:
                import pandas as pd

                ts = TimeSeries(data=pd.Series(values)) if _HAS_TIME else None
                if ts is not None and hasattr(self._detector, "detect_anomalies"):
                    result = self._detector.detect_anomalies(ts)
                    if isinstance(result, dict):
                        return {**result, "backend": "geo_infer_time"}
            except Exception as exc:
                logger.debug("TIME anomaly detection fallback for %s: %s", label, exc)

        arr = np.array(values, dtype=float)
        if len(arr) < 3:
            return {"anomalies": [], "mean": 0.0, "std": 0.0, "backend": "built_in"}

        mean = float(np.mean(arr))
        std = float(np.std(arr))

        anomalies = []
        for i, v in enumerate(arr):
            if std > 0 and abs(v - mean) > sigma_threshold * std:
                anomalies.append(
                    {
                        "index": i,
                        "value": float(v),
                        "z_score": round(float((v - mean) / std), 3),
                    }
                )

        return {
            "anomalies": anomalies,
            "total_anomalies": len(anomalies),
            "mean": round(mean, 4),
            "std": round(std, 4),
            "threshold_sigma": sigma_threshold,
            "backend": "built_in",
        }

    # -- Forecasting --------------------------------------------------------

    def forecast(
        self,
        values: list[float],
        horizon: int = 12,
        label: str = "series",
    ) -> dict[str, Any]:
        """Produce a simple forecast for a time series.

        Args:
            values: Historical values.
            horizon: Number of steps to forecast.
            label: Series label.

        Returns:
            Dict with ``forecast`` values and ``confidence``.
        """
        import numpy as np

        if self._forecaster is not None:
            try:
                import pandas as pd

                ts = TimeSeries(data=pd.Series(values)) if _HAS_TIME else None
                if ts is not None and hasattr(self._forecaster, "forecast_linear"):
                    result = self._forecaster.forecast_linear(ts, horizon=horizon)
                    if isinstance(result, dict):
                        return {**result, "backend": "geo_infer_time"}
            except Exception as exc:
                logger.debug("TIME forecasting fallback for %s: %s", label, exc)

        # Simple linear extrapolation
        arr = np.array(values, dtype=float)
        n = len(arr)
        if n < 2:
            return {
                "forecast": [float(arr[-1])] * horizon if n else [0.0] * horizon,
                "backend": "built_in",
            }

        x = np.arange(n, dtype=float)
        slope, intercept = np.polyfit(x, arr, 1)
        future_x = np.arange(n, n + horizon, dtype=float)
        forecast_vals = (slope * future_x + intercept).tolist()

        return {
            "forecast": [round(v, 4) for v in forecast_vals],
            "horizon": horizon,
            "method": "linear_extrapolation",
            "backend": "built_in",
        }

    # -- Convenience: analyze tide data -------------------------------------

    def analyze_tide_trends(self, tide_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze tide gauge data for trends, anomalies, and forecasts.

        Args:
            tide_data: Output from ``_NOAAWrapper.get_tide_gauge_data()``.

        Returns:
            Per-station trend analysis results.
        """
        results: dict[str, Any] = {}
        series = tide_data.get("series", {})
        for station_id, station_data in series.items():
            measurements = station_data.get("measurements", [])
            if not measurements:
                measurements = station_data.get("data", {}).get("data", [])

            values = []
            timestamps = []
            for m in measurements:
                wl = m.get("water_level") or m.get("v")
                if wl is not None:
                    try:
                        values.append(float(wl))
                        timestamps.append(m.get("time") or m.get("t", ""))
                    except (ValueError, TypeError):
                        continue

            if not values:
                results[station_id] = {"error": "no_valid_measurements"}
                continue

            results[station_id] = {
                "trend": self.detect_trend(
                    values, timestamps, label=f"tide_{station_id}"
                ),
                "anomalies": self.detect_anomalies(values, label=f"tide_{station_id}"),
                "forecast": self.forecast(
                    values, horizon=24, label=f"tide_{station_id}"
                ),
                "n_measurements": len(values),
            }

        return results

    # -- Convenience: analyze seismic rates ---------------------------------

    def analyze_seismic_rates(self, csz_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze Cascadia seismicity rate changes over time.

        Args:
            csz_data: Output from ``_USGSWrapper.get_cascadia_seismicity()``.

        Returns:
            Rate analysis with daily event counts and trend.
        """
        import numpy as np

        events = csz_data.get("events", [])
        if not events:
            return {"error": "no_events", "trend": None}

        # Bin events by day
        event_times = []
        for e in events:
            t = e.get("time")
            if t is not None:
                try:
                    event_times.append(datetime.utcfromtimestamp(t / 1000))
                except (ValueError, TypeError, OSError):
                    continue

        if not event_times:
            return {"error": "no_valid_timestamps", "trend": None}

        event_times.sort()
        min_day = event_times[0].date()
        max_day = event_times[-1].date()
        n_days = (max_day - min_day).days + 1

        daily_counts = [0] * max(1, n_days)
        for t in event_times:
            idx = (t.date() - min_day).days
            if 0 <= idx < len(daily_counts):
                daily_counts[idx] += 1

        trend = self.detect_trend(
            [float(x) for x in daily_counts], label="csz_daily_rate"
        )
        anomalies = self.detect_anomalies(
            [float(x) for x in daily_counts], label="csz_daily_rate"
        )

        return {
            "daily_counts": daily_counts,
            "total_days": n_days,
            "total_events": len(event_times),
            "mean_daily_rate": round(float(np.mean(daily_counts)), 2),
            "max_daily_count": int(np.max(daily_counts)),
            "trend": trend,
            "anomalies": anomalies,
        }
