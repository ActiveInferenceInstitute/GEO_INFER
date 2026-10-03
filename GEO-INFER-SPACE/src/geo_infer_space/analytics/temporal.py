"""
Temporal Analytics Module for GEO-INFER-SPACE.

This module provides methods for analyzing temporal patterns, trends,
and time-series data associated with spatial cells. It is backend-agnostic
and operates on standard data structures.
"""

import logging
import math
from typing import Any
from datetime import datetime

import numpy as np
from geo_infer_time import normalize_datetime_index

logger = logging.getLogger(__name__)


class TemporalAnalyzer:
    """
    Analyzer for temporal patterns in spatial data.

    Provides methods for analyzing temporal patterns, trends,
    and time-series data.
    """

    def __init__(self) -> None:
        """Initialize the TemporalAnalyzer."""
        self.analysis_history: list[dict[str, Any]] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def analyze_temporal_patterns(
        self,
        data: list[dict[str, Any]],
        timestamp_column: str,
        value_column: str,
        temporal_resolution: str = "hour",
    ) -> dict[str, Any]:
        """
        Analyze temporal patterns in data.

        Args:
            data: List of dictionaries containing data records
            timestamp_column: Key containing timestamps
            value_column: Key containing values to analyze
            temporal_resolution: Resolution ('hour', 'day', 'week', 'month')

        Returns:
            Dictionary containing temporal pattern analysis
        """
        if temporal_resolution not in {"hour", "day", "week", "month"}:
            raise ValueError("temporal resolution must be hour, day, week or month")
        if not data:
            return {"error": "No data provided"}

        # Extract and parse temporal data
        temporal_data = []
        for record in data:
            if timestamp_column in record and value_column in record:
                ts_val = record[timestamp_column]
                val = record[value_column]

                if ts_val is not None and val is not None:
                    if (
                        isinstance(val, (bool, str, bytes))
                        or not isinstance(val, (int, float))
                        or not math.isfinite(val)
                    ):
                        raise ValueError("Observed values must be finite numbers")
                    timestamp = self._parse_timestamp(ts_val)
                    if timestamp:
                        temporal_data.append(
                            {
                                "timestamp": timestamp,
                                "value": float(val),
                                "original_record": record,
                            }
                        )

        if not temporal_data:
            return {"error": "No valid temporal data found"}

        # Aggregate by temporal resolution
        aggregated_data = self._aggregate_by_temporal_resolution(
            temporal_data, temporal_resolution
        )

        # Analyze patterns
        patterns = self._analyze_patterns(aggregated_data, temporal_resolution)

        # Calculate statistics
        stats = self._calculate_temporal_stats(aggregated_data)

        return {
            "temporal_patterns": patterns,
            "aggregated_data": aggregated_data,
            "statistics": stats,
            "temporal_resolution": temporal_resolution,
            "data_points": len(temporal_data),
            "method": "Temporal Pattern Analysis",
        }

    def _parse_timestamp(self, ts: Any) -> datetime:
        """Normalize an explicit aware instant to UTC."""
        return normalize_datetime_index([ts])[0]

    def _aggregate_by_temporal_resolution(
        self, temporal_data: list[dict], resolution: str
    ) -> dict[int, dict[str, float]]:
        """Aggregate data by temporal resolution."""
        aggregated: dict[int, list[float]] = {}

        for item in temporal_data:
            ts = item["timestamp"]
            val = item["value"]

            if resolution == "hour":
                key = ts.hour
            elif resolution == "day":
                key = ts.weekday()
            elif resolution == "week":
                key = ts.isocalendar()[1]
            elif resolution == "month":
                key = ts.month
            else:
                key = ts.hour

            if key not in aggregated:
                aggregated[key] = []
            aggregated[key].append(val)

        # Calculate stats for each bucket
        result: dict[int, dict[str, float]] = {}
        for key, values in aggregated.items():
            stats = {
                "mean": float(np.mean(values)),
                "sum": float(np.sum(values)),
                "count": len(values),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
            result[key] = stats

        return result

    def _analyze_patterns(
        self, aggregated_data: dict, resolution: str
    ) -> dict[str, Any]:
        """Analyze temporal patterns in aggregated data."""
        if not aggregated_data:
            return {}

        # Find peak periods
        sorted_periods = sorted(
            aggregated_data.items(), key=lambda x: x[1]["mean"], reverse=True
        )

        peak_periods = []
        for period, stats in sorted_periods[:5]:
            peak_periods.append(
                {"period": period, "mean_value": stats["mean"], "count": stats["count"]}
            )

        return {"peak_periods": peak_periods, "total_periods": len(aggregated_data)}

    def _calculate_temporal_stats(self, aggregated_data: dict) -> dict[str, Any]:
        """Calculate overall statistics."""
        if not aggregated_data:
            return {}

        buckets = list(aggregated_data.values())
        count = sum(bucket["count"] for bucket in buckets)
        total = math.fsum(bucket["sum"] for bucket in buckets)
        mean = total / count
        variance = (
            math.fsum(
                bucket["count"] * (bucket["std"] ** 2 + (bucket["mean"] - mean) ** 2)
                for bucket in buckets
            )
            / count
        )
        return {
            "overall_mean": mean,
            "overall_std": math.sqrt(variance),
            "min_mean": min(bucket["mean"] for bucket in buckets),
            "max_mean": max(bucket["mean"] for bucket in buckets),
            "overall_min": min(bucket["min"] for bucket in buckets),
            "overall_max": max(bucket["max"] for bucket in buckets),
            "total_observations": count,
            "total_value": total,
        }
