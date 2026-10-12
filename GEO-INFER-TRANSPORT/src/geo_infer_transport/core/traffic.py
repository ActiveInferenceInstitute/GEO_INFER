"""
Traffic analysis module.

Provides traffic flow modeling, congestion analysis,
and simulation capabilities.
"""

import logging
import re
from typing import Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class TrafficCondition(Enum):
    """Traffic condition levels."""

    FREE_FLOW = "free_flow"
    LIGHT = "light"
    MODERATE = "moderate"
    HEAVY = "heavy"
    CONGESTED = "congested"
    BLOCKED = "blocked"


@dataclass
class TrafficCount:
    """Traffic count observation."""

    location_id: str
    timestamp: datetime
    count: int
    speed_kmh: float | None = None
    occupancy: float | None = None
    direction: str | None = None


@dataclass
class FlowResult:
    """Traffic flow analysis result."""

    segment_id: str
    volume: int  # vehicles per hour
    density: float  # vehicles per km
    speed: float  # km/h
    level_of_service: str  # A-F


class TrafficAnalyzer:
    """
    Analyze and model traffic flow patterns.

    Supports flow modeling, congestion detection, and
    traffic simulation.
    """

    # Level of Service thresholds (volume/capacity)
    LOS_THRESHOLDS = {"A": 0.35, "B": 0.55, "C": 0.75, "D": 0.87, "E": 0.95, "F": 1.0}

    def __init__(
        self,
        data_sources: list[str] | None = None,
        model_type: str = "bpr",
        time_resolution: str = "15min",
    ):
        """
        Initialize traffic analyzer.

        Args:
            data_sources: Traffic data sources
            model_type: Traffic model type; only ``'bpr'`` (the BPR delay
                function) is implemented and accepted
            time_resolution: Temporal resolution

        Raises:
            ValueError: If ``model_type`` is not ``'bpr'``
        """
        if model_type != "bpr":
            raise ValueError(
                "TrafficAnalyzer: model_type "
                f"{model_type!r} is not supported; only 'bpr' is implemented"
            )
        self.data_sources = data_sources or ["sensor", "probe"]
        self.model_type = model_type
        self.time_resolution = time_resolution
        self._traffic_counts: list[TrafficCount] = []
        logger.info(f"Initialized TrafficAnalyzer with {model_type} model")

    def analyze_flow(
        self,
        segment: dict[str, Any],
        counts: list[dict[str, Any]],
        time_period: str = "peak",
    ) -> FlowResult:
        """
        Analyze traffic flow on a road segment.

        ``hourly_volume`` scales the average per-interval count to a
        vehicles-per-hour rate using the analyzer's configured
        ``time_resolution`` (e.g. ``"15min"`` multiplies by 4,
        ``"5min"`` by 12).

        Args:
            segment: Road segment properties
            counts: Traffic count data
            time_period: Analysis time period

        Returns:
            Flow analysis result
        """
        # Calculate average volume
        total_count = sum(c.get("count", 0) for c in counts)
        avg_volume = total_count / len(counts) if counts else 0

        # Hourly volume: rescale per-interval counts to vehicles per hour
        hourly_volume = int(avg_volume * 3600 / self._seconds_per_interval())
        speeds = [
            c.get("speed_kmh", segment.get("speed_limit", 50))
            for c in counts
            if c.get("speed_kmh")
        ]
        avg_speed = (
            sum(speeds) / len(speeds) if speeds else segment.get("speed_limit", 50)
        )

        # Calculate density (vehicles per km)
        density = hourly_volume / avg_speed if avg_speed > 0 else 0

        # Determine Level of Service
        capacity = segment.get("capacity", 2000)  # Default capacity
        vc_ratio = hourly_volume / capacity if capacity > 0 else 0

        los = "F"
        for level, threshold in self.LOS_THRESHOLDS.items():
            if vc_ratio <= threshold:
                los = level
                break

        result = FlowResult(
            segment_id=segment.get("id", "unknown"),
            volume=hourly_volume,
            density=round(density, 2),
            speed=round(avg_speed, 1),
            level_of_service=los,
        )

        logger.info(f"Flow analysis for {segment.get('id')}: LOS {los}")
        return result

    @staticmethod
    def _parse_duration_seconds(duration: str) -> float:
        """Parse a duration string to seconds.

        Accepted formats combine a number with an s/m/h/d unit, e.g.
        ``"15min"``, ``"90m"``, ``"1h"``, ``"2 days"``.

        Raises:
            ValueError: If the duration cannot be parsed.
        """
        match = re.fullmatch(
            r"(\d+(?:\.\d+)?)\s*(s|sec|secs|seconds?|m|min|mins|minutes?|h|hr|hrs|hours?|d|days?)",
            duration.strip(),
            re.IGNORECASE,
        )
        if not match:
            raise ValueError(f"Unsupported duration: {duration!r}")
        value = float(match.group(1))
        unit = match.group(2).lower()
        if unit.startswith("s"):
            return value
        if unit.startswith("m"):
            return value * 60.0
        if unit.startswith("h"):
            return value * 3600.0
        return value * 86400.0

    def _seconds_per_interval(self) -> float:
        """Convert ``self.time_resolution`` to seconds per count interval.

        Raises:
            ValueError: If ``time_resolution`` cannot be parsed.
        """
        try:
            return self._parse_duration_seconds(self.time_resolution)
        except ValueError as exc:
            raise ValueError(
                f"Unsupported time_resolution: {self.time_resolution!r}"
            ) from exc

    def model_congestion(
        self,
        network_flows: dict[str, float],
        capacity_data: dict[str, float],
        algorithm: str = "bpr",
    ) -> dict[str, Any]:
        """
        Model congestion across the network.

        Args:
            network_flows: Volume on each segment
            capacity_data: Capacity of each segment
            algorithm: Congestion function

        Returns:
            Congestion analysis results
        """
        segments_out: list[dict[str, Any]] = []
        total_segments = len(network_flows)
        summary_out: dict[str, Any] = {
            "total_segments": total_segments,
            "congested_segments": 0,
            "average_delay_factor": 0,
        }
        congestion_results: dict[str, Any] = {
            "algorithm": algorithm,
            "timestamp": datetime.now().isoformat(),
            "segments": segments_out,
            "summary": summary_out,
        }

        total_delay = 0.0

        for segment_id, volume in network_flows.items():
            capacity = capacity_data.get(segment_id, 2000)

            # Calculate delay using BPR function: t = t0 * (1 + α(v/c)^β)
            # α = 0.15, β = 4 are standard BPR parameters
            if algorithm == "bpr":
                vc_ratio = volume / capacity if capacity > 0 else 0
                delay_factor = 1 + 0.15 * (vc_ratio**4)
            else:
                delay_factor = 1 + (volume / capacity) if capacity > 0 else 1

            # Determine condition
            if delay_factor < 1.05:
                condition = TrafficCondition.FREE_FLOW
            elif delay_factor < 1.2:
                condition = TrafficCondition.LIGHT
            elif delay_factor < 1.5:
                condition = TrafficCondition.MODERATE
            elif delay_factor < 2.0:
                condition = TrafficCondition.HEAVY
            else:
                condition = TrafficCondition.CONGESTED
                summary_out["congested_segments"] += 1

            total_delay += float(delay_factor)

            segments_out.append(
                {
                    "segment_id": segment_id,
                    "volume": volume,
                    "capacity": capacity,
                    "vc_ratio": round(volume / capacity, 3) if capacity > 0 else 0,
                    "delay_factor": round(delay_factor, 3),
                    "condition": condition.value,
                }
            )

        if network_flows:
            summary_out["average_delay_factor"] = round(
                total_delay / len(network_flows), 3
            )

        logger.info(f"Congestion modeled for {len(network_flows)} segments")
        return congestion_results

    def simulate_traffic(
        self,
        network: Any,
        demand_matrix: dict[str, Any],
        simulation_hours: int = 1,
        time_step_seconds: int = 60,
    ) -> dict[str, Any]:
        """
        Simulate traffic flow over time.

        Args:
            network: Transport network
            demand_matrix: OD demand matrix
            simulation_hours: Hours to simulate
            time_step_seconds: Simulation time step

        Returns:
            Simulation results
        """
        if time_step_seconds <= 0:
            raise ValueError(
                f"simulate_traffic: time_step_seconds must be > 0, got {time_step_seconds}"
            )
        num_steps = (simulation_hours * 3600) // time_step_seconds
        results_out: list[dict[str, Any]] = []
        statistics_out: dict[str, Any] = {
            "total_trips": 0,
            "completed_trips": 0,
            "average_travel_time": 0,
        }
        simulation: dict[str, Any] = {
            "simulation_id": f"sim_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "duration_hours": simulation_hours,
            "time_step_seconds": time_step_seconds,
            "total_steps": num_steps,
            "results": results_out,
            "statistics": statistics_out,
        }

        # Extract demand from OD matrix
        matrix = demand_matrix.get("matrix", [[]])
        total_demand = sum(sum(row) for row in matrix)
        statistics_out["total_trips"] = int(total_demand)

        # Trip release rate per step (uniform release across simulation)
        trips_per_step = total_demand / max(num_steps, 1)

        # Network capacity: extract from network or use defaults
        default_capacity = 2000  # vehicles/hour per link
        free_flow_speed = 60.0  # km/h

        # Track vehicles in the network and cumulative travel time
        vehicles_in_network = 0.0
        total_travel_time = 0.0
        completed_trips = 0

        # Average trip length estimate (steps to traverse a route)
        mean_trip_duration_steps = max(3, num_steps // 10)  # ~10% of sim duration

        for step in range(min(num_steps, 60)):  # Cap output detail
            step_time = step * time_step_seconds

            # Release new trips
            vehicles_in_network += trips_per_step

            # BPR delay: effective speed = free_flow / (1 + 0.15*(V/C)^4)
            vc_ratio = vehicles_in_network / max(default_capacity, 1)
            bpr_factor = 1 + 0.15 * (vc_ratio**4)
            effective_speed = free_flow_speed / bpr_factor

            # Complete trips that have been in the network long enough
            if step >= mean_trip_duration_steps:
                departing = trips_per_step  # Roughly FIFO
                vehicles_in_network = max(0, vehicles_in_network - departing)
                completed_trips += int(departing)
                total_travel_time += (
                    departing * mean_trip_duration_steps * time_step_seconds
                )

            # Classify congestion from V/C ratio
            if vc_ratio < 0.35:
                congestion = "free_flow"
            elif vc_ratio < 0.55:
                congestion = "light"
            elif vc_ratio < 0.75:
                congestion = "moderate"
            elif vc_ratio < 0.95:
                congestion = "heavy"
            else:
                congestion = "congested"

            results_out.append(
                {
                    "step": step,
                    "time_seconds": step_time,
                    "vehicles_in_network": int(vehicles_in_network),
                    "average_speed_kmh": round(effective_speed, 1),
                    "congestion_level": congestion,
                    "vc_ratio": round(vc_ratio, 3),
                    "bpr_delay_factor": round(bpr_factor, 3),
                }
            )

        statistics_out["completed_trips"] = completed_trips
        statistics_out["average_travel_time"] = round(
            total_travel_time / max(completed_trips, 1), 1
        )

        logger.info(f"Traffic simulation completed: {num_steps} steps")
        return simulation

    def detect_incidents(
        self,
        current_data: dict[str, Any],
        historical_baseline: dict[str, Any],
        threshold: float = 0.3,
    ) -> list[dict[str, Any]]:
        """
        Detect traffic incidents from anomalies.

        Args:
            current_data: Current traffic conditions
            historical_baseline: Historical patterns
            threshold: Anomaly detection threshold

        Returns:
            List of detected incidents
        """
        incidents: list[dict[str, Any]] = []

        for segment_id, current in current_data.items():
            baseline = historical_baseline.get(segment_id, {})

            current_speed = current.get("speed", 50)
            baseline_speed = baseline.get("speed", 50)

            # Detect significant speed drop
            if baseline_speed > 0:
                speed_deviation = (baseline_speed - current_speed) / baseline_speed

                if speed_deviation > threshold:
                    severity = "high" if speed_deviation > 0.5 else "moderate"

                    incidents.append(
                        {
                            "incident_id": f"inc_{segment_id}_{datetime.now().strftime('%H%M%S')}",
                            "segment_id": segment_id,
                            "type": "congestion_anomaly",
                            "severity": severity,
                            "current_speed": current_speed,
                            "expected_speed": baseline_speed,
                            "deviation": round(speed_deviation, 3),
                            "detected_at": datetime.now().isoformat(),
                        }
                    )

        logger.info(f"Detected {len(incidents)} potential incidents")
        return incidents

    def forecast_traffic(
        self,
        historical_data: list[dict[str, Any]],
        forecast_horizon: str = "1h",
        model: str = "arima",
    ) -> dict[str, Any]:
        """
        Forecast future traffic conditions.

        Args:
            historical_data: Historical traffic data
            forecast_horizon: Forecast duration string (number + s/m/h/d unit,
                e.g. ``"30m"``, ``"90m"``, ``"1h"``, ``"1d"``). Converted to a
                number of ``self.time_resolution``-sized forecast steps.
                Must span at least one full interval.
            model: Forecasting model

        Returns:
            Traffic forecast
        """
        # Parse forecast horizon into time_resolution-sized steps
        horizon_seconds = self._parse_duration_seconds(forecast_horizon)
        interval_seconds = self._seconds_per_interval()
        steps = int(horizon_seconds // interval_seconds)
        if steps <= 0:
            raise ValueError(
                f"forecast_horizon {forecast_horizon!r} is shorter than one "
                f"interval ({self.time_resolution!r})"
            )

        # Exponentially Weighted Moving Average (EWMA) forecast
        alpha = 0.3  # Smoothing factor
        window = min(12, len(historical_data)) if historical_data else 0

        if historical_data and window > 0:
            values = [d.get("volume", 0) for d in historical_data[-window:]]

            # Compute EWMA of historical values
            ewma = values[0]
            for v in values[1:]:
                ewma = alpha * v + (1 - alpha) * ewma

            # Estimate trend from linear regression on window
            if len(values) >= 3:
                n = len(values)
                x_mean = (n - 1) / 2.0
                y_mean = sum(values) / n
                numerator = sum(
                    (i - x_mean) * (v - y_mean) for i, v in enumerate(values)
                )
                denominator = sum((i - x_mean) ** 2 for i in range(n))
                trend = numerator / denominator if denominator > 0 else 0
            else:
                trend = 0

            # Residual std for prediction intervals
            residuals = [v - (ewma + trend * i) for i, v in enumerate(values)]
            residual_std = (
                sum(r**2 for r in residuals) / max(len(residuals), 1)
            ) ** 0.5
        else:
            ewma = 1000
            trend = 0
            residual_std = 200  # default uncertainty

        forecasts_out: list[dict[str, Any]] = []
        forecast: dict[str, Any] = {
            "model": model,
            "forecast_horizon": forecast_horizon,
            "generated_at": datetime.now().isoformat(),
            "parameters": {"alpha": alpha, "window": window, "trend": round(trend, 3)},
            "forecasts": forecasts_out,
        }

        # Generate forecast points with widening confidence intervals
        for i in range(steps):
            predicted = ewma + trend * (i + 1)
            # Prediction interval widens with horizon
            interval_width = residual_std * (1 + 0.1 * i)

            forecasts_out.append(
                {
                    "time_offset_minutes": (i + 1) * 15,
                    "predicted_volume": max(0, int(round(predicted))),
                    "confidence_lower": max(
                        0, int(round(predicted - 1.96 * interval_width))
                    ),
                    "confidence_upper": int(round(predicted + 1.96 * interval_width)),
                }
            )

        logger.info(f"Generated {steps}-step traffic forecast (EWMA α={alpha})")
        return forecast
