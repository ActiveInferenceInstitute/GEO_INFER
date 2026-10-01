# Temporal Analysis Guide

This guide covers temporal data handling and analysis within the GEO-INFER framework. It focuses on the `geo_infer_time` module's capabilities for time series decomposition, changepoint detection, stream processing, and integration with spatial analysis and Active Inference.

## Temporal Data Types in GEO-INFER

GEO-INFER handles three categories of temporal data:

| Data Type | Description | Example | Module |
|-----------|-------------|---------|--------|
| **Time series** | Regular or irregular measurements over time | Daily temperature at a weather station | `geo_infer_time.core.analysis` |
| **Event sequences** | Timestamped discrete events | Earthquake occurrences, fire ignitions | `geo_infer_time.core.event_detection` |
| **Trajectories** | Sequences of (time, location) pairs | Vehicle GPS tracks, animal movement | `geo_infer_time` + `geo_infer_space` |

## GEO-INFER-TIME Module Capabilities

The module is organized into core components:

| Component | File | Purpose |
|-----------|------|---------|
| `TemporalAnalyzer` | `core/analysis.py` | Trend detection, seasonality, decomposition |
| `StreamProcessor` | `core/stream_processing.py` | Real-time windowed processing |
| `EventDetector` | `core/event_detection.py` | Anomaly and event detection |
| `Forecaster` | `core/forecasting.py` | Time series forecasting |
| `AdvancedForecaster` | `core/advanced_forecasting.py` | Ensemble and multi-model forecasting |
| `TemporalInterpolator` | `core/interpolation.py` | Gap filling in time series |
| `TemporalStatistics` | `core/statistics.py` | Statistical tests and summaries |

## Time Zone Handling and UTC Normalization

All internal timestamps in GEO-INFER use UTC. Convert at the boundary (input/output), not inside analysis code.

```python
from datetime import datetime, timezone
import pandas as pd

# Input: convert local time to UTC
local_time = datetime(2025, 6, 15, 14, 30, tzinfo=timezone.utc)

# For pandas, use tz_localize then tz_convert
ts = pd.Timestamp("2025-06-15 14:30", tz="America/Los_Angeles")
ts_utc = ts.tz_convert("UTC")

# For a DataFrame with a datetime column
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

# When displaying results, convert back to local time
df["local_time"] = df["timestamp"].dt.tz_convert("America/Los_Angeles")
```

**Rules:**
- Store and compute in UTC
- Convert to local time only for display
- Always use timezone-aware datetime objects (never naive datetimes with assumed timezone)

## Trend Detection

The `TemporalAnalyzer` detects trends using linear regression, polynomial fitting, or moving average smoothing.

```python
from geo_infer_time.core.analysis import TemporalAnalyzer
from geo_infer_time.models.timeseries import TimeSeries
import numpy as np
import pandas as pd

# Create a time series with an upward trend and noise
dates = pd.date_range("2020-01-01", periods=365, freq="D")
values = np.linspace(10, 20, 365) + np.random.default_rng(42).normal(0, 1, 365)
ts = TimeSeries(timestamps=dates.tolist(), values=values.tolist())

analyzer = TemporalAnalyzer()

# Linear trend detection
trend_info = analyzer.detect_trend(ts, method="linear")
print(f"Direction: {trend_info['direction']}")       # "increasing"
print(f"Slope: {trend_info['slope']:.4f}")           # ~0.027 per day
print(f"R-squared: {trend_info['r_squared']:.4f}")   # strength of trend
```

## Seasonal Decomposition

Decompose a time series into trend, seasonal, and residual components. Requires `statsmodels`.

```python
# Decompose monthly temperature data
monthly_dates = pd.date_range("2015-01-01", periods=96, freq="M")
seasonal_pattern = 10 * np.sin(2 * np.pi * np.arange(96) / 12)
trend_component = np.linspace(15, 17, 96)
noise = np.random.default_rng(42).normal(0, 0.5, 96)
monthly_values = trend_component + seasonal_pattern + noise

ts = TimeSeries(timestamps=monthly_dates.tolist(), values=monthly_values.tolist())

decomp = analyzer.decompose(ts, period=12, model="additive")
# Returns: trend, seasonal, residual components
print(f"Seasonal amplitude: {decomp['seasonal'].max() - decomp['seasonal'].min():.2f}")
```

## Changepoint Detection

Detect points where the statistical properties of a time series change.

```python
def detect_changepoints(values: np.ndarray, min_segment: int = 20,
                         penalty: float = 3.0) -> list:
    """Detect changepoints using CUSUM (cumulative sum) method.

    Args:
        values: Time series values.
        min_segment: Minimum segment length between changepoints.
        penalty: Penalty factor for adding a changepoint (higher = fewer).

    Returns:
        List of changepoint indices.
    """
    n = len(values)
    cumsum = np.cumsum(values - np.mean(values))
    changepoints = []

    def _find_changepoint(start: int, end: int):
        if end - start < 2 * min_segment:
            return
        segment = cumsum[start:end] - np.linspace(cumsum[start], cumsum[end - 1], end - start)
        max_diff = np.max(np.abs(segment))
        if max_diff > penalty * np.std(values[start:end]) * np.sqrt(end - start):
            cp = start + np.argmax(np.abs(segment))
            changepoints.append(cp)
            _find_changepoint(start, cp)
            _find_changepoint(cp, end)

    _find_changepoint(0, n)
    return sorted(changepoints)


# Example: detect regime change
values = np.concatenate([
    np.random.default_rng(0).normal(5, 1, 100),
    np.random.default_rng(1).normal(8, 1, 100),
    np.random.default_rng(2).normal(5, 1, 100),
])
cps = detect_changepoints(values)
print(f"Changepoints at indices: {cps}")  # near 100 and 200
```

## Spatio-Temporal Autocorrelation

Measure how spatial and temporal proximity jointly affect correlation.

```python
def spatiotemporal_autocorrelation(
    locations: np.ndarray,
    times: np.ndarray,
    values: np.ndarray,
    spatial_lag: float,
    temporal_lag: float,
) -> float:
    """Compute spatio-temporal Moran's I for a given lag pair.

    Args:
        locations: shape (n, 2) spatial coordinates.
        times: shape (n,) timestamps as floats (e.g., days since epoch).
        values: shape (n,) observed values.
        spatial_lag: max spatial distance for neighbors.
        temporal_lag: max temporal distance for neighbors.
    """
    n = len(values)
    z = values - values.mean()

    numerator = 0.0
    w_sum = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            spatial_dist = np.sqrt(np.sum((locations[i] - locations[j]) ** 2))
            temporal_dist = abs(times[i] - times[j])

            if spatial_dist <= spatial_lag and temporal_dist <= temporal_lag:
                w = 1.0
                numerator += w * z[i] * z[j]
                w_sum += w

    if w_sum == 0:
        return 0.0

    denominator = np.sum(z ** 2) / n
    return (n * numerator) / (w_sum * denominator * n)
```

## Sliding Window Analysis

The `StreamProcessor` handles sliding, tumbling, and session windows for real-time data.

```python
from geo_infer_time.core.stream_processing import StreamProcessor
from datetime import datetime, timedelta

# 5-minute windows sliding every 1 minute
processor = StreamProcessor(
    window_size=timedelta(minutes=5),
    slide_interval=timedelta(minutes=1),
)

# Feed data points
import numpy as np
base_time = datetime(2025, 6, 15, 12, 0, 0)
for i in range(60):
    t = base_time + timedelta(seconds=i * 10)
    v = 20.0 + np.sin(i / 10) + np.random.default_rng(i).normal(0, 0.1)
    processor.add_data_point(timestamp=t, value=v)

# Retrieve computed windows
stats = processor.get_statistics()
print(f"Total points processed: {stats['total_points']}")
print(f"Windows computed: {stats['total_windows']}")
```

### Incremental Statistics

For streaming data, compute statistics incrementally to avoid re-scanning the buffer:

```python
class IncrementalStats:
    """Welford's online algorithm for running mean and variance."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, value: float) -> None:
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        delta2 = value - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        return self.m2 / self.n if self.n > 1 else 0.0

    @property
    def std(self) -> float:
        return np.sqrt(self.variance)
```

## Trajectory Analysis

For moving objects (vehicles, wildlife, ships), combine temporal and spatial analysis.

```python
import numpy as np
from dataclasses import dataclass

@dataclass
class TrajectoryPoint:
    timestamp: float  # seconds since epoch
    lat: float
    lng: float

def compute_trajectory_metrics(points: list[TrajectoryPoint]) -> dict:
    """Compute speed, distance, and bearing for a trajectory."""
    total_distance = 0.0
    speeds = []
    bearings = []

    for i in range(1, len(points)):
        dt = points[i].timestamp - points[i - 1].timestamp
        if dt <= 0:
            continue

        # Haversine distance (approximate for short distances)
        dlat = np.radians(points[i].lat - points[i - 1].lat)
        dlng = np.radians(points[i].lng - points[i - 1].lng)
        lat1 = np.radians(points[i - 1].lat)
        lat2 = np.radians(points[i].lat)

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        dist_m = 6371000 * c

        total_distance += dist_m
        speeds.append(dist_m / dt)

        # Bearing
        y = np.sin(dlng) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlng)
        bearing = np.degrees(np.arctan2(y, x)) % 360
        bearings.append(bearing)

    return {
        "total_distance_m": total_distance,
        "mean_speed_ms": np.mean(speeds) if speeds else 0.0,
        "max_speed_ms": np.max(speeds) if speeds else 0.0,
        "mean_bearing_deg": np.mean(bearings) if bearings else 0.0,
        "duration_s": points[-1].timestamp - points[0].timestamp if len(points) > 1 else 0.0,
    }
```

## Real-Time Stream Processing Patterns

### Pattern: Sensor Network Aggregation

```python
from geo_infer_time.core.stream_processing import StreamProcessor
from datetime import timedelta
from collections import defaultdict

class SpatioTemporalAggregator:
    """Aggregate streaming sensor data by H3 cell and time window."""

    def __init__(self, window_minutes: int = 5, h3_resolution: int = 7):
        self.h3_res = h3_resolution
        self.processors: dict = defaultdict(
            lambda: StreamProcessor(
                window_size=timedelta(minutes=window_minutes),
                slide_interval=timedelta(minutes=1),
            )
        )

    def ingest(self, timestamp, lat: float, lng: float, value: float):
        """Route a sensor reading to the appropriate cell processor."""
        import h3
        cell = h3.latlng_to_cell(lat, lng, self.h3_res)
        self.processors[cell].add_data_point(
            timestamp=timestamp, value=value, metadata={"cell": cell}
        )

    def get_cell_summary(self, cell: str) -> dict:
        if cell in self.processors:
            return self.processors[cell].get_statistics()
        return {}
```

### Pattern: Alerting on Anomalies

```python
from geo_infer_time.core.analysis import TemporalAnalyzer, AnomalyType

def check_for_alerts(values: np.ndarray, timestamps: list,
                      threshold_sigma: float = 3.0) -> list:
    """Check recent values for anomalies using z-score method."""
    if len(values) < 10:
        return []

    mean = np.mean(values[:-1])
    std = np.std(values[:-1])
    if std < 1e-8:
        return []

    latest = values[-1]
    z_score = abs(latest - mean) / std

    alerts = []
    if z_score > threshold_sigma:
        alerts.append({
            "timestamp": timestamps[-1],
            "value": float(latest),
            "expected": float(mean),
            "z_score": float(z_score),
            "severity": "high" if z_score > 5 else "medium",
        })
    return alerts
```

## Forecasting with Active Inference Priors

Active Inference provides a principled way to incorporate prior beliefs into forecasting. The generative model predicts future observations, and prediction errors update the model.

```python
from geo_infer_act.core.free_energy import FreeEnergyCalculator
import numpy as np

def forecast_with_prior(historical_values: np.ndarray,
                         prior_mean: float,
                         prior_precision: float,
                         horizon: int = 10) -> np.ndarray:
    """Forecast using a simple Active Inference generative model.

    Combines observed trend with prior beliefs about expected values.

    Args:
        historical_values: Past observations.
        prior_mean: Prior belief about the expected value.
        prior_precision: Confidence in the prior (higher = more weight on prior).
        horizon: Number of steps to forecast.
    """
    # Estimate trend from data
    n = len(historical_values)
    t = np.arange(n)
    slope, intercept = np.polyfit(t, historical_values, 1)
    data_precision = 1.0 / max(np.var(historical_values - (slope * t + intercept)), 1e-8)

    forecasts = np.empty(horizon)
    for h in range(horizon):
        future_t = n + h
        data_prediction = slope * future_t + intercept

        # Bayesian combination: weighted average of data trend and prior
        combined_precision = data_precision + prior_precision
        forecasts[h] = (
            data_precision * data_prediction + prior_precision * prior_mean
        ) / combined_precision

    return forecasts


# Example: temperature forecast with seasonal prior
winter_temps = np.array([2.0, 1.5, 3.0, 2.5, 1.0, 2.0, 3.5, 2.0])
# Prior: expect temperatures around 2 degrees in winter
forecasts = forecast_with_prior(
    winter_temps, prior_mean=2.0, prior_precision=2.0, horizon=5
)
print(f"Forecasts: {forecasts}")
```

## Integration: GEO-INFER-TIME + GEO-INFER-SPACE

### Spatio-Temporal Interpolation

Combine spatial GP interpolation with temporal trend models:

```python
from geo_infer_bayes.api.tfp_interface import TFPInterface
from geo_infer_time.core.analysis import TemporalAnalyzer
from geo_infer_time.models.timeseries import TimeSeries
import numpy as np

def spatiotemporal_interpolation(
    locations: np.ndarray,
    timestamps: np.ndarray,
    values: np.ndarray,
    query_location: np.ndarray,
    query_time: float,
) -> dict:
    """Interpolate a value at a given location and time.

    Strategy: fit GP spatially at the query time's temporal neighborhood,
    using temporally-detrended values.
    """
    analyzer = TemporalAnalyzer()

    # Step 1: Detrend at each location
    unique_locs = np.unique(locations, axis=0)
    detrended = np.copy(values)
    trend_at_query_time = {}

    for loc in unique_locs:
        mask = np.all(locations == loc, axis=1)
        loc_times = timestamps[mask]
        loc_values = values[mask]

        if len(loc_values) < 3:
            continue

        # Simple linear detrend
        coeffs = np.polyfit(loc_times, loc_values, 1)
        trend = np.polyval(coeffs, loc_times)
        detrended[mask] = loc_values - trend
        trend_at_query_time[tuple(loc)] = np.polyval(coeffs, query_time)

    # Step 2: Spatial GP on detrended values
    gp = TFPInterface(model_config={"lengthscale": 2.0, "variance": 1.0, "noise": 0.05})
    gp.create_spatial_gp_model(locations, detrended)

    from geo_infer_bayes.api.tfp_interface import _squared_exponential_kernel
    from scipy import linalg

    K_star = _squared_exponential_kernel(
        query_location.reshape(1, -1), gp._X, gp._lengthscale, gp._variance
    )
    spatial_residual = float(K_star @ gp._alpha)

    # Step 3: Combine spatial residual with temporal trend
    # Use nearest location's trend if query location is not in training set
    nearest_idx = np.argmin(np.sum((unique_locs - query_location) ** 2, axis=1))
    nearest_loc = tuple(unique_locs[nearest_idx])
    temporal_trend = trend_at_query_time.get(nearest_loc, np.mean(values))

    return {
        "predicted_value": temporal_trend + spatial_residual,
        "temporal_trend": temporal_trend,
        "spatial_residual": spatial_residual,
    }
```

### Time-Varying Spatial Fields

For data that changes over both space and time (e.g., air quality, temperature fields):

```python
def animate_spatial_field(timestamps: list,
                           locations: np.ndarray,
                           values_by_time: dict,
                           grid_resolution: int = 50):
    """Generate spatial field snapshots for each timestamp.

    Returns a list of (timestamp, grid_predictions) pairs for visualization.
    """
    gp = TFPInterface(model_config={"lengthscale": 3.0, "variance": 1.0, "noise": 0.1})

    x_range = np.linspace(locations[:, 0].min(), locations[:, 0].max(), grid_resolution)
    y_range = np.linspace(locations[:, 1].min(), locations[:, 1].max(), grid_resolution)
    grid = np.array(np.meshgrid(x_range, y_range)).reshape(2, -1).T

    snapshots = []
    for t in timestamps:
        if t not in values_by_time:
            continue
        y = values_by_time[t]
        gp.create_spatial_gp_model(locations, y)

        from geo_infer_bayes.api.tfp_interface import _squared_exponential_kernel
        K_star = _squared_exponential_kernel(grid, gp._X, gp._lengthscale, gp._variance)
        predictions = K_star @ gp._alpha

        snapshots.append((t, predictions.reshape(grid_resolution, grid_resolution)))

    return snapshots
```

## See Also

- [Bayesian Inference Guide](bayesian_inference_guide.md) -- GP models for spatial interpolation
- [Active Inference Guide](active_inference_guide.md) -- Active Inference fundamentals
- [Performance Optimization](advanced/performance_optimization.md) -- optimizing temporal computations
- [Custom Models](advanced/custom_models.md) -- building custom spatio-temporal models
