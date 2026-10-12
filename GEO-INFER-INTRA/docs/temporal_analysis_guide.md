# Temporal analysis and SPACE/TIME composition

TIME owns timestamp normalization, `TimeSeries`, temporal analysis, forecasting,
stream processing, and model schedules. SPACE owns ordered H3 domains and the
`align_h3_observations` adapter. Application scripts select these interfaces and
configuration; numerical implementations belong in the owning packages.

## Install and choose an interface

Use the root uv workspace and lock when developing in this checkout. Installed
applications can depend on `geo-infer-time` and `geo-infer-space`. Add TIME's
`visualization` extra for matplotlib and `streaming` extra for WebSocket/Kafka
clients. Importing a transport client does not establish a live service connection.

| Interface | Purpose and assumptions |
| --- | --- |
| `TimeSeries` | Owned tabular observations and metadata with a unique, increasing UTC axis; missing values are allowed. |
| `TemporalAnalyzer` | Univariate trend, seasonality, decomposition, anomaly and change-point analysis. Individual methods declare finite-data and regular-cadence requirements. |
| `ForecastingEngine` | Linear, moving-average, ARIMA and Holt–Winters forecasts on complete, regular univariate observations. |
| `AdvancedForecastingEngine` | Explicit statsmodels fitting options and seasonal periods; unknown top-level configuration keys fail. |
| `TemporalInterpolator` | Explicit gap filling or resampling; imputed data carry interpolation metadata. |
| `EventDetector` | Observed-value anomaly detection and complete-series change-point windows. |
| `StreamProcessor` | Bounded event-time buffers, watermark policies, window aggregation and handlers. |
| `inference_schedule` | Fixed-step validation with no inferred predictions or actions. |
| `action_observation_schedule` | Observation gaps with exactly the supplied intervening action history. |

## UTC identity at the boundary

Supply an offset or an aware datetime. Never pass a naive local wall clock through
`pd.to_datetime(..., utc=True)` to guess its timezone. Localize the source timezone
explicitly, resolve ambiguous/nonexistent daylight-saving times deliberately, and
then call TIME's helper. Numeric transport epochs require an explicit unit at the
transport adapter; the scalar normalization helper rejects them.

```python
from datetime import datetime
from zoneinfo import ZoneInfo
from geo_infer_time import normalize_datetime_index, normalize_timestamp

assert normalize_timestamp("2024-01-01T00:00:00Z") == normalize_timestamp(
    "2023-12-31T16:00:00-08:00"
)
local_zone = ZoneInfo("America/Los_Angeles")
fold_axis = normalize_datetime_index([
    datetime(2024, 11, 3, 1, 30, tzinfo=local_zone, fold=0),
    datetime(2024, 11, 3, 1, 30, tzinfo=local_zone, fold=1),
])
assert (fold_axis[1] - fold_axis[0]).total_seconds() == 3600
assert str(fold_axis.tz) == "UTC"
```

`normalize_datetime_index` preserves pandas nanoseconds. It rejects normalized
duplicates, reversed instants, NaT and numeric epochs rather than sorting,
coalescing or repairing the input. `TimeSeries` constructors, IO and query bounds
use this contract. Convert UTC to a local timezone only when displaying results.

## A trend and forecast with explicit units

Linear trends count samples. A slope for quarter-hour measurements is a change
per quarter-hour sample; dividing by 900 converts that rate to units per second.
The reported `r_squared` describes the fitted data, without establishing future
forecast accuracy.

```python
import numpy as np
import pandas as pd
from geo_infer_time import ForecastingEngine, TemporalAnalyzer, TimeSeries

axis = pd.date_range("2024-01-01", periods=8, freq="15min", tz="UTC")
series = TimeSeries(
    -5.0 + 0.25 * np.arange(8), timestamps=axis,
    metadata={"measurement_unit": "degC", "source": "analytical-fixture"},
)
trend = TemporalAnalyzer().detect_trend(series, method="linear")
assert trend["trend_direction"] == "increasing"
assert np.isclose(trend["slope_per_sample"], 0.25)
assert np.isclose(trend["r_squared"], 1.0)
rate_degC_per_second = trend["slope_per_sample"] / 900.0
assert np.isclose(rate_degC_per_second, 1.0 / 3600.0)
forecast = ForecastingEngine().forecast_linear(series, horizon=2)
np.testing.assert_allclose(forecast["forecast"], [-3.0, -2.75])
assert pd.Timestamp(forecast["timestamps"][0]) == axis[-1] + pd.Timedelta(minutes=15)
```

Seasonal periods and forecast horizons count samples. FFT frequencies are cycles
per sample. Use a chronological holdout to evaluate a fitted forecast; TIME's
forecast validator refits the same declared model on a prefix and scores the
untouched suffix. ARIMA convergence diagnostics and forecast intervals are
backend/model-specific results, not universal guarantees.

## Missing observations and ordered spatial columns

Long-form observations use the same explicit axis and H3 state order everywhere.
An observed zero and an absent observation have different meanings. Metadata
such as source and measurement unit follows the returned owned series.

```python
import h3
from geo_infer_space import H3StateSpace, align_h3_observations

center = h3.latlng_to_cell(41.75, -124.2, 8)
neighbor = sorted(set(h3.grid_disk(center, 1)) - {center})[0]
state_space = H3StateSpace([neighbor, center])
times = ["2024-01-01T00:00:00Z", "2024-01-01T00:01:00Z", "2024-01-01T00:02:00Z"]
records = pd.DataFrame([
    {"cell": center, "timestamp": times[1], "value": 4.0},
    {"cell": neighbor, "timestamp": times[0], "value": 0.0},
    {"cell": center, "timestamp": "2023-12-31T16:00:00-08:00", "value": 2.0},
])
records.attrs = {"source": "analytical-fixture", "measurement_unit": "degC"}
aligned = align_h3_observations(
    records, state_space=state_space, timestamps=times, max_entries=6,
)
assert list(aligned.data.columns) == [neighbor, center]
np.testing.assert_allclose(aligned.data.to_numpy(), [[0.0, 2.0], [np.nan, 4.0], [np.nan, np.nan]], equal_nan=True)
assert aligned.metadata["measurement_unit"] == "degC"
minute_sums = aligned.resample("1min", method="sum")
assert minute_sums.data.iloc[0, 0] == 0.0
assert np.isnan(minute_sums.data.iloc[1, 0])
assert minute_sums.data.iloc[2].isna().all()
```

The adapter rejects duplicate cell/time pairs, unknown cells/times, mixed
resolutions and nonfinite observed values before constructing the bounded matrix.
TIME's univariate analytical engines require a selected column. Choose and record
an interpolation policy before invoking methods requiring complete data;
`TemporalInterpolator.interpolate(..., method="time")` uses elapsed-time spacing,
whereas `method="linear"` uses positional spacing. Interpolation is a modeling
choice, not an observed measurement. `align_timeseries` defaults to forward
filling, so pass `fill_method=None` when retaining missingness is required.

## Bounded replay and live delivery

Replay is an explicit offline source. Live adapters open actual services and
propagate failures. Window values use the configured aggregation callback;
active buffers, late buffers, retained history and sliding work each have limits.

```python
import asyncio
from datetime import timedelta
from geo_infer_time import ReplayIngestAdapter, StreamProcessor

processor = StreamProcessor(timedelta(minutes=1), max_buffer_points=10)
replay = ReplayIngestAdapter([
    {"timestamp": "2024-01-01T00:00:00Z", "value": 2.0},
    {"timestamp": "2024-01-01T00:00:20Z", "value": 4.0},
])

async def ingest_replay():
    async with asyncio.timeout(5):
        return await processor.ingest_adapter_stream(replay)

assert asyncio.run(ingest_replay()) == 2
window = processor.process_window()
assert window["aggregated_value"] == 3.0
assert processor.get_stats()["total_points"] == 2
```

Configure lateness and watermark behavior explicitly. Kafka acknowledgement
occurs only after successful downstream processing. Client-boundary tests do not
replace a disposable live-broker acceptance check.

## Model time and action histories

General observations may be irregular, but a discrete model's transition step is
an explicit physical duration. A missing observation does not remove an elapsed
transition. Supply the actions taken during the gap; schedules validate them and
do not choose policies or insert observations.

```python
from geo_infer_time.core.inference_schedule import inference_schedule
from geo_infer_time.core.action_schedule import action_observation_schedule

regular = inference_schedule(times[:2], step_seconds=60)
assert len(regular) == 2
schedule = action_observation_schedule([
    {"timestamp": times[0], "actions": []},
    {"timestamp": times[2], "actions": [0, 1]},
], step_seconds=60, num_actions=2)
assert schedule[1].prediction_count == 2
assert schedule[1].prediction_timestamps[-1] == pd.Timestamp(times[2])
```

For BAYES, declare centroid coordinate order and coordinate scale, apply the
observed-value mask, and derive elapsed seconds from the UTC axis. For ACT,
preserve the H3 order in priors, likelihoods and transitions; measurements must
be mapped through a declared observation model rather than treated as posterior
probabilities. See [the composition contract](../../GEO-INFER-SPACE/docs/CROSS_MODULE_COMPOSITION.md),
[the TIME method contracts](../../GEO-INFER-TIME/docs/method_contracts.md),
[the action schedule](../../GEO-INFER-TIME/docs/action_observation_schedule.md),
and [the Active Inference guide](active_inference_guide.md).

## Verification

From the repository root, run the narrow numerical and composition tests first:

```bash
uv run python -m pytest GEO-INFER-TIME/tests/unit/test_method_contracts.py
uv run python -m pytest GEO-INFER-TEST/tests/integration/test_space_time_composition_contract.py
uv run python GEO-INFER-TEST/run_unified_tests.py --module TIME
uv run python GEO-INFER-TEST/validate_doc_examples.py
```

Plotting requires TIME's visualization extra. Forecast dashboards accept the
`ForecastingEngine` result directly and require forecast timestamps to follow the
historical axis. Rendered figures and live services need their own retained
acceptance; file inventory or client construction cannot establish those outcomes.
