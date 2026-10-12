---
name: geo-infer-time
description: Time series analysis and temporal modeling for geospatial data. Use when analyzing temporal patterns, forecasting spatial time series, detecting change points, or working with spatio-temporal datasets.
prerequisites:
  required: []
  recommended:
    - geo-infer-data
    - geo-infer-space
    - geo-infer-iot
difficulty: intermediate
estimated_time: 45min
examples_dir: ../GEO-INFER-EXAMPLES/examples/
---

# GEO-INFER-TIME

## Instructions

### Core Capabilities

- **Time series analysis**: Seasonal decomposition, trend detection, seasonality
  detection, stationarity tests, autocorrelation, rolling statistics
- **Forecasting**: ARIMA/SARIMAX, Holt-Winters exponential smoothing, linear
  regression, moving average (statsmodels-backed)
- **Anomaly & changepoint detection**: z-score/IQR/rolling-z-score/isolation-forest
  anomaly detection; CUSUM and binary-segmentation changepoint detection
  (`TemporalAnalyzer.detect_change_points`), plus moving-window mean-shift
  changepoints (`EventDetector`)
- **UTC axes and storage**: Aware timestamp normalization, owned `TimeSeries`
  buffers and metadata, resampling, CSV/JSON/Parquet IO, and `InMemoryStore`
  queries by validated UTC bounds
- **Composition schedules**: Fixed model steps and explicit action histories
  through observation gaps; SPACE owns H3 state indexing and observation alignment

### Additional Public Surface

The package also exports `TemporalInterpolator`, `TemporalStatistics`,
`TemporalVisualization`, `AdvancedForecastingEngine`, and the `db`
(TimeSeriesStore/InMemoryStore), `io` (read/write_timeseries), and `utils`
(validate/align/fill_gaps) subpackages:

```python
from geo_infer_time import TemporalInterpolator, AdvancedForecastingEngine, db, io, utils
```

For sequential inference pipelines, `geo_infer_time.core.inference_schedule`
validates aware timestamps against an explicit fixed-interval model step:
off-grid, missing, or duplicated instants raise; the function never fills gaps
or silently resamples.

```python
from geo_infer_time.core.inference_schedule import inference_schedule
steps = inference_schedule(sensor_timestamps, step_seconds=60.0)
```

Use `geo_infer_time.core.action_schedule.action_observation_schedule` when
observations have gaps and every intervening action is supplied explicitly.
Read [action histories](docs/action_observation_schedule.md) before composing a
discrete inference model; the schedule does not choose policies or impute data.

### Key Imports

```python
from geo_infer_time import (
    TemporalAnalyzer, ForecastingEngine, EventDetector, TimeSeries,
    StreamProcessor, ReplayIngestAdapter, WebSocketIngestAdapter, KafkaIngestAdapter,
)
```

### Integrations

- Feed timestamped measurements from GEO-INFER-DATA or GEO-INFER-IOT into the explicit replay or live transport adapters.
- Combine temporal windows with GEO-INFER-SPACE H3 indices when records carry spatial identifiers.
- Pass processed windows to the anomaly and forecasting APIs described in the module documentation.
- Use the public `normalize_timestamp` and `normalize_datetime_index` helpers
  for UTC boundary validation; the scalar helper imports no analytical engines.
- Compose explicit H3/time axes with `geo_infer_space.align_h3_observations`;
  missing observations remain NaN and duplicate pairs fail.
- Select one value column before univariate analysis. Forecasting and sample-lag
  methods require complete observations and a regular UTC cadence. Resample or
  interpolate explicitly; missing samples are never silently dropped.
- Linear trend slopes count samples, not elapsed seconds. Use the returned
  `slope_per_sample` and `r_squared` for their distinct meanings.

## Examples

```python
import asyncio
from datetime import timedelta
from geo_infer_time import ReplayIngestAdapter, StreamProcessor

processor = StreamProcessor(timedelta(minutes=1))
records = [{"timestamp": "2024-01-01T00:00:00Z", "value": 21.5}]
assert asyncio.run(processor.ingest_adapter_stream(ReplayIngestAdapter(records))) == 1
window = processor.process_window()
assert window["aggregated_value"] == 21.5
```

## Guidelines

- Select `ReplayIngestAdapter` for recorded or offline input. Network adapters connect to real services and never supply replacement measurements.
- Install the TIME `streaming` extra for WebSocket and Kafka ingestion.
- Install `geo-infer-time[visualization]` for matplotlib figures; a base
  installation raises an explicit extra error when plotting is requested.
- Supply explicit, timezone-aware event timestamps (naive datetimes or offset-less ISO strings raise `ValueError`); output is timezone-aware UTC.
- Numeric transport timestamps require adapter `timestamp_unit="s"` or `"ms"`;
  TIME does not infer epoch units from magnitude.
- Configure `AdvancedForecastingEngine` through `arima_fit_kwargs`,
  `smoothing_fit_kwargs`, or `seasonal_period`. Unknown top-level configuration
  keys raise. Forecast validation refits the same model on a chronological
  prefix and reports errors on the untouched holdout.
- Read [streaming migration and delivery contracts](docs/streaming_migration.md) before changing callers.
- Read [UTC TimeSeries migration](docs/utc_timeseries_migration.md) when moving
  constructors, IO, query bounds or spatial composition to 0.4.0.
- Read [method contracts](docs/method_contracts.md) for finite-data requirements,
  undefined statistics, allocation/work limits, and backend failure behavior.
- Run `uv run python -m pytest GEO-INFER-TIME/tests/unit/test_method_contracts.py`
  for the small analytical and failure-boundary regressions first.
- Run `uv run python GEO-INFER-TEST/run_unified_tests.py --module TIME` for local verification.
- Run the explicit live Kafka service check against a disposable broker when validating network delivery.
