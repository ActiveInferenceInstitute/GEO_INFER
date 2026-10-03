# GEO-INFER-TIME/src/geo_infer_time/core

Core workspace within `GEO-INFER-TIME`.

## Contents

- `__init__.py`
- `_validation.py`
- `action_schedule.py`
- `advanced_forecasting.py`
- `analysis.py`
- `event_detection.py`
- `forecasting.py`
- `inference_schedule.py`
- `interpolation.py`
- `statistics.py`
- `stream_ingest.py`
- `stream_processing.py`
- `timestamps.py`
- `visualization.py`

## Public Interface

- `action_schedule.py:ScheduledObservation` (class)
- `action_schedule.py:action_observation_schedule` (function)
- `advanced_forecasting.py:fit_arima_forecast` (function)
- `advanced_forecasting.py:fit_exponential_smoothing_forecast` (function)
- `advanced_forecasting.py:AdvancedForecastingEngine` (class)
- `analysis.py:AnomalyType` (class)
- `analysis.py:Anomaly` (class)
- `analysis.py:TemporalAnalyzer` (class)
- `event_detection.py:EventDetector` (class)
- `forecasting.py:ForecastingEngine` (class)
- `inference_schedule.py:inference_schedule` (function)
- `interpolation.py:TemporalInterpolator` (class)
- `statistics.py:TemporalStatistics` (class)
- `stream_ingest.py:StreamIngestAdapter` (class)
- `stream_ingest.py:ReplayIngestAdapter` (class)
- `stream_ingest.py:WebSocketIngestAdapter` (class)
- `stream_ingest.py:KafkaIngestAdapter` (class)
- `stream_processing.py:StreamProcessor` (class)
- `timestamps.py:normalize_timestamp` (function)
- `timestamps.py:normalize_datetime_index` (function)

## Module Metadata

- Module: `GEO-INFER-TIME`
- Package: `geo_infer_time`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-time`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module TIME`

## Dependencies

- `numpy>=1.20.0`
- `pandas>=1.3.0`
- `scikit-learn>=1.6.1`
- `scipy>=1.7.0`
- `statsmodels>=0.13.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module TIME
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
