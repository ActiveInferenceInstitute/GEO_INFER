# GEO-INFER-IOT/tests/unit

Unit workspace within `GEO-INFER-IOT`.

## Contents

- `test_calibration_topology_sensor_api.py`
- `test_data_ingestion.py`
- `test_fixwave_regressions.py`
- `test_inference_behavior.py`
- `test_ingest_error_and_latency.py`
- `test_ingestion.py`
- `test_optional_visualization.py`
- `test_performance_monitor.py`
- `test_quality_control.py`
- `test_quality_control_history.py`
- `test_radiation_monitoring.py`
- `test_registry.py`
- `test_sensor_api.py`
- `test_sensor_data.py`
- `test_streaming_forward.py`
- `test_timestamp_spatial_contracts.py`
- `test_visualization.py`

## Public Interface

- No public Python symbols are defined directly in this directory.

## Module Metadata

- Module: `GEO-INFER-IOT`
- Package: `geo_infer_iot`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-iot`
- Tests: `uv run python -m pytest GEO-INFER-IOT/tests/unit`

## Dependencies

- `aiomqtt>=2.4.0`
- `fastapi>=0.100.0`
- `geo-infer-bayes>=0.4.0`
- `geo-infer-space>=0.4.0`
- `geo-infer-time>=0.4.0`
- `h3>=4.5.0,<5`
- `networkx>=2.6`
- `numpy>=1.20.0`
- `paho-mqtt>=1.6.0`
- `pandas>=1.3.0`
- `pydantic>=2.0.0`
- `pyyaml>=6.0`
- `scikit-learn>=1.0.0`
- `scipy>=1.7.0`
- `uvicorn>=0.15.0`
- `websockets>=10.0`


## Validation

```bash
uv run python -m pytest GEO-INFER-IOT/tests/unit
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
