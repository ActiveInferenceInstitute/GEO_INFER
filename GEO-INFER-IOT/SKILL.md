---
name: geo-infer-iot
description: IoT sensor data ingestion and real-time streaming for geospatial monitoring. Use when connecting to MQTT brokers, processing sensor streams, validating spatial sensor data quality, or building real-time geospatial monitoring pipelines.
prerequisites:
  required:
    - geo-infer-data
    - geo-infer-time
  recommended:
    - geo-infer-space
difficulty: intermediate
estimated_time: 45min
examples_dir: ../GEO-INFER-EXAMPLES/examples/
---

# GEO-INFER-IOT

## Instructions

### Core Capabilities

- **Ingestion**: `IoTDataIngestion` — dict/SensorMeasurement ingestion, H3 spatial indexing, MQTT (paho-mqtt thread bridge + aiomqtt async), Bayesian spatial inference via GEO-INFER-BAYES
- **Registry**: `SensorRegistry` — sensor/network registration (`SensorMetadata`, `SensorNetworkRecord`) with H3 spatial queries
- **Streaming**: `StreamingAPI` — FastAPI WebSocket `/ws/sensor-stream` with async broadcasts; `SensorAPI` — REST for sensors/measurements/networks
- **Quality control**: `QualityController` — range validation, temporal consistency, outlier detection (IsolationForest over the retained window plus a running 3-sigma z-score check), spatial consistency
- **Bayesian inference**: `BayesianSpatialInference` (in `geo_infer_iot.core.inference`) — Gaussian-process spatial posteriors over an H3 grid
- **Radiation monitoring**: `RadiationMonitoringSystem` — end-to-end radiation ingestion, anomaly detection, and spatial inference

### Key Imports

```python
from geo_infer_iot import (
    IoTDataIngestion,
    SensorRegistry,
    QualityController,
    IoTSystem,
    RadiationMonitoringSystem,
)
from geo_infer_iot.core.inference import BayesianSpatialInference
```

## Examples

```python
from geo_infer_iot import SensorRegistry, IoTDataIngestion

registry = SensorRegistry()
ingestion = IoTDataIngestion(registry, {"mqtt": {"host": "localhost", "port": 1883}})

measurement = {
    "sensor_id": "temp_001",
    "variable": "temperature",
    "value": 21.5,
    "unit": "celsius",
    "latitude": 47.6062,
    "longitude": -122.3321,
}

import asyncio
asyncio.run(ingestion.ingest_measurement(measurement))
print(ingestion.get_measurement_statistics())
```

```python
from geo_infer_iot import QualityController

checker = QualityController({
    "variable_ranges": {"temperature": (-40.0, 60.0)},
    "temporal_consistency": {"max_change_rate": 0.1},
})
result = checker.validate_measurement({
    "sensor_id": "temp_001",
    "variable": "temperature",
    "value": 21.5,
    "timestamp": "2026-01-01T00:00:00+00:00",
})
print(f"Passed: {result.passed}, score: {result.quality_score:.2f}, issues: {result.issues}")
```

## Guidelines

- MQTT handlers use real paho-mqtt (thread bridge) or aiomqtt (async); broker connection happens in `start_stream_processing` inside a running event loop
- geo_infer_space, geo_infer_bayes, scipy, scikit-learn, folium, matplotlib, paho-mqtt and aiomqtt are hard dependencies — they import unconditionally (no availability flags or degraded fallbacks) and fail loudly on a broken install
- Timestamps are timezone-aware UTC end to end (same contract as GEO-INFER-TIME `normalize_timestamp`): `QualityController.validate_measurement`, `IoTDataIngestion` dict input, `SensorMeasurement` and the `Measurement` models accept aware datetimes or ISO-8601 strings with an offset (`Z` included), normalize them to UTC, and raise `ValueError` on naive or malformed input (`TypeError` for non-string, non-datetime values); helper: `geo_infer_iot.models.measurement.normalize_timestamp`. A missing timestamp defaults to the current UTC time on ingestion and skips temporal checks in quality control
- Quality-control config nests under documented keys (e.g. `temporal_consistency.max_change_rate`); flat top-level keys such as `max_change_rate` are not read
- Test: `uv run python GEO-INFER-TEST/run_unified_tests.py --module IOT`

### Integrations

- **DATA** → Sensor data feeds into ETL pipelines
- **TIME** → Time-series analysis of sensor streams
- **SPACE** → Spatial indexing of sensor locations
- **OPS** → Monitoring sensor health and uptime (`PredictiveMaintenance`)
- **COMMS** → Alert broadcasting on sensor thresholds

