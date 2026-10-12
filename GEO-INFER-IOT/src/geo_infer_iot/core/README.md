# GEO-INFER-IOT/src/geo_infer_iot/core

Core workspace within `GEO-INFER-IOT`.

## Contents

- `inference.py`
- `ingestion.py`
- `quality_control.py`
- `registry.py`
- `spatial_fusion.py`
- `systems.py`

## Public Interface

- `inference.py:BayesianSpatialInference` (class)
- `ingestion.py:SpatialOperations` (class)
- `ingestion.py:SensorMeasurement` (class)
- `ingestion.py:SpatialInferenceConfig` (class)
- `ingestion.py:IoTDataIngestion` (class)
- `ingestion.py:RadiationMonitoringSystem` (class)
- `ingestion.py:GlobalRadiationMonitor` (class)
- `quality_control.py:QualityCheckResult` (class)
- `quality_control.py:QualityController` (class)
- `registry.py:SensorMetadata` (class)
- `registry.py:SensorNetworkRecord` (class)
- `registry.py:SensorRegistry` (class)
- `spatial_fusion.py:SpatialDataFusion` (class)
- `systems.py:IoTSystem` (class)
- `systems.py:GlobalMonitoringSystem` (class)
- `systems.py:MultiModalFusion` (class)
- `systems.py:AdaptiveSampling` (class)
- `systems.py:PredictiveMaintenance` (class)

## Module Metadata

- Module: `GEO-INFER-IOT`
- Package: `geo_infer_iot`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-iot`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module IOT`

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
uv run python GEO-INFER-TEST/run_unified_tests.py --module IOT
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
