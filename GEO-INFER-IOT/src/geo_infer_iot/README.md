# GEO-INFER-IOT/src/geo_infer_iot

Geo Infer Iot workspace within `GEO-INFER-IOT`.

## Contents

- `api/`
- `core/`
- `models/`
- `performance/`
- `utils/`
- `__init__.py`

## Public Interface

- No public Python symbols are defined directly in this directory.

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
