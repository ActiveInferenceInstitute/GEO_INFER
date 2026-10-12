# GEO-INFER-SEC/tests/integration

Integration workspace within `GEO-INFER-SEC`.

## Contents

- `test_sec_integration.py`
- `test_serialization_security.py`

## Public Interface

- No public Python symbols are defined directly in this directory.

## Module Metadata

- Module: `GEO-INFER-SEC`
- Package: `geo_infer_sec`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-sec`
- Tests: `uv run python -m pytest GEO-INFER-SEC/tests/integration`

## Dependencies

- `cryptography>=36.0.0`
- `flask>=2.0`
- `geopandas>=0.13.0`
- `h3>=4.5.0,<5`
- `numpy>=1.20.0`
- `pandas>=1.3.0`
- `pyjwt>=2.3.0`
- `pyyaml>=6.0`
- `shapely>=2.0.0`


## Validation

```bash
uv run python -m pytest GEO-INFER-SEC/tests/integration
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
