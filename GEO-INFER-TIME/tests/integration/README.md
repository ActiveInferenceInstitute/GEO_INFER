# GEO-INFER-TIME/tests/integration

Integration workspace within `GEO-INFER-TIME`.

## Contents

- `kafka_service_check.py`
- `test_integration.py`

## Public Interface

- `kafka_service_check.py:check_broker` (function)

## Module Metadata

- Module: `GEO-INFER-TIME`
- Package: `geo_infer_time`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-time`
- Tests: `uv run python -m pytest GEO-INFER-TIME/tests/integration`

## Dependencies

- `numpy>=1.20.0`
- `pandas>=1.3.0`
- `scikit-learn>=1.6.1`
- `scipy>=1.7.0`
- `statsmodels>=0.13.0`


## Validation

```bash
uv run python -m pytest GEO-INFER-TIME/tests/integration
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
