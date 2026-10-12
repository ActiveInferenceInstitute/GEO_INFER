# GEO-INFER-PEP/tests/unit

Unit workspace within `GEO-INFER-PEP`.

## Contents

- `test_api_error_handling.py`
- `test_crm.py`
- `test_crm_models.py`
- `test_crm_talent_endpoints.py`
- `test_hr.py`
- `test_hr_models.py`
- `test_methods.py`
- `test_orchestrator.py`
- `test_pep_engine.py`
- `test_performance_reviews_api.py`
- `test_store_backed_endpoints.py`
- `test_talent.py`
- `test_talent_models.py`
- `test_validator.py`
- `test_visualizations_import_purity.py`

## Public Interface

- No public Python symbols are defined directly in this directory.

## Module Metadata

- Module: `GEO-INFER-PEP`
- Package: `geo_infer_pep`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-pep`
- Tests: `uv run python -m pytest GEO-INFER-PEP/tests/unit`

## Dependencies

- `fastapi>=0.100.0`
- `starlette>=0.27.0`
- `uvicorn[standard]>=0.23.2`
- `pydantic>=2.0`
- `pandas>=2.0`
- `matplotlib>=3.7.0`
- `seaborn>=0.13.0`


## Validation

```bash
uv run python -m pytest GEO-INFER-PEP/tests/unit
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
