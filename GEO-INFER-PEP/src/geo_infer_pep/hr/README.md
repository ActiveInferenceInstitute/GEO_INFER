# GEO-INFER-PEP/src/geo_infer_pep/hr

Hr workspace within `GEO-INFER-PEP`.

## Contents

- `__init__.py`
- `importer.py`
- `transformer.py`

## Public Interface

- `importer.py:BaseHRImporter` (class)
- `importer.py:CSVHRImporter` (class)
- `transformer.py:clean_employee_data` (function)
- `transformer.py:enrich_employee_data` (function)
- `transformer.py:convert_employees_to_dataframe` (function)

## Module Metadata

- Module: `GEO-INFER-PEP`
- Package: `geo_infer_pep`
- Version: `0.3.0`
- Install: `uv sync --package geo-infer-pep`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module PEP`

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
uv run python GEO-INFER-TEST/run_unified_tests.py --module PEP
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
