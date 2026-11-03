# GEO-INFER-INSURANCE/src/geo_infer_insurance

Geo Infer Insurance workspace within `GEO-INFER-INSURANCE`.

## Contents

- `underwriting/`
- `__init__.py`

## Public Interface

- `__init__.py:create_underwriting_system` (function)
- `__init__.py:underwrite_insurance_policy` (function)
- `__init__.py:process_insurance_claim` (function)

## Module Metadata

- Module: `GEO-INFER-INSURANCE`
- Package: `geo_infer_insurance`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-insurance`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module INSURANCE`

## Dependencies

- `numpy>=1.20.0`
- `pandas>=1.3.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module INSURANCE
```


## Package Layout

- `__init__.py` exports the package surface; implementation lives in the
  `underwriting/` subpackage.
- Run module tests with
  `uv run python GEO-INFER-TEST/run_unified_tests.py --module INSURANCE`.

## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
