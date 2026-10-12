# GEO-INFER-INSURANCE/src/geo_infer_insurance/underwriting

Underwriting workspace within `GEO-INFER-INSURANCE`.

## Contents

- `core/`
- `models/`
- `utils/`
- `__init__.py`

## Public Interface

- `__init__.py:underwrite_policy` (function)
- `__init__.py:process_claim` (function)
- `__init__.py:assess_risk` (function)
- `__init__.py:calculate_premium` (function)
- `__init__.py:create_pricing_engine` (function)
- `__init__.py:create_underwriting_engine` (function)
- `__init__.py:create_risk_assessment` (function)
- `__init__.py:create_policy_manager` (function)
- `__init__.py:create_claims_processor` (function)

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


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
