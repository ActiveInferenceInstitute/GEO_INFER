# GEO-INFER-INSURANCE/tests/unit

Unit workspace within `GEO-INFER-INSURANCE`.

## Contents

- `test_compliance_engine.py`
- `test_data_integration_fetches.py`
- `test_identifier_generation.py`
- `test_models.py`
- `test_package_safety.py`
- `test_policy_lifecycle.py`
- `test_portfolio_management.py`
- `test_pricing_and_claims.py`
- `test_silent_fabrication_fixes.py`
- `test_underwriting_decisions.py`
- `test_underwriting_rules.py`

## Public Interface

- No public Python symbols are defined directly in this directory.

## Module Metadata

- Module: `GEO-INFER-INSURANCE`
- Package: `geo_infer_insurance`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-insurance`
- Tests: `uv run python -m pytest GEO-INFER-INSURANCE/tests/unit`

## Dependencies

- `numpy>=1.20.0`
- `pandas>=1.3.0`


## Validation

```bash
uv run python -m pytest GEO-INFER-INSURANCE/tests/unit
```


## Unit Tests

- Eleven tracked unit tests cover underwriting, pricing, claims, compliance,
  identifiers, and portfolio logic without external services.
- Run `uv run python -m pytest GEO-INFER-INSURANCE/tests/unit`.

## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
