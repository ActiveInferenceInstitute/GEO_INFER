# GEO-INFER-SEC/tests/unit

Unit workspace within `GEO-INFER-SEC`.

## Contents

- `test_acceptance_sec.py`
- `test_access_control.py`
- `test_audit.py`
- `test_audit_logging.py`
- `test_authentication.py`
- `test_authorization.py`
- `test_cli_exit_codes.py`
- `test_cli_handlers.py`
- `test_digital_indicators.py`
- `test_digital_packaged_config.py`
- `test_encryption.py`
- `test_geospatial_utils.py`
- `test_input_validation.py`
- `test_integrated_security.py`
- `test_physical_security.py`
- `test_security_api.py`
- `test_token_lifecycle.py`

## Public Interface

- No public Python symbols are defined directly in this directory.

## Module Metadata

- Module: `GEO-INFER-SEC`
- Package: `geo_infer_sec`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-sec`
- Tests: `uv run python -m pytest GEO-INFER-SEC/tests/unit`

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
uv run python -m pytest GEO-INFER-SEC/tests/unit
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
