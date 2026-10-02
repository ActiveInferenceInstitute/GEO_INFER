# GEO-INFER-SEC/src/geo_infer_sec/utils

Utils workspace within `GEO-INFER-SEC`.

## Contents

- `__init__.py`
- `geospatial_utils.py`
- `security_utils.py`

## Public Interface

- `geospatial_utils.py:GeoSpatialUtils` (class)
- `security_utils.py:SecurityConfig` (class)
- `security_utils.py:SecurityUtils` (class)
- `security_utils.py:create_security_utils` (function)
- `security_utils.py:hash_password_simple` (function)
- `security_utils.py:verify_password_simple` (function)
- `security_utils.py:generate_secure_token` (function)
- `security_utils.py:check_pii_columns` (function)
- `security_utils.py:validate_spatial_bounds` (function)
- `security_utils.py:detect_outliers` (function)

## Module Metadata

- Module: `GEO-INFER-SEC`
- Package: `geo_infer_sec`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-sec`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module SEC`

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
uv run python GEO-INFER-TEST/run_unified_tests.py --module SEC
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
