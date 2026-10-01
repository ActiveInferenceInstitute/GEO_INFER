# GEO-INFER-SEC/src/geo_infer_sec

Geo Infer Sec workspace within `GEO-INFER-SEC`.

## Contents

- `api/`
- `config/`
- `core/`
- `models/`
- `utils/`
- `__init__.py`
- `cli.py`

## Public Interface

- `__init__.py:SecurityFramework` (class)
- `cli.py:configure_logging` (function)
- `cli.py:setup_parser` (function)
- `cli.py:load_geospatial_data` (function)
- `cli.py:save_geospatial_data` (function)
- `cli.py:command_anonymize` (function)
- `cli.py:command_encrypt` (function)
- `cli.py:command_decrypt` (function)
- `cli.py:command_check_compliance` (function)
- `cli.py:command_audit` (function)
- `cli.py:command_risk_assessment` (function)
- `cli.py:command_generate_token` (function)
- `cli.py:main` (function)

## Module Metadata

- Module: `GEO-INFER-SEC`
- Package: `geo_infer_sec`
- Version: `0.3.0`
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
