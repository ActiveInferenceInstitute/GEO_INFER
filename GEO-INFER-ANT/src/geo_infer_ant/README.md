# GEO-INFER-ANT/src/geo_infer_ant

Geo Infer Ant workspace within `GEO-INFER-ANT`.

## Contents

- `algorithms/`
- `analysis/`
- `applications/`
- `core/`
- `utils/`
- `__init__.py`

## Public Interface

- `__init__.py:setup_ant_module` (function)
- `__init__.py:get_available_components` (function)

## Module Metadata

- Module: `GEO-INFER-ANT`
- Package: `geo_infer_ant`
- Version: `0.3.0`
- Install: `uv sync --package geo-infer-ant`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module ANT`

## Dependencies

- `jsonschema>=4.0.0`
- `networkx>=2.8`
- `numpy>=1.21.0`
- `pyyaml>=6.0`
- `scikit-learn>=1.1.0`
- `scipy>=1.7.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module ANT
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
