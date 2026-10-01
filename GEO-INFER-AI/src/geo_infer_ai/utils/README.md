# GEO-INFER-AI/src/geo_infer_ai/utils

Utils workspace within `GEO-INFER-AI`.

## Contents

- `__init__.py`
- `rng.py`

## Public Interface

- `rng.py:resolve_rng` (function)
- `rng.py:resolve_optional_rng` (function)
- `rng.py:spawn_rng` (function)
- `rng.py:derive_int_seed` (function)

## Module Metadata

- Module: `GEO-INFER-AI`
- Package: `geo_infer_ai`
- Version: `0.3.0`
- Install: `uv pip install -e ./GEO-INFER-AI`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module AI`

## Dependencies

- `numpy>=1.20.0`
- `pandas>=1.3.0`
- `scikit-learn>=1.0.0`
- `h3>=4.5.0,<5`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module AI
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
