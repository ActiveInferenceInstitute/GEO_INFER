# GEO-INFER-EMERGENCY/examples

Examples workspace within `GEO-INFER-EMERGENCY`.

## Contents

- `emergency_response_simulation.py`
- `multi_hazard_assessment.py`

## Public Interface

- `emergency_response_simulation.py:main` (function)
- `multi_hazard_assessment.py:main` (function)

## Module Metadata

- Module: `GEO-INFER-EMERGENCY`
- Package: `geo_infer_emergency`
- Version: `0.3.0`
- Install: `uv sync --package geo-infer-emergency`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module EMERGENCY`

## Dependencies

- `networkx>=2.6.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module EMERGENCY
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
