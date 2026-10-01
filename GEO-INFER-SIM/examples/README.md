# GEO-INFER-SIM/examples

Examples workspace within `GEO-INFER-SIM`.

## Contents

- `basic_abm.py`
- `module_simulations_example.py`
- `urban_growth_simulation.py`

## Public Interface

- `basic_abm.py:main` (function)
- `module_simulations_example.py:main` (function)
- `urban_growth_simulation.py:Developer` (class)
- `urban_growth_simulation.py:Household` (class)
- `urban_growth_simulation.py:initial_land_use` (function)
- `urban_growth_simulation.py:main` (function)

## Module Metadata

- Module: `GEO-INFER-SIM`
- Package: `geo_infer_sim`
- Version: `0.3.0`
- Install: `uv sync --package geo-infer-sim`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module SIM`

## Dependencies

- `numpy>=1.20.0`
- `pandas>=1.3.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module SIM
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
