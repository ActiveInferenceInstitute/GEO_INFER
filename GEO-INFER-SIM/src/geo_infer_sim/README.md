# GEO-INFER-SIM/src/geo_infer_sim

Geo Infer Sim workspace within `GEO-INFER-SIM`.

## Contents

- `core/`
- `paradigms/`
- `scenarios/`
- `__init__.py`
- `module_simulations.py`

## Public Interface

- `module_simulations.py:ModuleSimulationConfig` (class)
- `module_simulations.py:ModuleSimulations` (class)

## Module Metadata

- Module: `GEO-INFER-SIM`
- Package: `geo_infer_sim`
- Version: `0.4.0`
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
