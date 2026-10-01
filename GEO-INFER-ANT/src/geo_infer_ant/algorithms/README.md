# GEO-INFER-ANT/src/geo_infer_ant/algorithms

Algorithms workspace within `GEO-INFER-ANT`.

## Contents

- `__init__.py`
- `abc.py`
- `aco.py`
- `pso.py`

## Public Interface

- `abc.py:ABCParameters` (class)
- `abc.py:FoodSource` (class)
- `abc.py:ArtificialBeeColony` (class)
- `aco.py:ACOParameters` (class)
- `aco.py:OptimizationResult` (class)
- `aco.py:AntColonyOptimization` (class)
- `pso.py:PSOParameters` (class)
- `pso.py:Particle` (class)
- `pso.py:ParticleSwarmOptimization` (class)

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
