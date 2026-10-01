# GEO-INFER-MATH/src/geo_infer_math/integration/act

Act workspace within `GEO-INFER-MATH`.

## Contents

- `__init__.py`
- `belief_updating.py`
- `free_energy.py`
- `generative_models.py`
- `policy_optimization.py`
- `variational_inference.py`

## Public Interface

- `belief_updating.py:BeliefUpdating` (class)
- `free_energy.py:FreeEnergyCalculator` (class)
- `generative_models.py:GenerativeModels` (class)
- `policy_optimization.py:PolicyOptimization` (class)
- `variational_inference.py:VariationalInferenceHelpers` (class)

## Module Metadata

- Module: `GEO-INFER-MATH`
- Package: `geo_infer_math`
- Version: `0.3.0`
- Install: `uv sync --package geo-infer-math`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module MATH`

## Dependencies

- `numpy>=1.20.0`
- `scipy>=1.7.0`
- `pandas>=1.3.0`
- `scikit-learn>=1.0.0`
- `sympy>=1.9.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module MATH
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
