# GEO-INFER-MATH/src/geo_infer_math/integration/bayes

Bayes workspace within `GEO-INFER-MATH`.

## Contents

- `__init__.py`
- `bayesian_optimization.py`
- `mcmc_helpers.py`
- `model_selection.py`
- `posterior_helpers.py`
- `prior_builders.py`

## Public Interface

- `bayesian_optimization.py:BayesianOptimization` (class)
- `mcmc_helpers.py:MCMCHelpers` (class)
- `model_selection.py:ModelSelection` (class)
- `posterior_helpers.py:PosteriorHelpers` (class)
- `prior_builders.py:PriorBuilders` (class)

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
