# GEO-INFER-MATH/src/geo_infer_math/api/convenience

Convenience workspace within `GEO-INFER-MATH`.

## Contents

- `__init__.py`
- `act_convenience.py`
- `ai_convenience.py`
- `bayes_convenience.py`
- `information_convenience.py`
- `integration_convenience.py`
- `spatial_convenience.py`

## Public Interface

- `act_convenience.py:free_energy_calculation` (function)
- `act_convenience.py:variational_inference_helper` (function)
- `act_convenience.py:belief_updating_helper` (function)
- `act_convenience.py:ActiveInferenceConvenience` (class)
- `ai_convenience.py:gradient_helper` (function)
- `ai_convenience.py:spatial_loss_function` (function)
- `ai_convenience.py:optimization_wrapper` (function)
- `ai_convenience.py:AIConvenience` (class)
- `bayes_convenience.py:posterior_helper` (function)
- `bayes_convenience.py:prior_builder` (function)
- `bayes_convenience.py:mcmc_wrapper` (function)
- `bayes_convenience.py:bayesian_optimization_helper` (function)
- `bayes_convenience.py:BayesianConvenience` (class)
- `information_convenience.py:spatial_entropy_helper` (function)
- `information_convenience.py:mutual_information_helper` (function)
- `information_convenience.py:kl_divergence_helper` (function)
- `information_convenience.py:InformationTheoryConvenience` (class)
- `integration_convenience.py:cross_module_helper` (function)
- `integration_convenience.py:IntegrationConvenience` (class)
- `spatial_convenience.py:enhanced_spatial_analysis` (function)

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
