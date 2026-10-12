# GEO-INFER-BAYES/src/geo_infer_bayes/core

Core workspace within `GEO-INFER-BAYES`.

## Contents

- `__init__.py`
- `abc.py`
- `evaluation.py`
- `hmc.py`
- `inference.py`
- `mcmc.py`
- `model_comparison.py`
- `posterior.py`
- `smc.py`
- `variational.py`

## Public Interface

- `abc.py:ApproximateBayesianComputation` (class)
- `evaluation.py:crps` (function)
- `evaluation.py:crps_pointwise` (function)
- `evaluation.py:crps_gaussian` (function)
- `evaluation.py:pinball_loss` (function)
- `evaluation.py:empirical_coverage` (function)
- `evaluation.py:coverage_calibration_error` (function)
- `evaluation.py:interval_score` (function)
- `evaluation.py:pit_values` (function)
- `evaluation.py:pit_gaussian` (function)
- `evaluation.py:pit_uniformity_statistic` (function)
- `evaluation.py:log_predictive_density` (function)
- `evaluation.py:log_predictive_density_pointwise` (function)
- `evaluation.py:log_predictive_density_gaussian` (function)
- `evaluation.py:evaluate_predictive` (function)
- `evaluation.py:evaluate_gaussian` (function)
- `hmc.py:HMC` (class)
- `inference.py:BayesianInference` (class)
- `mcmc.py:MCMC` (class)
- `model_comparison.py:ModelComparison` (class)

## Module Metadata

- Module: `GEO-INFER-BAYES`
- Package: `geo_infer_bayes`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-bayes`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module BAYES`

## Dependencies

- `arviz>=0.12.0`
- `matplotlib>=3.5.0`
- `numpy>=1.20.0,<2.0`
- `pandas>=1.3.0`
- `scipy>=1.7.0`
- `tqdm>=4.60.0`
- `xarray>=2022.3.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module BAYES
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
