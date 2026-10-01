# GEO-INFER-BAYES/src/geo_infer_bayes/utils

Utils workspace within `GEO-INFER-BAYES`.

## Contents

- `__init__.py`
- `data_processing.py`
- `diagnostics.py`
- `likelihoods.py`
- `priors.py`
- `rng.py`
- `visualization.py`

## Public Interface

- `data_processing.py:prepare_spatial_data` (function)
- `data_processing.py:load_geospatial_data` (function)
- `data_processing.py:validate_spatial_data` (function)
- `data_processing.py:create_spatial_grid` (function)
- `data_processing.py:sample_spatial_data` (function)
- `data_processing.py:save_processed_data` (function)
- `diagnostics.py:mcmc_diagnostics` (function)
- `diagnostics.py:convergence_metrics` (function)
- `likelihoods.py:SpatialLikelihood` (class)
- `likelihoods.py:PoissonProcess` (class)
- `likelihoods.py:GaussianLikelihood` (class)
- `priors.py:SpatialPrior` (class)
- `priors.py:TemporalPrior` (class)
- `priors.py:GaussianProcessPrior` (class)
- `rng.py:resolve_rng` (function)
- `rng.py:resolve_optional_rng` (function)
- `rng.py:spawn_rng` (function)
- `rng.py:derive_int_seed` (function)
- `visualization.py:plot_posterior` (function)
- `visualization.py:plot_spatial_prediction` (function)

## Module Metadata

- Module: `GEO-INFER-BAYES`
- Package: `geo_infer_bayes`
- Version: `0.3.0`
- Install: `uv pip install -e ./GEO-INFER-BAYES`
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
