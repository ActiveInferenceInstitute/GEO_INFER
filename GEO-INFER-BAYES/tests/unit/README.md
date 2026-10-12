# GEO-INFER-BAYES/tests/unit

Unit workspace within `GEO-INFER-BAYES`.

## Contents

- `test_abc_smc.py`
- `test_api_interfaces.py`
- `test_base_model.py`
- `test_civic_intel.py`
- `test_data_processing.py`
- `test_diagnostics.py`
- `test_distributional_uncertainty.py`
- `test_evaluation_metrics.py`
- `test_gaussian_process.py`
- `test_geo_observations.py`
- `test_hmc.py`
- `test_inference.py`
- `test_likelihoods.py`
- `test_mcmc.py`
- `test_model_comparison.py`
- `test_model_contracts.py`
- `test_posterior.py`
- `test_posterior_prediction.py`
- `test_priors.py`
- `test_psis_loo_contract.py`
- `test_reproducibility.py`
- `test_rng.py`
- `test_sparse_spatial_gp.py`
- `test_spatial_gp.py`
- `test_spatiotemporal_gp.py`
- `test_variational.py`
- `test_visualization_utils.py`

## Public Interface

- No public Python symbols are defined directly in this directory.

## Module Metadata

- Module: `GEO-INFER-BAYES`
- Package: `geo_infer_bayes`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-bayes`
- Tests: `uv run python -m pytest GEO-INFER-BAYES/tests/unit`

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
uv run python -m pytest GEO-INFER-BAYES/tests/unit
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
