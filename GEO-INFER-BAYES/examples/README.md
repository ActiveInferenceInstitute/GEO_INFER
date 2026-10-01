# GEO-INFER-BAYES/examples

Examples workspace within `GEO-INFER-BAYES`.

## Contents

- `spatial_bayesian_analysis.py`
- `spatial_gp_example.py`

## Public Interface

- `spatial_bayesian_analysis.py:simulate_gp_field` (function)
- `spatial_bayesian_analysis.py:log_likelihood_matrix` (function)
- `spatial_bayesian_analysis.py:main` (function)
- `spatial_gp_example.py:generate_synthetic_data` (function)
- `spatial_gp_example.py:plot_spatial_data` (function)
- `spatial_gp_example.py:main` (function)

## Module Metadata

- Module: `GEO-INFER-BAYES`
- Package: `geo_infer_bayes`
- Version: `0.3.0`
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
