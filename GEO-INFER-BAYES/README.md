# GEO-INFER-BAYES

Comprehensive Bayesian inference framework with probabilistic modeling, uncertainty quantification, and computational methods for geospatial data.

## Contents

- `config/`
- `docs/`
- `examples/`
- `src/`
- `tests/`
- `SKILL.md`
- `mcmc_traces.png`
- `mean_prediction.png`
- `posterior_distributions.png`
- `pyproject.toml`
- `spatial_data.png`
- `uncertainty.png`

## Public Interface

- No public Python symbols are defined directly in this directory.

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


## Visualization Contracts

- Spatial prediction, uncertainty, posterior, and model-comparison plots
  validate finite aligned numeric inputs and confidence levels.
- Spatial prediction handles the single-panel case when uncertainty is omitted
  and returns a valid figure for both single- and multi-panel layouts.

## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
