# GEO-INFER-BAYES/src/geo_infer_bayes/models

Models workspace within `GEO-INFER-BAYES`.

## Contents

- `__init__.py`
- `_model_utils.py`
- `base.py`
- `bayesian_network.py`
- `bayesian_timeseries.py`
- `dirichlet_process.py`
- `dynamic_spatial.py`
- `gaussian_process.py`
- `hierarchical.py`
- `multilevel.py`
- `spatial_causal.py`
- `spatial_clustering.py`
- `spatial_gp.py`
- `spatiotemporal_gp.py`

## Public Interface

- `base.py:BayesianModel` (class)
- `bayesian_network.py:BayesianNetwork` (class)
- `bayesian_timeseries.py:BayesianTimeSeriesModel` (class)
- `dirichlet_process.py:DirichletProcessMixture` (class)
- `dynamic_spatial.py:DynamicSpatialModel` (class)
- `gaussian_process.py:SpatialCovariance` (class)
- `gaussian_process.py:GaussianProcess` (class)
- `hierarchical.py:HierarchicalBayesianModel` (class)
- `multilevel.py:MultilevelModel` (class)
- `spatial_causal.py:SpatialCausalModel` (class)
- `spatial_clustering.py:SpatialClusteringModel` (class)
- `spatial_gp.py:SpatialGP` (class)
- `spatial_gp.py:SparseSpatialGP` (class)
- `spatiotemporal_gp.py:SpatioTemporalConfig` (class)
- `spatiotemporal_gp.py:SpatioTemporalGP` (class)
- `spatiotemporal_gp.py:create_spatiotemporal_gp` (function)

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
