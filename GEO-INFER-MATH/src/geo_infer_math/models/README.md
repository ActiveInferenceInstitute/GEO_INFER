# GEO-INFER-MATH/src/geo_infer_math/models

Models workspace within `GEO-INFER-MATH`.

## Contents

- `__init__.py`
- `clustering.py`
- `regression.py`

## Public Interface

- `clustering.py:ClusteringResults` (class)
- `clustering.py:SpatialKMeans` (class)
- `clustering.py:SpatiallyConstrainedKMeans` (class)
- `clustering.py:SpatialDBSCAN` (class)
- `clustering.py:SKATERClustering` (class)
- `clustering.py:HierarchicalClustering` (class)
- `clustering.py:spatial_clustering_analysis` (function)
- `regression.py:RegressionResults` (class)
- `regression.py:OrdinaryLeastSquares` (class)
- `regression.py:SpatialLagModel` (class)
- `regression.py:GeographicallyWeightedRegression` (class)
- `regression.py:SpatialErrorModel` (class)
- `regression.py:SpatialDurbinModel` (class)
- `regression.py:spatial_regression_analysis` (function)

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
