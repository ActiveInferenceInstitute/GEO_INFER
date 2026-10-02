# GEO-INFER-MATH/src/geo_infer_math/api

Api workspace within `GEO-INFER-MATH`.

## Contents

- `convenience/`
- `__init__.py`
- `spatial_analysis.py`

## Public Interface

- `spatial_analysis.py:DescriptiveStatsRequest` (class)
- `spatial_analysis.py:DescriptiveStatsResponse` (class)
- `spatial_analysis.py:AutocorrelationRequest` (class)
- `spatial_analysis.py:AutocorrelationResponse` (class)
- `spatial_analysis.py:HotspotAnalysisRequest` (class)
- `spatial_analysis.py:HotspotAnalysisResponse` (class)
- `spatial_analysis.py:ClusteringRequest` (class)
- `spatial_analysis.py:ClusteringResponse` (class)
- `spatial_analysis.py:InterpolationRequest` (class)
- `spatial_analysis.py:InterpolationResponse` (class)
- `spatial_analysis.py:SpatialDataset` (class)
- `spatial_analysis.py:SpatialAnalysisAPI` (class)

## Module Metadata

- Module: `GEO-INFER-MATH`
- Package: `geo_infer_math`
- Version: `0.4.0`
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
