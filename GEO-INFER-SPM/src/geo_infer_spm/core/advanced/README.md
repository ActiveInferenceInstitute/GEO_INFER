# GEO-INFER-SPM/src/geo_infer_spm/core/advanced

Advanced workspace within `GEO-INFER-SPM`.

## Contents

- `__init__.py`
- `mixed_effects.py`
- `model_validation.py`
- `nonparametric.py`
- `spatial_regression.py`

## Public Interface

- `mixed_effects.py:MixedEffectsSPM` (class)
- `mixed_effects.py:fit_mixed_effects` (function)
- `model_validation.py:ModelValidator` (class)
- `model_validation.py:validate_spm_model` (function)
- `nonparametric.py:NonparametricSPM` (class)
- `nonparametric.py:fit_nonparametric` (function)
- `spatial_regression.py:SpatialRegression` (class)
- `spatial_regression.py:fit_spatial_model` (function)

## Module Metadata

- Module: `GEO-INFER-SPM`
- Package: `geo_infer_spm`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-spm`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module SPM`

## Dependencies

- `geopandas>=0.13.0`
- `h5py>=3.6.0`
- `matplotlib>=3.5.0`
- `numpy>=1.20.0`
- `pandas>=1.3.0`
- `plotly>=5.0.0`
- `rasterio>=1.2.0`
- `scikit-learn>=1.0.0`
- `scipy>=1.7.0`
- `xarray>=0.20.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module SPM
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
