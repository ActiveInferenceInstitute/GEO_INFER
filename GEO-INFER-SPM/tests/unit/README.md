# GEO-INFER-SPM/tests/unit

Unit workspace within `GEO-INFER-SPM`.

## Contents

- `test_acceptance_spm.py`
- `test_advanced_models.py`
- `test_api_endpoints.py`
- `test_bayesian.py`
- `test_compiler_profile.py`
- `test_contrasts.py`
- `test_data_io.py`
- `test_glm.py`
- `test_helpers.py`
- `test_helpers_reproducibility.py`
- `test_package_exports.py`
- `test_preprocessing.py`
- `test_rft.py`
- `test_rft_fwe_contract.py`
- `test_spatial_analysis.py`
- `test_temporal_analysis.py`
- `test_validation.py`
- `test_visualization.py`

## Public Interface

- No public Python symbols are defined directly in this directory.

## Module Metadata

- Module: `GEO-INFER-SPM`
- Package: `geo_infer_spm`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-spm`
- Tests: `uv run python -m pytest GEO-INFER-SPM/tests/unit`

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
uv run python -m pytest GEO-INFER-SPM/tests/unit
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
