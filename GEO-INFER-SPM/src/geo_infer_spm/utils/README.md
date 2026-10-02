# GEO-INFER-SPM/src/geo_infer_spm/utils

Utils workspace within `GEO-INFER-SPM`.

## Contents

- `__init__.py`
- `data_io.py`
- `helpers.py`
- `preprocessing.py`
- `rng.py`
- `validation.py`

## Public Interface

- `data_io.py:load_data` (function)
- `data_io.py:load_geotiff` (function)
- `data_io.py:load_netcdf` (function)
- `data_io.py:load_geojson` (function)
- `data_io.py:load_geopackage` (function)
- `data_io.py:load_csv_with_coords` (function)
- `data_io.py:load_hdf5` (function)
- `data_io.py:load_json_data` (function)
- `data_io.py:save_spm` (function)
- `helpers.py:create_design_matrix` (function)
- `helpers.py:generate_coordinates` (function)
- `helpers.py:generate_synthetic_data` (function)
- `helpers.py:create_spatial_basis_functions` (function)
- `helpers.py:compute_power_analysis` (function)
- `preprocessing.py:preprocess_data` (function)
- `preprocessing.py:handle_missing_data` (function)
- `preprocessing.py:normalize_data` (function)
- `preprocessing.py:remove_outliers` (function)
- `preprocessing.py:spatial_filter` (function)
- `preprocessing.py:temporal_filter` (function)

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
