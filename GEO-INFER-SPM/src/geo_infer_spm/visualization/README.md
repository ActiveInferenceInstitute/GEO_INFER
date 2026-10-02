# GEO-INFER-SPM/src/geo_infer_spm/visualization

Visualization workspace within `GEO-INFER-SPM`.

## Contents

- `__init__.py`
- `diagnostics.py`
- `interactive.py`
- `maps.py`

## Public Interface

- `diagnostics.py:plot_model_diagnostics` (function)
- `diagnostics.py:plot_contrast_results` (function)
- `interactive.py:create_interactive_map` (function)
- `interactive.py:create_dashboard` (function)
- `interactive.py:create_time_series_explorer` (function)
- `maps.py:create_statistical_map` (function)
- `maps.py:plot_spm_results` (function)

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
