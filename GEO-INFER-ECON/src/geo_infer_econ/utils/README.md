# GEO-INFER-ECON/src/geo_infer_econ/utils

Utils workspace within `GEO-INFER-ECON`.

## Contents

- `__init__.py`
- `data_loader.py`
- `indicators.py`
- `rng.py`
- `validator.py`
- `visualizer.py`

## Public Interface

- `data_loader.py:DataSourceConfig` (class)
- `data_loader.py:DataValidationResult` (class)
- `data_loader.py:EconomicDataLoader` (class)
- `data_loader.py:example_data_loading` (function)
- `indicators.py:EconomicIndicators` (class)
- `rng.py:resolve_rng` (function)
- `rng.py:resolve_optional_rng` (function)
- `rng.py:spawn_rng` (function)
- `rng.py:derive_int_seed` (function)
- `validator.py:ModelValidator` (class)
- `visualizer.py:ResultsVisualizer` (class)

## Module Metadata

- Module: `GEO-INFER-ECON`
- Package: `geo_infer_econ`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-econ`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module ECON`

## Dependencies

- `numpy>=1.20.0`
- `pandas>=1.3.0`
- `scipy>=1.7.0`
- `geopandas>=0.13.0`
- `shapely>=2.0.0`
- `scikit-learn>=1.0.0`
- `matplotlib>=3.5.0`
- `seaborn>=0.12.0`
- `h3>=4.5.0,<5`
- `requests>=2.28.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module ECON
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
