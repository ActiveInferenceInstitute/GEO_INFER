# GEO-INFER-ECON/src/geo_infer_econ/integrations

Integrations workspace within `GEO-INFER-ECON`.

## Contents

- `__init__.py`
- `data_integration.py`
- `logistics_integration.py`
- `space_integration.py`
- `time_integration.py`

## Public Interface

- `data_integration.py:DataIntegration` (class)
- `logistics_integration.py:LogisticsEconomicAnalyzer` (class)
- `space_integration.py:SpaceIntegration` (class)
- `time_integration.py:TimeIntegration` (class)

## Module Metadata

- Module: `GEO-INFER-ECON`
- Package: `geo_infer_econ`
- Version: `0.3.0`
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
