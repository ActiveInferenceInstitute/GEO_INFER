# GEO-INFER-PLACE/src/geo_infer_place/locations/del_norte_county/dashboard

Dashboard workspace within `GEO-INFER-PLACE`.

## Contents

- `__init__.py`
- `analyzers.py`
- `config.py`
- `core.py`

## Public Interface

- `analyzers.py:ClimateAnalyzer` (class)
- `analyzers.py:ZoningAnalyzer` (class)
- `analyzers.py:AgroEconomicAnalyzer` (class)
- `config.py:LayerConfig` (class)
- `core.py:AdvancedDashboard` (class)

## Module Metadata

- Module: `GEO-INFER-PLACE`
- Package: `geo_infer_place`
- Version: `0.3.0`
- Install: `uv sync --package geo-infer-place`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module PLACE`

## Dependencies

- `urllib3>=2.0.6`
- `geopandas>=0.13.0`
- `networkx>=2.6.0`
- `shapely>=2.0.0`
- `h3>=4.5.0,<5`
- `numpy>=1.20.0`
- `pandas>=1.3.0`
- `pyyaml>=6.0`
- `folium>=0.14.0`
- `plotly>=5.0.0`
- `matplotlib>=3.5.0`
- `branca>=0.6.0`
- `requests>=2.28.0`
- `geo-infer-space`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module PLACE
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
