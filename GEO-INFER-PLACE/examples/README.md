# GEO-INFER-PLACE/examples

Examples workspace within `GEO-INFER-PLACE`.

## Contents

- `demo_output/`
- `del_norte_county_demo.py`
- `del_norte_demo.log`

## Public Interface

- `del_norte_county_demo.py:check_dependencies` (function)
- `del_norte_county_demo.py:load_api_keys` (function)
- `del_norte_county_demo.py:demonstrate_data_sources` (function)
- `del_norte_county_demo.py:demonstrate_api_connections` (function)
- `del_norte_county_demo.py:run_comprehensive_demo` (function)
- `del_norte_county_demo.py:demonstrate_h3_spatial_analysis` (function)
- `del_norte_county_demo.py:run_advanced_demo` (function)
- `del_norte_county_demo.py:main` (function)

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
