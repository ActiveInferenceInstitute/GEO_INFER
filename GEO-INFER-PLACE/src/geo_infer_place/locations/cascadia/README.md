# GEO-INFER-PLACE/src/geo_infer_place/locations/cascadia

Cascadia workspace within `GEO-INFER-PLACE`.

## Contents

- `config/`
- `core/`
- `data_modules/`
- `__init__.py`
- `cli.py`
- `server.py`

## Public Interface

- `cli.py:parse_counties` (function)
- `cli.py:initialize_analysis` (function)
- `cli.py:initialize_modules_with_enhanced_data_management` (function)
- `cli.py:generate_reports` (function)
- `cli.py:parse_arguments` (function)
- `cli.py:main` (function)
- `cli.py:run_comprehensive_analysis_with_enhanced_data` (function)
- `cli.py:calculate_enhanced_redevelopment_score` (function)
- `cli.py:export_results_with_visualizations` (function)
- `cli.py:print_analysis_summary` (function)
- `server.py:create_app` (function)
- `server.py:run_stdlib_server` (function)
- `server.py:parse_args` (function)
- `server.py:main` (function)

## Module Metadata

- Module: `GEO-INFER-PLACE`
- Package: `geo_infer_place`
- Version: `0.4.0`
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
- `geo-infer-space>=0.4.0`
- `geo-infer-time>=0.4.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module PLACE
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
