# GEO-INFER-PLACE/locations/cascadia

Cascadia workspace within `GEO-INFER-PLACE`.

## Contents

- `config/`
- `data/`
- `docs/`
- `generated/`
- `output/`
- `src/`
- `tests/`
- `cascadia_main.py`
- `cascadia_server.py`
- `cleanup_data.py`
- `.gitignore`
- `.python-version`
- `DATA_STRUCTURE.md`
- `pyproject.toml`
- `requirements.txt`
- `run_analysis.sh`
- `uv.lock`

## Public Interface

- `cascadia_main.py:parse_counties` (function)
- `cascadia_main.py:initialize_analysis` (function)
- `cascadia_main.py:initialize_modules_with_enhanced_data_management` (function)
- `cascadia_main.py:generate_reports` (function)
- `cascadia_main.py:parse_arguments` (function)
- `cascadia_main.py:main` (function)
- `cascadia_main.py:run_comprehensive_analysis_with_enhanced_data` (function)
- `cascadia_main.py:calculate_enhanced_redevelopment_score` (function)
- `cascadia_main.py:export_results_with_visualizations` (function)
- `cascadia_main.py:print_analysis_summary` (function)
- `cascadia_server.py:create_app` (function)
- `cascadia_server.py:run_stdlib_server` (function)
- `cascadia_server.py:parse_args` (function)
- `cascadia_server.py:main` (function)
- `cleanup_data.py:setup_logging` (function)
- `cleanup_data.py:cleanup_old_logs` (function)
- `cleanup_data.py:cleanup_pycache` (function)
- `cleanup_data.py:main` (function)

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
