# GEO-INFER-ART/src/geo_infer_art

Geo Infer Art workspace within `GEO-INFER-ART`.

## Contents

- `core/`
- `utils/`
- `__init__.py`
- `cli.py`

## Public Interface

- `cli.py:ensure_directory` (function)
- `cli.py:process_geo_art` (function)
- `cli.py:process_style_transfer` (function)
- `cli.py:process_place_art` (function)
- `cli.py:process_generative_map` (function)
- `cli.py:process_procedural_art` (function)
- `cli.py:process_cultural_map` (function)
- `cli.py:process_map_style` (function)
- `cli.py:process_animation` (function)
- `cli.py:process_custom_algorithm` (function)
- `cli.py:process_performance` (function)
- `cli.py:process_3d_viz` (function)
- `cli.py:process_realtime` (function)
- `cli.py:process_web_map` (function)
- `cli.py:main` (function)

## Module Metadata

- Module: `GEO-INFER-ART`
- Package: `geo_infer_art`
- Version: `0.3.0`
- Install: `uv sync --package geo-infer-art`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module ART`

## Dependencies

- `geopandas>=0.13.0`
- `matplotlib>=3.4.0`
- `numpy>=1.21.0`
- `pillow>=8.3.0`
- `rasterio>=1.2.0`
- `scipy>=1.7.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module ART
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
