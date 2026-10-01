# GEO-INFER-ART/src/geo_infer_art/core/place

Place workspace within `GEO-INFER-ART`.

## Contents

- `__init__.py`
- `cultural_map.py`
- `place_art.py`

## Public Interface

- `cultural_map.py:CulturalMap` (class)
- `place_art.py:PlaceArt` (class)

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
