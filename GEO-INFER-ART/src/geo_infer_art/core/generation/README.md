# GEO-INFER-ART/src/geo_infer_art/core/generation

Generation workspace within `GEO-INFER-ART`.

## Contents

- `__init__.py`
- `custom_algorithms.py`
- `generative_map.py`
- `performance_optimizer.py`
- `procedural_art.py`

## Public Interface

- `custom_algorithms.py:CustomAlgorithmFramework` (class)
- `custom_algorithms.py:example_spiral_algorithm` (function)
- `custom_algorithms.py:example_cellular_growth_algorithm` (function)
- `custom_algorithms.py:example_fractal_landscape_algorithm` (function)
- `generative_map.py:GenerativeMap` (class)
- `performance_optimizer.py:PerformanceOptimizer` (class)
- `performance_optimizer.py:cache_result` (function)
- `performance_optimizer.py:parallel_map` (function)
- `performance_optimizer.py:time_execution` (function)
- `procedural_art.py:ProceduralArt` (class)

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
