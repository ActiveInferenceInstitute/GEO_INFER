# GEO-INFER-PLACE/locations/cascadia/src/core/visualization

Visualization workspace within `GEO-INFER-PLACE`.

## Contents

- `__init__.py`
- `comprehensive_visualization.py`
- `datashader_visualization.py`
- `deepscatter_visualization.py`
- `interactive_h3_visualization.py`
- `static_visualization.py`

## Public Interface

- `comprehensive_visualization.py:ComprehensiveVisualizationEngine` (class)
- `comprehensive_visualization.py:create_comprehensive_visualization_engine` (function)
- `datashader_visualization.py:CascadiaDatashaderVisualizer` (class)
- `datashader_visualization.py:create_datashader_visualization` (function)
- `deepscatter_visualization.py:CascadiaDeepscatterVisualizer` (class)
- `deepscatter_visualization.py:create_deepscatter_visualization` (function)
- `interactive_h3_visualization.py:InteractiveH3Visualization` (class)
- `interactive_h3_visualization.py:create_interactive_h3_visualization` (function)
- `static_visualization.py:create_static_plots` (function)
- `static_visualization.py:create_summary_statistics` (function)
- `static_visualization.py:create_data_export` (function)

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
