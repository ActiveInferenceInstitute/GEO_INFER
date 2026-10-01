---
name: geo-infer-art
description: Generative geospatial art and cartographic visualization. Use when creating artistic map visualizations, procedural spatial art, animation sequences, color palettes, style transfer, or aesthetically-focused geographic rendering.
difficulty: beginner
estimated_time: 30min
examples_dir: ../GEO-INFER-EXAMPLES/examples/
---

# GEO-INFER-ART

## Instructions

### Core Capabilities

- **Generative art**: 23 procedural algorithms (noise fields, fractals, cellular automata, L-systems, boids, DLA, WFC, ...) via `ProceduralArt`
- **Cartographic design**: Aesthetic map styling via `GeoArt.apply_style` / `MapStyle`, color palettes via `ColorPalette`
- **Interactive display**: mplcursors-based interactive map exploration (`GeoArt.add_interactive_elements`)
- **Animation**: GIF/video export via `GeoArt.create_animation` and `GenerativeMap.create_animation`
- **Style transfer**: Neural style transfer on geospatial data via `StyleTransfer` (requires the `neural` extra: `uv sync --extra neural` or install `tensorflow`)
- **Place-based art**: `PlaceArt` and `CulturalMap` generate art from coordinates or place names

### Key Imports

```python
# Package-level exports (all 10 public classes)
from geo_infer_art import (
    GeoArt,
    MapStyle,
    ColorPalette,
    StyleTransfer,
    GenerativeMap,
    ProceduralArt,
    PlaceArt,
    CulturalMap,
    CustomAlgorithmFramework,
    PerformanceOptimizer,
)

# Or by subpackage
from geo_infer_art.core.visualization import GeoArt, MapStyle
from geo_infer_art.core.aesthetics import StyleTransfer, ColorPalette
from geo_infer_art.core.generation import (
    GenerativeMap,
    ProceduralArt,
    CustomAlgorithmFramework,
    PerformanceOptimizer,
)
from geo_infer_art.core.place import PlaceArt, CulturalMap
```

## Examples

```python
from geo_infer_art import ProceduralArt, GeoArt

# Procedural art seeded by geographic coordinates (works offline)
art = ProceduralArt.from_geo_coordinates(
    lat=40.7128, lon=-74.0060, algorithm="noise_field"
)
art.save("nyc_noise.png")
art.show()

# Styled map from a GeoJSON file
geo_art = GeoArt.load_geojson("data.geojson")
geo_art.apply_style(style="default", color_palette="ocean", legend=True)
geo_art.save("styled_map.png", dpi=300)

# Procedural art with explicit algorithm selection
art = ProceduralArt(
    algorithm="koch_snowflake",
    params={"iterations": 3, "color_palette": "ocean", "seed": 42},
    resolution=(800, 800),
)
art.generate()
```

Randomness is per instance: `ProceduralArt` seeds a `numpy.random.Generator`
from `params["seed"]`; `GenerativeMap`, `GenerativeMap.from_elevation`,
`PlaceArt` and `StyleTransfer` take a `seed` (int or `Generator`). Saved
custom-algorithm files must carry `function_name` (as written by
`CustomAlgorithmFramework.save_algorithms_to_file`); entries without it are
rejected with `ValueError` before any source is executed.

Neural style transfer (requires `tensorflow`, packaged as the `neural` extra;
raises a clear `ImportError` when absent):

```python
from geo_infer_art import StyleTransfer

styled = StyleTransfer.apply(geo_data=geo_art.data, style="watercolor")
styled.save("stylized.png")
```

## Guidelines

- `ProceduralArt.ALGORITHMS` lists all 23 supported algorithm names; `generate()`
  dispatches to `_generate_<name>` for every entry (unknown names raise `ValueError`).
- Heavy backends are optional: `plotly`/`folium`/`psutil` (`integrations` extra),
  `mayavi` (`viz3d` extra), `tensorflow` (`neural` extra). Guarded imports raise
  informative errors when missing.

### Integrations

- No required cross-module dependencies; ART is a leaf module that consumes plain
  GeoDataFrames / numpy arrays.
- Test: `uv run --no-sync python -m pytest tests/ -v`
