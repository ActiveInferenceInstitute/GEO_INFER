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

### Agent interfaces

Use `uv run --frozen geo-infer catalog ART` to discover public source
APIs, then `geo-infer describe` before calling a selected target. Scripts use
`geo_infer_api.agent_runtime.AgentSession`; the optional stdio MCP server is
`geo-infer-mcp`. See [requests, native objects and composition](../GEO-INFER-INTRA/docs/integration/agent_interfaces.md).
The adapter retains this module's dependencies and authorization requirements.

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

Custom algorithm persistence accepts schema version 1 registry references only.
Supply trusted callables to `CustomAlgorithmFramework(algorithm_registry=...)`;
legacy source-bearing JSON is rejected before any state change. Direct callable
registration remains available, but only registry-listed functions can be saved.

### Compose and export maps

`GeoArt.apply_style` accepts caller-owned `ax`, a vector `column`, and shared
scalar `vmin`/`vmax` limits for comparable panels. A legend requires its color
attribute; specify `column` explicitly. Raster inputs support 2D scalar arrays,
band-first `(bands, height, width)` and channel-last `(height, width, channels)`.
Use `raster_layout` for ambiguous shapes and zero-based `raster_band` for a
selected scalar band. Loaded rasters retain band-first layout and NoData masks.

Use a `with GeoArt(...) as art` context or `close()` to release owned figures.
Caller-provided axes remain caller-owned. `save` creates nested directories and
accepts `close`, `transparent` and native Matplotlib `metadata`; retain a figure
with `close=False` when exporting several formats. Choose `.svg` for vector
artwork and `.png` for raster output. Folium exports reproject declared vector
CRS to WGS84 and retain geometry holes/multipart features; Plotly scatter maps
require point geometry.

Discover the current signatures with `geo-infer describe
geo_infer_art.core.visualization.geo_art:GeoArt.apply_style`. Use a retained
`GeoArt` handle in one script, persistent CLI session or MCP session to compose
`apply_style`, `save` and `close`; the shared interfaces preserve native arrays
and GeoDataFrames. See [visualization contracts](../GEO-INFER-INTRA/docs/geospatial/visualization/index.md).
