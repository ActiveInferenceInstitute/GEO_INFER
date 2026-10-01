# GEO-INFER-ART User Guide

## Getting Started

### Installation

GEO-INFER-ART is a member of the root uv workspace. From the repository root:

```bash
uv sync --package geo-infer-art
# Optional extras: neural (TensorFlow style transfer), integrations
# (folium/plotly/psutil), viz3d (mayavi)
uv sync --package geo-infer-art --extra neural
```

### Reproducibility

Stochastic generators draw from a per-instance ``numpy.random.Generator``:
``ProceduralArt`` seeds it from ``params["seed"]``, and ``GenerativeMap``,
``PlaceArt`` and ``StyleTransfer`` accept a ``seed`` argument (an int or a
``Generator``). The process-wide ``numpy.random`` state is never used.

### Quick Start

```python
from geo_infer_art import GeoArt

# Load your data
geo_art = GeoArt.load_geojson("my_data.geojson")

# Create beautiful map
geo_art.apply_style(style="watercolor", title="My Map")

# Save
geo_art.save("my_map.png", dpi=300)
```

## Creating Maps

### Basic Map

```python
from geo_infer_art import GeoArt

geo_art = GeoArt.load_geojson("city_boundaries.geojson")
map_img = geo_art.apply_style(
    style="minimal",
    color_palette="monochrome",
    title="City Boundaries",
)
```

### Multi-Layer Map

```python
# GeoArt methods chain; layering is composition, not an add_layer API
geo_art = (
    GeoArt.load_geojson("city_boundaries.geojson")
    .apply_style(style="minimal", title="Base layer")
    .set_projection("plate_carree")
    .add_annotations([{"text": "Downtown", "coords": (-74.0, 40.7)}])
)
geo_art.save("layered_map.png")
```

## Styling Options

| Style | Best For |
|-------|----------|
| `watercolor` | Artistic, decorative |
| `minimal` | Clean infographics |
| `blueprint` | Technical linework |

## 3D Visualization

```python
from geo_infer_art import GeoArt

terrain = GeoArt.load_raster("dem.tif")
# Returns a GeoArt3D handle; requires mayavi or plotly installed
scene = terrain.create_3d_visualization()
```

## Creating Animations

```python
geo_art = GeoArt.load_geojson("temporal_data.geojson")
animation_path = geo_art.create_animation(
    output_path="timelapse.mp4",
    style_sequence=["minimal", "watercolor"],
    duration=5.0,
    fps=30,
)
```

## Generative Art

```python
from geo_infer_art import GenerativeMap

artwork = GenerativeMap.from_elevation(
    region="city_grid",
    style="flow",
    abstraction_level=0.7,
)
artwork.save("city_art.png")
```


