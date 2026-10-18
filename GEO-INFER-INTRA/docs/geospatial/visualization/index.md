# Geospatial Visualization

This hub documents the visualization contracts that are implemented in the
repository. The runnable visualization surface is currently centered on the
SPACE and PLACE modules, with native H3 boundaries converted to GeoJSON order
(``[longitude, latitude]``) and Folium order (``[latitude, longitude]``) at the
client boundary.

## Current guides

- [H3 visualization techniques](../data_formats/h3/h3_visualization_techniques.md)
- [First map](../../getting_started/first_map.md)
- [Coordinate systems](../concepts/coordinate_systems.md)
- [Spatial data models](../concepts/spatial_data_models.md)
- [Reusable manuscript figures and source data](../../integration/research_artifacts.md)

## Implemented visualization entry points

- `geo_infer_space.core.visualization_engine.VisualizationEngine`
- `geo_infer_space.backends.h3.visualization.H3Visualization`
- `geo_infer_place.core.visualization_engine.InteractiveVisualizationEngine`
- `geo_infer_place.core.unified_backend.CascadianAgriculturalH3Backend`
- `geo_infer_space.core.geolibre_projects` — deterministic `.geolibre.json`
  project emission (GeoLibre format v0.1.0) for H3 grids and GeoJSON layers,
  so results open in the GeoLibre web/desktop/Jupyter viewer.
- `geo_infer_art.core.visualization.geo_art.GeoArt` — composable Matplotlib
  axes, explicit raster layouts/bands, attribute legends and shared scalar
  limits, PNG/SVG export, and optional Plotly/Folium maps.

## Composed figures

Pass an existing Matplotlib `ax` to `GeoArt.apply_style` for subplots and use
the same `vmin`/`vmax` for comparable scalar data. Name a vector `column` for
its legend. For raster data specify `raster_layout="band_first"` or
`"channel_last"` when the shape is ambiguous; `raster_band` selects a zero-based
scalar band. Rasterio-loaded data retains its band-first layout and NoData mask.

`GeoArt.optimize_for_performance` takes a positive integer `(height, width)`
and the same `raster_layout` selection. It resizes spatial axes, retaining the
band axis, original footprint and scaled affine transform. Bilinear display
sampling uses full pixel extents, replicates boundary pixels and weights only
valid masked contributors. An output without valid contributors remains masked.
Integer arrays retain their dtype; masked arrays use floating interpolation.
This resampler does not promise area averages or conservation of cell totals.
Vector simplification uses a finite nonnegative tolerance in CRS units and
preserves the active geometry column. Both operations return a copy.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from geo_infer_art import GeoArt

bands = np.arange(48, dtype=float).reshape(3, 4, 4)
source = GeoArt(bands, metadata={"raster_layout": "band_first"})
with source.optimize_for_performance((2, 2)) as resized:
    assert resized.data.shape == (3, 2, 2)
    np.testing.assert_allclose(resized.data[0], [[2.5, 4.5], [10.5, 12.5]])
    with TemporaryDirectory() as temporary:
        output = Path(temporary) / "resized.png"
        resized.apply_style(raster_band=0).save(str(output))
        assert output.stat().st_size > 100
assert source.data.shape == (3, 4, 4)
```

APP's `AgentVisualization` emits independent strict-JSON snapshots for both map
features and all dashboard widgets. Other native values use their string form;
NaN and Infinity raise `ValueError`. Callers choose how to represent missing
measurements before rendering.

An internally created figure is owned by `GeoArt`; `close()` and its context
manager release it. Caller-provided axes stay caller-owned. `save` can retain
the figure with `close=False` for multiple exports, creates output parents,
and accepts native format-specific metadata and transparency. SVG supplies
vector artwork; a raster image embedded in SVG remains a raster. Closing a
figure does not close unrelated callers' figures.

Folium export requires a declared vector CRS, reprojects to WGS84, retains
holes/multipart geometry through GeoJSON, and escapes display attributes.
Plotly scatter maps require points. Optional visualization dependencies retain
their declared package floors. These classes and their public methods are
discoverable through `geo-infer catalog ART` and callable through the shared
[script, CLI and MCP interfaces](../../integration/agent_interfaces.md).

INTRA's documentation preview uses a local spherical orthographic view to
display H3 cells across the antimeridian and near poles. Its PNG metadata names
the display projection. It is an illustrative preview, not an area/distance
measurement or a replacement for an analysis CRS.

Generated HTML, PNG, and dashboard artifacts belong in a caller-provided
output directory. The repository's validation suites check finite metrics,
closed GeoJSON rings, H3 cell metadata, and output isolation; they do not claim
that a historical image in `GEO-INFER-SPACE/reports/visualizations/status/` is
a current assessment.

## Coordinate-order contract

H3's `cell_to_boundary` returns native `(latitude, longitude)` pairs. Convert
explicitly at the output boundary:

```python
import h3

cell = h3.latlng_to_cell(37.7749, -122.4194, 9)
ring = [[lng, lat] for lat, lng in h3.cell_to_boundary(cell)]
ring.append(ring[0])
assert ring[0] == ring[-1] and len(ring) >= 7
```

Use `[lat, lng]` only when passing coordinates to Folium. Never pass the
removed H3 v3 `geo_json=` argument to `cell_to_boundary`.
