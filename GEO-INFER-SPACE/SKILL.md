---
name: geo-infer-space
description: H3 hexagonal spatial indexing and multi-backend spatial operations. Use when working with H3 cells, spatial indexing, coordinate systems, raster/vector operations, or any spatial backend dispatch (H3, SRAI).
prerequisites:
  required:
    - geo-infer-math
  recommended:
    - geo-infer-data
difficulty: intermediate
estimated_time: 45min
examples_dir: ../GEO-INFER-EXAMPLES/examples/
---

# GEO-INFER-SPACE

## Instructions

### Core Capabilities

- **H3 v4.5 indexing**: Cell operations, hierarchical resolution, grid-disk neighborhoods
- **Nested H3 hierarchies**: Parent/child closure, same-resolution adjacency,
  and deterministic aggregation for ordered H3 resolutions
- **Backend dispatch**: Interface pattern for H3 and SRAI backends
- **Coordinate systems**: CRS transformations, EPSG management
- **Spatial operations**: Buffers, intersections, unions, containment
- **Visualization**: Choropleth maps, heatmaps, spatial dashboards

### Key Imports

```python
from geo_infer_space.backends.h3 import H3Backend
from geo_infer_space import GISManager, H3StateSpace, align_h3_observations
from geo_infer_space.core import (
    GeometricOperationsInterface,
    SpatialIndexingInterface,
)
from geo_infer_space.nested import NestedH3Grid
```

### H3 v4.5 API (Critical)

```python
import h3
# CORRECT (v4):
cell = h3.latlng_to_cell(lat, lng, resolution)
lat, lng = h3.cell_to_latlng(cell)
neighbors = h3.grid_disk(cell, k)

# Pre-v4 coordinate and neighborhood spellings are unsupported. The repository
# H3 contract validator rejects them in runtime source.
```

## Examples

```python
from geo_infer_space.backends.h3 import H3Backend

backend = H3Backend()

# Index a point to an H3 cell at resolution 7
cell = backend.latlng_to_cell(45.5231, -122.6765, resolution=7)
print(f"H3 cell: {cell}")

# Get neighbors within 2 grid rings (excludes the center cell)
neighbors = backend.get_cells_within_radius(cell, k=2)
print(f"Neighbors (k=2): {len(neighbors)} cells")

# Convert a polygon region to H3 cells
from shapely.geometry import box
region = box(-122.8, 45.4, -122.5, 45.6)
cells = backend.polygon_to_cells(region, resolution=8)
print(f"Region coverage: {len(cells)} cells")
```

```python
from geo_infer_space.nested import NestedH3Grid

grid = NestedH3Grid("sf_nested")
hierarchy = grid.build_h3_hierarchy_from_cells(
    ["89283082803ffff"],
    resolutions=[7, 8, 9],
)

assert hierarchy["validation"]["is_valid"]
assert hierarchy["validation"]["orphan_count"] == 0
```

```python
from geo_infer_space import GISManager

gis = GISManager()
x, y = gis.transform_coordinates(
    (-122.6, 45.5),
    from_crs="EPSG:4326",
    to_crs="EPSG:32610",
)
print(f"UTM Zone 10N: ({x:.0f}, {y:.0f})")
```

## Guidelines

- Always use H3 v4 API — zero legacy calls allowed
- Runtime and dependency metadata must use real `h3>=4.5.0,<5`.
- Backend-agnostic: use the dispatcher pattern, not direct H3 calls
- For nested H3, construct hierarchies through `NestedH3Grid`; validate
  `parent_child_map`, `child_parent_map`, `same_level_neighbors`, and
  `validation["orphan_count"] == 0` before handing cells to ACT.
- EPSG:4326 (WGS84) is the default CRS
- Test:
  `uv run pytest GEO-INFER-SPACE/tests/unit/test_nested_h3_contract.py -q`
  and `uv run python GEO-INFER-TEST/run_unified_tests.py --module SPACE`

### Integrations

- **MATH** → Spatial weights for statistics
- **TIME** → Spatio-temporal analysis
- `align_h3_observations` maps finite records into explicit `H3StateSpace`
  columns and ordered UTC timestamps. It rejects unknown cells/times and
  duplicate pairs; absent observations stay NaN.
- SPACE temporal analytics require aware UTC-normalizable timestamps. Weighted
  space-time interpolation is `SpatioTemporalAnalyzer.interpolate_spatiotemporal`
  with softened inverse-distance weights, rather than a kriging estimator.
- Follow [the composition guide](docs/CROSS_MODULE_COMPOSITION.md) for data
  axes, missingness, coordinate order and model schedule requirements.
- **DATA** → Spatial indexing of datasets
- **PLACE** → Boundary tessellation with H3
- Nearly every module depends on SPACE for geographic indexing
