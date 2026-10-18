# Spatial Indexing

Spatial indexes reduce the candidate set for a spatial query. Choose an index
for the data and the query: a point-neighborhood query, a polygon-intersection
query, and an update-heavy feature store need different representations.
This page separates those design choices from the APIs actually shipped in
GEO-INFER-SPACE.

## Choosing a representation

| Representation | Useful starting point | Qualification needed |
| --- | --- | --- |
| H3 cells | Geographic aggregation, cell neighborhoods, multi-resolution keys | Resolution, pentagons, boundary behavior, and the difference between grid and metric distance |
| Bounding-box tree | Candidate selection for vector geometries | Exact geometry predicate after envelope filtering, update model, and CRS |
| KD-tree | Nearest neighbors among numeric points | Coordinate system, distance metric, and rebuilding after updates |
| Regular grid or quadtree | Raster blocks or application-specific spatial bins | Extent, resolution, skewed occupancy, and boundary ownership |

This is a selection guide, not a measured performance ranking. The repository
has no benchmark receipt supporting the former fixed timings for one million
points; use a representative workload and the [profiling guide](../../integration/backend_profiling.md).
Do not infer that every structure in this table has a GEO-INFER wrapper.

## Executable H3 example

```python
from geo_infer_space import cell_to_latlng, latlng_to_cell
from geo_infer_space.core.spatial_indexing import SpatialIndexingInterface

cell = latlng_to_cell(37.7749, -122.4194, resolution=9)
index = SpatialIndexingInterface(backend="h3")
neighborhood = index.get_cell_neighbors(cell, k=1)
assert cell not in neighborhood  # This method excludes the origin cell.
assert len(neighborhood) == 6  # This example is away from a pentagon.
latitude, longitude = cell_to_latlng(cell)
assert -90 <= latitude <= 90
assert -180 <= longitude <= 180
print(cell, latitude, longitude)
```

These calls take WGS84 latitude then longitude. GeoJSON coordinate arrays use
longitude then latitude. A returned cell center is a representative point,
not a recovery of the original input coordinate.

H3 contains hexagons and twelve pentagons at each resolution; cells are not
equal-area. Use actual cell areas when normalizing counts, and consult the
[official H3 cell statistics](https://h3geo.org/docs/core-library/restable/).
A grid ring counts adjacency steps. It does not establish a travel-time,
road-distance, or exact geodesic-radius query.

`SpatialIndexingInterface.get_neighbors(position, radius, resolution,
max_cells=...)` converts a metric radius to a grid disk using average edge
length and an allocation bound. Apply a distance predicate to the original
features when an exact metric cutoff is required. Inspect the current
implementation and signature before choosing a large radius or fine
resolution.

## Vector candidate selection

The [geometric algorithms guide](geometric_algorithms.md) contains an executable
Shapely `STRtree` example with an exact intersection predicate. A bounding-box
hit alone is only a candidate: envelopes can overlap while geometries do not.
GEO-INFER does not provide the old `geo_infer.spatial.indexing.RTreeIndex`,
`QuadtreeIndex`, or `H3Index` namespace shown in earlier drafts.

## Agent discovery and validation

```bash
uv run --frozen geo-infer catalog SPACE --query SpatialIndexingInterface --limit 20
uv run --frozen geo-infer describe geo_infer_space.core.spatial_indexing:SpatialIndexingInterface.get_neighbors
uv run --frozen python GEO-INFER-TEST/run_unified_tests.py --module SPACE
```

Catalog discovery does not install a backend. Script, CLI, and MCP calls retain
the dependencies and resource cost of the selected implementation; see the
[shared agent interfaces](../../integration/agent_interfaces.md).

## Related material

- [H3 v4 reference](../data_formats/h3/index.md)
- [Spatial analysis methods](../analysis/index.md)
- [SPACE module guide](../../modules/geo-infer-space.md)
