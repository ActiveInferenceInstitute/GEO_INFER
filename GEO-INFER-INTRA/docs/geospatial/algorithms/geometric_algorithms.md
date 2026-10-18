# Geometric Algorithms

Geometric operations answer questions about shape, containment, intersection,
and proximity. GEO-INFER-SPACE exposes backend-dispatched operations through
`geo_infer_space.core.geometric_operations.GeometricOperationsInterface`.
Shapely is also a declared SPACE dependency and can be used directly when a
workflow requires its specific geometry semantics.

## Coordinates, units, and boundaries

Shapely computes in a Cartesian plane. Its areas and distances use the input
coordinate units; longitude/latitude degrees do not become metres merely by
calling `area`, `distance`, or `buffer`. Transform to a suitable projected CRS
for local metric analysis, or select an explicitly geodesic implementation.
Shapely itself does not perform CRS transformation. See the
[Shapely manual](https://shapely.readthedocs.io/en/2.1.2/manual.html).

Specify whether a boundary counts as contained. In the example below,
`contains` excludes a point on the polygon boundary while `covers` includes it.
Do not use one predicate as an undocumented substitute for the other.

## Executable planar geometry example

The coordinates below are synthetic Cartesian coordinates in a common local
unit system, not WGS84 observations:

```python
from shapely.geometry import LineString, Point, Polygon

polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
assert polygon.is_valid
assert polygon.area == 100.0
assert polygon.centroid.equals(Point(5, 5))
assert polygon.contains(Point(5, 5))
assert not polygon.contains(Point(0, 5))
assert polygon.covers(Point(0, 5))

line = LineString([(-1, 5), (11, 5)])
clipped = line.intersection(polygon)
assert clipped.equals(LineString([(0, 5), (10, 5)]))
assert line.buffer(0.5).area > 0
```

For production inputs, check validity, empty results, geometry type, CRS,
coordinate units, and the handling of holes and multipart geometries. Do not
silently repair invalid geometry when repair could change the scientific
meaning of the input.

## Spatial-index candidate selection

This complete example uses Shapely 2's index results as array positions and
requests an exact predicate:

```python
from shapely.geometry import LineString, Polygon
from shapely.strtree import STRtree

polygons = [
    Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
    Polygon([(20, 20), (21, 20), (21, 21), (20, 21)]),
]
query = LineString([(-1, 5), (11, 5)])
index = STRtree(polygons)
positions = index.query(query, predicate="intersects")
assert positions.tolist() == [0]
intersections = [query.intersection(polygons[int(i)]) for i in positions]
assert len(intersections) == 1
```

A bare `query` filters by envelopes; the explicit `intersects` predicate checks
the geometries. `STRtree` indexes a fixed collection rather than exposing a
mutable insert API. Rebuild it if that collection changes.

## Algorithm concepts and limits

| Operation | Important distinction |
| --- | --- |
| Point in polygon | Boundary and hole policy are part of the result contract |
| Line intersection | Collinear overlap can return a line, not one point |
| Convex hull | Degenerate inputs may yield a line or point |
| Simplification | Tolerance uses coordinate units; preserving topology does not preserve every measurement |
| Buffer | A planar offset is not automatically a geodesic service area |
| Area and centroid | Ring orientation, holes, and coordinate system affect interpretation |
| Voronoi / Delaunay | Degeneracy, duplicate points, and clipping need explicit handling |

For a single simple planar ring, the shoelace area uses the signed sum
`A = sum(x_i*y_(i+1) - x_(i+1)*y_i) / 2`. Centroid formulas must use that signed
area consistently; taking its absolute value before applying the centroid
formula reverses the sign for clockwise input. A zero-area ring has no area
centroid. Prefer the geometry library for holes and multipart polygons.

Complexity depends on the actual algorithm, input, numerical robustness, and
output size. This page does not assign universal runtime bounds to a backend's
buffer or overlay operation, nor claim a benchmark speedup.

## Use the owning public interface

```bash
uv run --frozen geo-infer describe geo_infer_space.core.geometric_operations:GeometricOperationsInterface
uv run --frozen geo-infer catalog SPACE --query intersection --limit 20
uv run --frozen python GEO-INFER-TEST/run_unified_tests.py --module SPACE
```

Inspect the selected backend for coordinate order, units, supported operations,
and availability. The previous `geo_infer.spatial.geometry`,
`geo_infer.spatial.algorithms`, and `geo_infer.parallel` imports were not shipped
interfaces. Use actual module targets and the
[agent execution contract](../../integration/agent_interfaces.md) for scripts,
CLI, and MCP.

- [Spatial indexing](spatial_indexing.md)
- [SPACE module guide](../../modules/geo-infer-space.md)
- [API reference policy](../../api/reference.md)
