# Spatial Concepts

A spatial vocabulary distinguishes features, geometries, rasters, cells, networks,
and coordinate reference systems. Names such as contains, intersects, touches,
and disjoint express predicates with specific geometric meanings. An ontology
can record those meanings, but this page does not provide a runtime ontology manager.

## Geometry and indexing are different contracts

Shapely/GeoJSON points use [longitude, latitude] in EPSG:4326; H3 indexing takes
(latitude, longitude). The example verifies polygon containment and explicit H3
state location using the respective real interfaces.

```python
import h3
from shapely.geometry import Point, Polygon
from geo_infer_space import H3StateSpace

point = Point(-124.2, 41.75)
region = Polygon([(-124.3, 41.6), (-124.1, 41.6), (-124.1, 41.8), (-124.3, 41.8)])
assert region.contains(point)
assert not region.disjoint(point)
cell = h3.latlng_to_cell(point.y, point.x, 8)
space = H3StateSpace([cell])
assert space.locate(point.y, point.x) == 0
assert h3.cell_area(cell, unit="m^2") > 0
```

## Properties, relationships, and metrics

Attach geometry, CRS, coordinate order, bounds, units, and measurement support to
a feature. A relation derived from geometry should retain the geometry versions
and predicate used to produce it. `contains` excludes a boundary-only point;
`covers` has different boundary semantics. Network connection and physical
adjacency are also distinct relations.

EPSG:4326 coordinates are angular. Euclidean length or area in degrees is not a
metric length or area; choose a suitable projection or geodesic operation. H3 cell
membership does not prove full polygon containment. H3 neighborhood degree comes
from topology, including pentagons, rather than assuming six neighbors everywhere.

An application may serialize its vocabulary in RDF/OWL and use an explicitly
selected reasoner. That does not make its geometric predicates or inferred domain
facts automatically valid. See [ontology modeling](ontology_modeling.md),
[H3](../geospatial/data_formats/h3/index.md), and [SPACE](../modules/geo-infer-space.md).
