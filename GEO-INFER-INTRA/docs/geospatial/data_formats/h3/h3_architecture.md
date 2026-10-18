# H3 Technical Architecture

H3 is a hierarchical global grid with integer resolutions 0 through 15. Its
cells are mostly hexagons, with twelve pentagons at each resolution. GEO-INFER
uses the H3 v4 API and WGS84 latitude/longitude inputs.

## Grid construction and hierarchy

The [official architecture overview](https://h3geo.org/docs/core-library/overview/)
describes a grid constructed on an icosahedron and projected onto a sphere,
using a Dymaxion orientation. That construction does not make the resulting
cells an equal-area tiling or make neighboring centers exactly equidistant.
The former fixed 0.33% distortion, 4% area-variation, and ±5% distance claims
were not supported limits and should not be used in an error budget.

A hexagon has seven immediate children in the index hierarchy; a pentagon has
six. Logical parent/child membership is exact, but geographic boundaries at
successive resolutions are not exact nested subdivisions. A child's polygon
may cross its logical parent's polygon boundary. Use the logical hierarchy for
aggregation and an explicit geometry predicate for exact boundary questions.
See [H3 indexing semantics](https://h3geo.org/docs/highlights/indexing/).

## Cell statistics

H3's [official statistics](https://h3geo.org/docs/core-library/restable/) report
varying cell areas; the maximum-to-minimum hexagon area ratio approaches two
at fine resolutions, excluding pentagons. Average edge length and area describe
a resolution, not a guaranteed size for each cell.

Query the installed implementation instead of copying a static approximate table:

```python
import h3

cell = h3.latlng_to_cell(37.7749, -122.4194, 9)
assert h3.get_resolution(cell) == 9
assert len(h3.get_pentagons(9)) == 12
assert h3.get_num_cells(9) == 2 + 120 * 7**9
print(h3.average_hexagon_area(9, unit="km^2"))
print(h3.average_hexagon_edge_length(9, unit="km"))
print(h3.cell_area(cell, unit="km^2"))
parent = h3.cell_to_parent(cell, 8)
assert cell in h3.cell_to_children(parent, 9)
```

Bound expansion before generating children or large covers. A finer resolution
can increase output cardinality substantially even when the input polygon is
small in geographic terms.

## Cell index representation

The underlying cell index is a 64-bit integer. The usual Python string API
represents it as a hexadecimal string. In cell mode the packed fields are:

| Field | Bits |
| --- | --- |
| Reserved high bit | 1 |
| Mode | 4 |
| Reserved mode-dependent field | 3 |
| Resolution | 4 |
| Base cell | 7 |
| Fifteen hierarchy digits | 45 |

Unused hierarchy digits are set to 7. Use H3 inspection and hierarchy functions
rather than hand-editing packed values or relying on numeric proximity for an
exact spatial query. Other index modes have different field interpretations;
see the [official cell index layout](https://h3geo.org/docs/library/index/cell/).

## Distance and topology

A grid distance counts adjacency hops, not metres or travel time. Pentagon
distortion and other limits can make a requested grid-distance operation fail;
preserve that failure rather than inventing a distance. See the
[traversal API](https://h3geo.org/docs/api/traversal/).
A metric query needs an explicit distance model and, where necessary, a final
predicate against the original geometry.

- [Resolution system](h3_resolution_system.md)
- [Spatial indexing in SPACE](../../algorithms/spatial_indexing.md)
- [Mobility analysis](h3_mobility_analysis.md)
