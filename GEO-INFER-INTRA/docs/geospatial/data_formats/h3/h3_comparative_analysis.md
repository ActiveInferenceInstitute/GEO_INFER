# Comparing Spatial Indexing Systems

Choose an index for a declared query and data contract. This page describes
representation differences; it does not report comparative runtime measurements,
qualified cloud services, or deployment support for every alternative.

## Representation and hierarchy

| Choice | Representation | Boundary consideration |
| --- | --- | --- |
| H3 | Predominantly hexagonal global cells, integer resolutions 0–15 | Logical hierarchy is exact; parent/child geographic boundaries need not be nested |
| S2 | Spherical quadrilateral cells with recursive four-child subdivision | Cell levels and region coverings have S2-specific semantics |
| Geohash | Encoded longitude/latitude bins | Prefix locality has boundary effects; close points may have different prefixes |
| Geometry tree | Index of geometry envelopes | Envelope overlap is a candidate, not an exact geometry relation |

H3 cells include pentagons and vary in area. Their neighbors are not an
exact equal-distance metric stencil. See [H3 architecture](h3_architecture.md)
for the relevant source documentation and executable statistics.
[S2's own guide](https://s2geometry.io/devguide/s2cell_hierarchy.html) describes
its spherical cells, levels 0–30, and hierarchy. These levels are not equivalent
to H3 resolutions. Redis documents one concrete
[Geohash command implementation](https://redis.io/docs/latest/commands/geohash/);
a similarly named Python package may expose a different API.

## Match the question to the index

- For repeated aggregation by geographic cell, retain resolution, original
  coordinate support, and the treatment of points near cell boundaries.
- For a vector intersection query, use a geometry index to narrow candidates
  and an exact predicate to determine the result.
- For a radius or nearest-neighbor query, declare the distance metric and CRS.
  Cell adjacency and string-prefix agreement alone do not establish a metric
  cutoff.
- For a multi-resolution analysis, distinguish logical roll-up from clipping to
  an administrative or other geographic polygon.

## Executable H3 example

```python
import h3

cell = h3.latlng_to_cell(37.7749, -122.4194, 9)
assert cell == "89283082803ffff"
disk = h3.grid_disk(cell, 1)
assert cell in disk
assert len(disk) == 7  # Includes the origin, away from pentagons.
assert h3.get_num_cells(9) == 4_842_432_842
```

The direct H3 `grid_disk` includes the origin. In contrast, SPACE's
`SpatialIndexingInterface.get_cell_neighbors` excludes the origin. Consult the
actual function contract instead of treating similarly named methods as aliases.
The former example's printed cell identifier and trillion-cell count for
resolution 9 were incorrect.

## Performance comparisons require evidence

The earlier timing, speedup, memory, database-function-count, and cloud-support
tables had no reproducible benchmark or versioned compatibility evidence.
They should not guide capacity planning. In particular, eight bytes of raw
integer payload does not describe the total memory of Python string objects,
lists, or an indexed database.

A meaningful comparison should retain input data/hash, distribution, query
semantics, resolution/precision matching, implementation versions, interpreter,
hardware, warm/cold state, repeated timings, memory method, and a correctness
oracle. Compare equal queries and output accuracy before reporting speedups.
An index-only approximate answer and an exact polygon predicate are different
operations.

Use [backend profiling](../../../integration/backend_profiling.md) to capture a
bounded source-bound workload, and
[geometric algorithms](../../algorithms/geometric_algorithms.md) for a real
Shapely tree example. Optional S2, database, or service integrations require
separate dependencies and their own actual qualification.
