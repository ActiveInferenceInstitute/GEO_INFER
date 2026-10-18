# Mobility Analysis with H3

H3 can group ordered location observations into cells and aggregate trip
endpoints. It does not infer a road route, travel time, or exact physical
movement from cell adjacency. The examples here use synthetic coordinates and
real H3 calls; they are not a validated mobility model or measured trip dataset.

## Preserve the observation contract

Before binning, validate coordinate ranges and CRS, timestamps and timezone,
trip segmentation, observation order, duplicate timestamps, and missing samples.
Retain the original observations and measurement precision. Sorting an entire
vehicle history is not the same as identifying individual trips, and removing
repeated cells discards dwell-time information unless timestamps are retained
separately.

## Executable trajectory and endpoint aggregation

```python
from collections import Counter
from itertools import groupby
import h3


def trajectory_cells(points, resolution=9):
    cells = [h3.latlng_to_cell(lat, lng, resolution) for lat, lng in points]
    return [cell for cell, _ in groupby(cells)]


# Ordered, synthetic latitude/longitude observations for one already segmented trip.
points = [
    (37.7749, -122.4194),
    (37.7749, -122.4194),
    (37.7849, -122.4094),
    (37.7749, -122.4194),
]
sequence = trajectory_cells(points)
assert len(sequence) == 3
assert sequence[0] == sequence[-1]  # A revisit survives consecutive deduplication.
assert sequence[0] != sequence[1]
assert trajectory_cells([]) == []

trips = {"synthetic-trip-1": sequence}
origin_destination = Counter((cells[0], cells[-1]) for cells in trips.values() if cells)
assert sum(origin_destination.values()) == 1

coarse_od = Counter()
for (origin, destination), count in origin_destination.items():
    coarse_od[(h3.cell_to_parent(origin, 7), h3.cell_to_parent(destination, 7))] += count
assert sum(coarse_od.values()) == sum(origin_destination.values())
```

`groupby` removes only consecutive repeats, preserving a later return to a cell.
The endpoint aggregation counts one record per supplied trip, including a trip
that ends in its origin cell. An empty trip contributes nothing. The coarser
roll-up preserves the count but loses finer spatial detail; it does not mean
the actual path was contained by each parent's geographic polygon.

## Distance, duration, and routes

[H3 grid traversal](https://h3geo.org/docs/api/traversal/) defines grid distance
in adjacency hops and documents cases in which a distance or path operation
fails. Do not replace a failure with a fabricated route. A grid path does not
account for roads, one-way restrictions, barriers, transport modes, or GPS gaps.

For physical speed, use a declared spatial distance method and a positive elapsed
time from the original ordered observations. State how position error, sampling
gaps, stopped periods, and outliers are treated. Cell-center distances lose
within-cell movement. A radius expressed as a number of rings is not an exact
metric service area or isochrone.

## Neighborhood and density limits

H3 has pentagons and position-dependent cell geometry. It is not a uniform
metric lattice. Use actual cell areas for density and retain resolution in every
aggregate. A small example away from pentagons does not prove that every cell
has six neighbors. See [architecture](h3_architecture.md) and
[spatial indexing](../../algorithms/spatial_indexing.md).

## Evaluation and publication

Validate a mobility analysis against its intended question: endpoint counts,
route matching, travel-time prediction, and accessibility need different
reference data. Hold out appropriate trips or time periods, preserve spatial
and temporal leakage controls, and report errors in physical units where the
target is physical. No generic H3 example establishes predictive accuracy.

For a reproducible result, retain input provenance, trip definition, preprocessing,
coordinate/time conventions, resolution, code and dependency revisions, output
schema, and an actual validation receipt. Publish counts and uncertainty at a
support appropriate to the data and the authorized application.
