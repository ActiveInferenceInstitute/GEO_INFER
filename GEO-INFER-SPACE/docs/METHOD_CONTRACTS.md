# SPACE method contracts and review boundaries

SPACE separates ordered H3 domains, geometric coordinates, temporal axes and
backend capabilities. Start with the public composition surfaces
`H3StateSpace` and `align_h3_observations`; see
[CROSS_MODULE_COMPOSITION.md](CROSS_MODULE_COMPOSITION.md) for DATA/TIME/ACT
integration and missing observations.

## Native H3 methods

`core.spatial_methods.SpatialMethods` validates canonical, unique,
same-resolution cell domains for observation operations. Set overlay and
coverage deduplicate repeated cells explicitly. Filters retain input order.
Aggregation requires a coarser or equal resolution and a declared reducer;
unknown reducers raise. Equal disaggregation conserves extensive totals;
`proportional` replicates an intensive per-cell value. It does not represent
an area-weighted mass transfer. Complete pentagon children have six cells,
while ordinary hexagon children have seven at the next resolution.

`buffer_analysis` returns disjoint minimum-distance rings around the source
union. `calculate_spatial_weights` uses disk neighborhoods excluding the
focal cell, with row standardization. Rook and queen agree on H3 edge
adjacency. `find_spatial_outliers` returns descriptive standardized Moran
quadrants and sets `significance_tested=False`; it does not run a permutation
significance test.

Topology failures propagate from H3 cell/grid methods, density, interpolation,
weights, clustering and transition construction. Unsupported pentagon grid
distance pairs are errors, even when adjacency neighborhoods are available.
The explicit CPU distance-matrix kernel retains its documented `-1` sentinel
for incomparable pairs. Callers must check that sentinel before arithmetic.
The H3 interpolation backend's `linear` option means inverse-distance weights
with power one; it is not a piecewise-linear triangulation method.

`H3Cell` rejects noncanonical indexes and mismatched resolutions. Its identity,
derived geometry and creation instant are immutable. Observation `properties`
remain mutable and own a deep copy of the supplied mapping. Exported GeoJSON
and DataFrame properties also own their nested values. `H3Grid.cells` is a
read-only tuple snapshot; membership changes go through `add_cell` and `remove_cell`,
which update the lookup index together and invalidate the snapshot. Batch
additions retain constant-time append; previous snapshots keep their membership.
Construct a new cell to change its
identity instead of assigning to `index` or `resolution`.
`H3Grid.bounds()` encloses actual cell boundary vertices, including a single
cell. Its longitude extrema use conventional [-180, 180] coordinates; a grid
crossing the antimeridian has a wide span, not a wrapped interval.

Explicit invalid
parent/child resolutions raise; the default parent at resolution zero is
`None`, and default children at resolution fifteen are empty. `children`,
`neighbors`, `H3Grid.from_center`, `H3Grid.uncompact`, spatial buffers and disaggregation
accept `max_cells` and bound allocation before constructing the result.
The native `H3Backend` ring/disk, children and uncompact entry points have the
same explicit allocation budget. Uncompact validates ancestor overlap in
at most fifteen parent lookups per source cell, without comparing every pair.
`SpatialIndexingInterface.get_neighbors` uses average H3 edge length to choose
a conservative grid-ring count; this is a discretization rule, not an exact
metric-distance predicate. It also accepts `max_cells`. Use a geodesic
predicate when exact distance membership matters.

```python
import h3
from geo_infer_space.core.spatial_methods import SpatialMethods

methods = SpatialMethods()
parent = h3.latlng_to_cell(37.7, -122.4, 8)
children = methods.disaggregate_to_cells([parent], [42.0], 9)
assert abs(sum(children["disaggregated"].values()) - 42.0) < 1e-12
back = methods.aggregate_to_region(
    list(children["disaggregated"]),
    list(children["disaggregated"].values()),
    8,
    "sum",
)
assert abs(back["aggregated"][parent]["value"] - 42.0) < 1e-12
```

## Time, observations and metadata

Generic and H3 temporal pattern analyzers require explicitly aware instants
and normalize to UTC through TIME. Their hour/day/week/month pattern buckets
are cyclic calendar components, not chronological resampling bins. Overall
mean and standard deviation weight all observed values, including within-bucket
variance. Missing values are excluded; an observed zero participates.
`H3TemporalAnalyzer` accepts one grid or an ordered list of grids and requires
unique, increasing creation instants. Trends and anomalies exclude absent
properties instead of inventing zero observations.

H3 ML features use UTC calendar components. Missing target values do not become
training targets. Empty neighbor means and extrema are NaN; explicit counts,
density and sums identify the empty neighborhood. Declared observed targets
must be finite numbers. Feature preprocessing must choose an explicit missing
value strategy for the selected estimator.

Cell/grid/dataset and nested-system generated metadata uses aware UTC defaults.
Messages normalize supplied metadata timestamps before expiry comparisons;
zero TTL means immediate expiry. Spatial Pydantic metadata models validate raw
inputs before datetime coercion, so implicit numeric epochs are rejected.

## Geometries, dispatch and configuration

`GeometricOperationsInterface.transform_geometry` dispatches on the geometric
capability surface. Vector topology operations keep coordinates bound to their
actual CRS during metric operations and restore the source CRS afterward.
Unknown operations fail before computation. GeoLibre reference bounds traverse
Point, all multi-geometries and GeometryCollection, including null geometries.

Required H3 loading and internal dispatcher/import errors propagate. SRAI's
missing extra is handled only when the missing module is `srai` itself;
broken transitive imports are installation failures. Optional-backend absence
tests do not establish real backend behavior.

`LocationConfigLoader` merges defaults, packaged/base YAML and location YAML
into independent mappings. Returned configuration and cached snapshots do not
share nested mutable values. Location identifiers cannot escape the config
directory; bounds must be finite WGS84 coordinates, and crossing the
antimeridian is supported. Resolution zero is valid, booleans and fractional
resolutions are rejected.

`BaseAnalysisModule` loads an explicitly supplied JSON/YAML mapping and raises
for a malformed or missing file. `spatial.h3_resolution` supplies the default;
an explicit constructor `h3_resolution` overrides it. The optional constructor
`output_dir` controls cache placement. Invalid configuration is rejected before
output-directory creation. Processing durations use a monotonic clock.

File processing requires a declared CRS and transforms coordinates to WGS84
before native H3 operations. Points map to one containing cell. Polygon and
MultiPolygon coverage preserves holes and uses native center containment;
a polygon smaller than a cell may yield no centers. Unsupported, invalid or
empty geometries and backend failures raise instead of yielding partial or
empty success results. Feature IDs, zero values and source properties survive
indexing, and geometries in the result use WGS84.

`run_analysis(use_cache=True)` validates local cache H3 identities and reuses
that derived artifact. It does not establish upstream freshness. Set
`use_cache=False` to reacquire and regenerate explicitly. New cache bytes are
written and synced to a private temporary file before atomic replacement;
failed acquisition, conversion, persistence or final analysis propagates.
Invalid existing cache bytes survive failed regeneration for diagnosis. The
old file-size heuristic and swallowed processing failures are removed.

## Statistical reference assumptions

`SpatialStatistics.nearest_neighbor_index(cells)` returns actual WGS84 geodesic
centroid nearest distances in kilometres with `reference_tested=False`.
Pass `study_area_km2=...` to obtain a Clark-Evans ratio and a normal-approximation
p-value against homogeneous complete spatial randomness. The area must be
finite and positive. Edge effects are not corrected, and the output reports
`edge_corrected=False`. H3 grid steps are not metric distances and observation
count alone cannot determine spatial density.

`getis_ord_g(cells, values, distance=1)` requires a nonempty canonical, unique,
same-resolution H3 axis, matching finite values and a positive integer grid
distance. Its binary neighborhood includes the focal cell. Undefined G* scores
are `None`, with per-cell reasons in `undefined`: too few observations, zero
global variance, or a neighborhood covering the entire observed domain. These
cells do not become hotspots or coldspots. Backend and topology failures raise.

`variance_mean_ratio(values)` accepts nonnegative finite counts or intensities.
Fewer than two values or a zero mean produce `None` for the ratio, chi-square,
p-value and pattern, with an explicit `undefined` reason. Fractional intensities
receive a descriptive ratio and `reference_tested=False`. Integer counts use a
two-sided chi-square reference with the stated assumption of independent Poisson
counts and equal exposure.

`quadrat_count(cells, values=None, quadrat_size=2)` uses integer parent-resolution
steps from zero through the common input resolution. Zero retains exact cell
IDs. The entire cell/value axis is validated before dispatch; duplicate or mixed
resolution cells, mismatched lengths, negative or nonfinite counts, and invalid
steps raise. Results retain `quadrat_counts` by actual parent ID,
`parent_resolution` and `scope="observed_parent_quadrats"`. Unobserved quadrats
are omitted; observed zero counts remain zero. Undefined dispersion results stay
`None`, and backend failures propagate without a partial result.

Migration: callers expecting `nni`, `pattern`, `z_score` or `p_value` must now
supply their real study window area. Cell/grid callers relying on swallowed
invalid topology must handle raised errors. Naive timestamps require an
explicit localization before ingestion. ML callers must handle NaN neighbor
features deliberately. Quadrat callers must pass integer resolution steps and
complete value axes; statistical callers must handle explicit `None` for
undefined results instead of interpreting an invented zero score as evidence.

## Verification scope

The method inventory indexes every SPACE source callable with signatures,
source hashes and proof classifications. AST inventory, diff inspection,
executed-line coverage and analytical oracle tests are separate evidence.
Executing a line does not prove the method's numerical correctness.

`tests/unit/test_method_contracts.py`, `test_spatial_methods.py` and
`test_statistics_failure_contracts.py` provide tiny
real hexagon/pentagon, mass conservation, UTC, metadata, configuration,
CRS/ownership and statistical reference oracles. Failure-injection tests check
that unsupported topology and broken imports propagate; they do not replace
real backend tests. The complete unit and integration suites remain required.

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module SPACE --category unit
uv run python GEO-INFER-TEST/run_unified_tests.py --module SPACE --category integration
uv run python GEO-INFER-TEST/measure_module_coverage.py --modules GEO-INFER-SPACE
uv run python GEO-INFER-SPACE/examples/demo_all_methods.py --resolution 8 --max-cells 100
```

The example retains its historical filename but verifies six stated native
composition invariants. Importing it performs no analysis. It does not claim
to verify every method. SPACE currently has no dedicated performance or system
test files; aggregate lanes must record those module selections as empty while
executing the modules that declare those categories.

CPU tests establish CPU behavior. GPU execution, SRAI optional algorithms,
licensed Whitebox tools, remote services and historical PROJ causality require
separate environment-specific evidence. No passing claim for them follows
from an import probe, static inventory or absent-extra test.
