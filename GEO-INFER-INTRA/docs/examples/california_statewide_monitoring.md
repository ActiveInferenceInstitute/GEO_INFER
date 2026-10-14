# California: three environmental indicators and a shared monitoring budget

The question is how to allocate limited monitoring while keeping different
hazards, evidence types and approximation levels explicit. PLACE owns the
California orchestration. SPACE supplies sparse graph statistics and area
aggregation. ACT evaluates GNN factored policies and resource allocations.
The sibling Generalized_Notation_Notation repository owns the exporter.

The [runnable example](../../../GEO-INFER-PLACE/locations/california/README.md)
provides the commands and strict design file. The example is a historical
indicator and monitoring experiment. Its posterior probabilities refer to
declared binary indicators, rather than future hazard impacts.

## Measurements and primary sources

| Field | Source | Measurement | Default positive label |
| --- | --- | --- | --- |
| Administrative footprint | [California counties service](https://services.gis.ca.gov/arcgis/rest/services/Boundaries/CA_Counties/FeatureServer/0) | Union of all 58 county polygons, including any waters represented in that source | Defines the clipped analysis carrier |
| Fire | [CAL FIRE fire perimeters](https://www.fire.ca.gov/what-we-do/fire-resource-assessment-program/fire-perimeters) | Annual union of mapped perimeters intersecting each clipped cell | At least 1% of clipped cell area |
| Seismic activity | [USGS event catalogue](https://earthquake.usgs.gov/fdsnws/event/1/) | Annual count of catalogued earthquakes at or above the configured magnitude within each cell | At least one magnitude 3 or greater event |
| Drought | [U.S. Drought Monitor GIS data](https://droughtmonitor.unl.edu/DmData/GISData.aspx) | D1-or-worse polygon union at the last Tuesday of each year | At least 50% of clipped cell area |

The default analysis uses 2016–2023 for empirical priors and reserves 2024
indicator labels for the sensor experiment. Drought uses a **single year-end
snapshot** per year. Fire and earthquakes use annual windows. These distinct
windows are retained in the indicator roster. They do not become interchangeable
observations merely because their threshold labels are binary.

Fire perimeter coverage is not burn severity, and mapped records do not imply
a complete inventory of all fires. Earthquake location counts are not shaking
probabilities. Drought area is not annual drought frequency. No arbitrary sum
of these quantities is presented as a composite risk score. Population,
vulnerability, asset damage and causal compound-hazard interactions are not
represented by these indicator fields.

The drought layer is derived from the U.S. Drought Monitor, jointly produced
by NDMC at the University of Nebraska–Lincoln, USDA, NOAA and NASA. Credit the
partnership and [NDMC's attribution guidance](https://droughtmonitor.unl.edu/About/Permission.aspx)
when reusing its data or maps. These are derived H3 maps, rather than original
USDM map images. Preserve original source attribution separately from the
repository's software license.

## Acquisition and source custody

`acquisition.py` captures exact response bytes with a shared wall-clock deadline,
a 20 MiB per-response limit and a 128 MiB capture limit. It uses PLACE's existing
isolated download worker. Workers cannot leave a successful receipt after a
deadline failure. ArcGIS selection records metadata and object IDs both before
and after capture. Pages must contain exactly the requested distinct IDs and
must not report transfer truncation. USGS selections at the catalogue limit
are refused. Captured drought ZIPs retain their original projection metadata.

`check_sources` reads the closed file roster, design and hashes without network
access. This is supplied-receipt integrity. It is separate from an independent
primary-service recapture, geographic normalization, producer reproduction and
simulation replay. Local HTTP endpoints in tests are clearly identified
numerical fixtures and do not count as real California acquisition evidence.

## H3 geometry, boundaries and aggregation

The carrier uses overlapping H3 polygon coverage at resolution 5 by default,
then clips each candidate to the captured county union. Multipart fill results
are deduplicated before constructing the carrier. The union of clipped patches
must cover the source footprint within the declared residual tolerance. H3
resolution, valid cell IDs, unique order, parent resolution and complete
in-carrier adjacency are validated when loading a landscape.

Distances and areas in the geometry workflow use California Albers,
EPSG:3310. Latitude/longitude is retained for interchange. Source
polygons are densified before projection using the configurable maximum segment
length `projection_segment_degrees` (default 0.02 degree). County unions,
clipping, hazard intersections and area attribution all use the same metric
frame. County intersections must cover each patch within a relative tolerance
of 1e-8; a geographic/projected mismatch cannot be hidden by normalizing weights.
Earthquake points are attributed by containment in these same projected patches.
An exact shared-boundary tie selects the first index in the sorted cell roster.
Spherical H3 point indexing is not substituted for metric polygon containment;
a selected point with no containing patch fails rather than moving to a neighbor.
Captured ArcGIS queries default to a maximum allowable offset of 0.001 degree and output
coordinate precision of six decimal places. Both settings are configurable
and checked against the capture design. These generalized geometries support
an approximate administrative footprint and cell-area intersections; they
are not cadastral boundary measurements. Polygon repairs
require the explicit `repair_geometry` setting and produce a record for every
affected source feature. Repairing a polygon establishes usable topology; it
does not recover an unknowable intended boundary.

County polygons can overlap. The workflow retains the measured source overlap
and each patch's raw county-intersection coverage ratio. County allocation
weights are normalized intersected areas:

\[
w_{ic} = \frac{|H_i \cap C_c|}{\sum_d |H_i \cap C_d|}.
\]

This defines an allocation rule, not an authoritative boundary correction.
Parent and county summaries weight probabilities by **clipped cell area**.
The allocated county area is \(a_i w_{ic}\). Both area and area-weighted values
must be conserved by aggregation. Parent labels describe H3 hierarchy;
aggregated support is the allocated child footprint, rather than an assertion
that a parent polygon exactly equals the union of child polygons.

SPACE's `weighted_group_means` is reusable for other spatial carriers. Its
`graph_moran` computes a sparse, undirected Moran statistic with a declared
permutation count and seed. The two-sided permutation comparison is centered
on the randomization expectation, with a plus-one correction. Isolated nodes
are counted. Constant fields and invalid graphs are refused. This descriptive
randomization test conditions on the captured graph and values; it does not
identify causal effects or establish physical forecasting skill.

## Empirical priors and spatial inference

For each hazard, let \(y_{it}\) be its earlier-year binary label, let \(T\)
be the number of training years, and let \(\bar y\) be the mean training label
over the cell-year carrier. The full-support empirical prior is

\[
p_i = \frac{\sum_{t < t_*} y_{it} + \kappa \bar y + 1/2}{T + \kappa + 1}.
\]

The target-year label does not enter this calculation. A negative control
changes target truth while retaining training labels and confirms that the
first allocation and policy costs remain unchanged.

Spatial fields use a declared binary Markov interaction on the H3 adjacency
graph, with edge weight \(1/6\). For cumulative local sensor log likelihood
\(\ell_i\), the mean-field update is

\[
q_i = \sigma\left(\operatorname{logit}(p_i) + \ell_i(1)-\ell_i(0)
       + \frac{\beta}{6}\sum_{j\sim i}(2q_j-1)\right).
\]

The implementation damps updates by one half and records the fixed-point
residual, iteration count and variational objective, excluding the unknown
joint normalizing constant. Degree is at most six and \(0\leq\beta\leq1\);
the undamped map's infinity-norm derivative bound is at most \(\beta/2\).
Exhausted iteration budgets and loss of floating-point full support fail.
Zero coupling is tested against independent binary Bayes. A two-node test
also compares the approximation with enumeration of the exact joint field.

The three hazard fields are conditionally separate. Their common monitoring
budget couples their actions. This does not introduce a physical causal link
between fire, earthquakes and drought.

## Real GNN models and exact local active inference

Each explicit GNN source declares two factors: binary indicator status and
sensor mode (passive, coarse or detailed). There are six joint states. One
control factor selects the next sensor mode. The indicator is static during
the teaching episode. Modalities are indicator report (no report, negative or
positive) and deterministic resource use.

Sensor coverage and classification accuracy are **declared assumptions**,
configurable independently from the historical indicators. Default coarse
coverage/accuracy are 0.8/0.8; detailed coverage/accuracy are 0.95/0.97.
Passive mode produces no report. Resource-use log preferences are
\((0,-c,-3c)\), with default \(c=0.03\). The enumerated one-step policy prior
is uniform. Preferences are log utilities; the resource units used by the
hard budget are separately declared as \((0,1,3)\).

The GNN exporter validates the explicit factor, control and modality axes and
produces `gnn-geo-infer/factored/1`. A separate GNN process independently
reproduces each artifact. The PLACE consumer checks source bytes, declared
arrays, physical step duration, source provenance and the ACT contract.
Matching dimensions alone are insufficient.

ACT's `FactoredPolicyEvaluator` compiles joint transition and emission tensors
once and evaluates batches of posterior beliefs. It computes exactly

\[
G_i(u) = -\mathbb E[\log C(o)\mid u] - I(s;o\mid u).
\]

Its outputs retain expected preference cost, information gain, expected free
energy and the support-aware policy posterior. Costs use the declared
unnormalized log utilities. Normalizing them would add an action-independent
constant, leaving policy selection and passive-relative allocation gains
unchanged. Batch outputs are compared against ACT's existing exact
observation-tree solver, including correlated
joint beliefs and zero-prior policies. The batch API supports **explicit
one-step policies**; longer horizons are refused and remain the reference
solver's responsibility. This is real categorical active inference, with no
unadvertised approximation inside the local policy evaluation.

The field belief supplied to each local model is mean-field approximate.
Summing local policy gains does not give exact global epistemic value for a
correlated statewide field. Exact local inference, approximate spatial
beliefs and approximate global factorization must be described separately.

## Shared resource allocation and simulation

ACT's `allocate_resources` solves a finite multiple-choice knapsack exactly.
Each cell-hazard site can receive one sensor action, with a shared budget of
96 units by default. The objective maximizes the sum of passive-policy EFE
minus selected-policy EFE. Brute-force small examples verify optimality,
including optional group expenditure floors and infeasible constraints.

Three strategies are retained:

| Strategy | Allocation rule | Interpretation |
| --- | --- | --- |
| `active` | Exact optimum for pooled local EFE gains | High uncertainty in one hazard can dominate the budget |
| `quota_active` | Exact optimum subject to a configurable expenditure floor per hazard | The default 12-unit floor is a declared allocation constraint |
| `rotating` | Seeded rotating coarse observations under the same budget ceiling | Observation baseline with the same sensor likelihood and inference |

Counter-seeded simulated reports use the actual GNN likelihood tensor and
held-out historical indicator labels. Reports at a given hazard, cell, round,
seed and mode replay independently of call order. The planner has no target
truth argument. Observed likelihoods accumulate once, and spatial inference
restarts from the fixed empirical prior rather than re-assimilating an already
conditioned posterior as new evidence.

Brier score, log score and posterior entropy are retained per hazard and round.
Variational residuals and resource accounting are checked. Scores against
historical labels characterize this simulated sensor experiment. Two sensor
seeds do not establish calibration, generalization or future field performance.
Changing expenditure floors changes the allocation objective; it must not be
reported as a free improvement to an otherwise identical policy.

## Configurability and validation

The design exposes H3 resolution/parent resolution, training and target years,
fire/drought thresholds, drought category, earthquake magnitude, prior
shrinkage, spatial coupling, sensor accuracy/coverage, observation cost,
rounds, seeds, total budget, hazard floors, repair permission and cell limits.
Unknown fields, booleans in numeric controls, nonfinite values and excessive
work are rejected. Source selection must agree with the captured year and
magnitude roster. Changed sensor parameters require regenerated and
independently reproduced GNN artifacts.

The CLI has one operation per invocation and rejects unused/conflicting inputs.
Its libraries own the methods; orchestration scripts do not implement inference.
Existing PLACE public exports load on demand, so importing acquisition or the
California configuration does not import unrelated forecasting or ML components.
Missing requested components still raise real import failures.

Source receipt integrity, actual GNN reproduction, numerical replay, image
inspection and video acceptance are separate checks. No catalogue output,
numerical agreement or visualization establishes native Lean proof.
