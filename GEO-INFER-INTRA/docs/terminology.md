# GEO-INFER Terminology Glossary

This glossary defines the technical terms used throughout the GEO-INFER framework
documentation, source code, and API. Terms are organized into three sections:
Active Inference, Geospatial, and GEO-INFER-Specific.

---

## Section A: Active Inference Terms

**action**: An intervention that an agent executes to change its environment or
its own sensory state. In Active Inference, actions are selected to minimize
expected free energy. See also: *policy*, *expected free energy*.

**active inference**: A framework derived from the Free Energy Principle that
unifies perception, learning, and decision-making as processes of minimizing
variational free energy (perception) and expected free energy (action). Unlike
passive inference, the agent actively samples the environment to reduce
uncertainty. See also: *free energy principle*, *passive inference*.

**affordance**: An opportunity for action that the environment presents to an
agent. In Active Inference, affordances are encoded as feasible policies that
reduce expected free energy. See also: *policy*, *epistemic value*.

**attention**: The process of modulating the precision (inverse variance) of
sensory signals. High attention on a modality increases its influence on belief
updating. Implemented in GEO-INFER-COG. See also: *precision*.

**belief**: A probability distribution representing the agent's estimate of
hidden states. Updated through Bayesian inference when new observations arrive.
In code: a numpy array that sums to 1.0. See also: *posterior*, *prior*.

**ELBO (Evidence Lower Bound)**: A lower bound on the log model evidence
`log p(o)`. Maximizing the ELBO is equivalent to minimizing variational free
energy: `ELBO = -F`. See also: *variational free energy*.

**epistemic value**: The component of expected free energy that quantifies
information gain --- the expected reduction in uncertainty about hidden states
from taking an action. Drives exploratory behavior. See also:
*pragmatic value*, *expected free energy*, *information gain*.

**expected free energy (EFE)**: The quantity `G(pi)` that agents minimize when
selecting policies. Decomposes into epistemic value (information gain) and
pragmatic value (goal achievement): `G = -epistemic - pragmatic`. Lower EFE
means a policy is more desirable. See also: *free energy*, *policy selection*.

**free energy**: In Active Inference, a functional `F` that provides an upper
bound on surprise (negative log model evidence):
`F = E_q[log q(s) - log p(o,s)]`. Minimizing free energy through perception
brings the approximate posterior closer to the true posterior. See also:
*variational free energy*, *surprise*.

**free energy principle (FEP)**: The principle that any self-organizing system
at nonequilibrium steady state must minimize its variational free energy.
Proposed by Karl Friston. The theoretical foundation of Active Inference.

**generative model**: A probabilistic model specifying how hidden states produce
observations. Components: prior `P(s)`, likelihood `P(o|s)`, transition dynamics
`P(s'|s,a)`, and preferences `P(o)`. The agent uses this model for both
inference and planning. See also: *recognition model*, *likelihood*, *prior*.

**hidden state**: An unobservable variable in the environment that the agent
infers from sensory observations. In GEO-INFER, hidden states include true
soil moisture, actual species abundance, real traffic density, and other
quantities that sensors measure imperfectly. See also: *observation*, *belief*.

**information gain**: The expected reduction in entropy of the posterior over
hidden states from making an observation. A key component of epistemic value.
Formally: `I = E_q[D_KL[q(s|o,pi) || q(s|pi)]]`. See also: *epistemic value*.

**KL divergence**: Kullback-Leibler divergence `D_KL[q || p]`, a non-negative
measure of the difference between two probability distributions. In free energy:
measures the divergence of the approximate posterior from the prior (complexity).
See also: *free energy*, *ELBO*.

**Laplace approximation**: An approximation of the posterior distribution using
a Gaussian centered at the mode, with covariance given by the inverse Hessian
of the negative log posterior. Used in some Active Inference implementations
for continuous state spaces. See also: *variational Bayes*.

**likelihood**: The probability of an observation given a hidden state, `P(o|s)`.
In GEO-INFER-ACT, stored as the matrix `A` where `A[o,s] = P(o|s)`. See also:
*generative model*, *observation model*.

**Markov blanket**: A statistical boundary that separates a system from its
environment. Consists of sensory states (receiving information from the
environment) and active states (influencing the environment). Internal states
are conditionally independent of external states given the blanket. See also:
*active inference*, *free energy principle*.

**observation**: Sensory data that the agent receives from the environment.
In GEO-INFER, observations include satellite imagery pixels, sensor readings,
survey responses, and GPS traces. See also: *hidden state*, *likelihood*.

**observation model**: The mapping from hidden states to observations, defining
the likelihood function `P(o|s)`. Also called the likelihood matrix or the `A`
matrix. See also: *likelihood*, *generative model*.

**passive inference**: Inference without action --- updating beliefs based on
observations without actively changing the environment. Standard Bayesian
inference is passive. Active Inference extends this to include action selection.
See also: *active inference*, *perceptual inference*.

**perceptual inference**: The process of updating beliefs about hidden states to
minimize free energy, holding actions fixed. Also called state estimation or
recognition. See also: *active inference*, *belief*.

**policy**: A sequence of actions over a planning horizon. In Active Inference,
agents evaluate multiple policies by computing their expected free energy and
select probabilistically. See also: *policy selection*, *expected free energy*.

**policy selection**: The process of choosing a policy by computing expected free
energy for each candidate policy and applying a softmax function weighted by
precision: `P(pi) = softmax(-gamma * G(pi))`. See also: *policy*,
*expected free energy*, *precision*.

**posterior**: The probability distribution over hidden states after incorporating
observations: `P(s|o)`. The result of Bayesian inference. In variational methods,
approximated by `q(s)`. See also: *prior*, *belief*.

**pragmatic value**: The component of expected free energy that quantifies how
well a policy achieves the agent's preferred outcomes. Drives goal-directed
behavior. See also: *epistemic value*, *expected free energy*, *preferences*.

**precision**: The inverse variance of a probability distribution. In Active
Inference, precision controls the relative influence of different information
sources: high sensory precision means the agent trusts its observations; high
prior precision means the agent trusts its model. See also: *attention*.

**prediction error**: The difference between predicted and actual observations.
Drives belief updating: large prediction errors cause large belief changes. In
predictive coding, both sensory prediction errors (bottom-up) and prior
prediction errors (top-down) are computed. See also: *predictive coding*.

**predictive coding**: A theory of neural processing where each level of a
hierarchy predicts the activity of the level below and passes up prediction
errors. A process theory implementation of variational inference. See also:
*prediction error*, *perceptual inference*.

**preferences**: The agent's desired observations, encoded as a log-probability
distribution `C` over observations. High values in `C` for an observation mean
the agent prefers that outcome. See also: *pragmatic value*.

**prior**: A probability distribution representing beliefs before observing data.
In Active Inference: `D` for state priors, `E` for policy priors. See also:
*posterior*, *belief*.

**recognition model**: The mapping from observations to hidden states, `q(s|o)`.
In variational inference, this is the approximate posterior. Sometimes called the
encoder. See also: *generative model*.

**sensory attenuation**: The reduction of sensory precision during action
execution, allowing the agent to move without being pulled back by prediction
errors from its current position. See also: *precision*, *active inference*.

**surprise**: The negative log probability of an observation under the generative
model: `-log p(o)`. Also called surprisal or self-information. Free energy is an
upper bound on surprise. See also: *free energy*.

**variational Bayes**: A family of methods that approximate intractable posterior
distributions by optimizing a simpler distribution `q(s)` to minimize the
KL divergence from the true posterior. The core inference algorithm in Active
Inference. See also: *ELBO*, *KL divergence*, *free energy*.

**variational free energy**: The specific functional minimized in variational
inference: `F = E_q[log q(s) - log p(o,s)]`. Provides an upper bound on surprise
and a lower bound (negated) on the log model evidence. See also: *free energy*,
*ELBO*.

---

## Section B: Geospatial Terms

**bounding box**: A rectangle defined by minimum and maximum coordinates that
encloses a geographic feature: `(min_lng, min_lat, max_lng, max_lat)`. Used for
spatial filtering and extent queries. See also: *convex hull*.

**buffer**: A geometric operation that creates a zone of specified distance
around a point, line, or polygon. In geographic coordinates, buffer distance
is in degrees unless the geometry is first projected to a metric CRS. See also:
*CRS*, *projection*.

**cell**: A single spatial unit in a grid system. In H3, a hexagonal cell at
a given resolution. See also: *H3*, *resolution*.

**centroid**: The geometric center of a feature. For a polygon, the average of
all vertex coordinates (not guaranteed to be inside the polygon for concave
shapes). See also: *polygon*.

**choropleth**: A thematic map where geographic areas are shaded according to a
statistical variable. Used for visualizing per-region data like population
density or temperature. See also: *vector*.

**clip**: A spatial operation that extracts the portion of one geometry that falls
within another geometry. See also: *intersection*, *overlay*.

**convex hull**: The smallest convex polygon that contains all points of a
geometry. See also: *polygon*, *bounding box*.

**coordinate system**: A system for specifying positions on the Earth's surface.
Geographic coordinate systems use latitude/longitude in degrees. Projected
coordinate systems use easting/northing in meters. See also: *CRS*, *EPSG*.

**CRS (Coordinate Reference System)**: A framework that defines how 2D
coordinates map to locations on the Earth. Identified by EPSG codes. GEO-INFER
default: EPSG:4326 (WGS84). See also: *EPSG*, *WGS84*, *projection*.

**difference**: A spatial operation producing the portion of geometry A that does
not overlap geometry B. See also: *intersection*, *union*.

**dissolve**: A GIS operation that merges adjacent polygons sharing a common
attribute into a single polygon. See also: *polygon*, *overlay*.

**EPSG**: The EPSG Geodetic Parameter Dataset, providing standardized numeric
codes for coordinate reference systems. EPSG:4326 = WGS84 geographic,
EPSG:3857 = Web Mercator. See also: *CRS*, *WGS84*.

**GeoDataFrame**: A pandas DataFrame with a `geometry` column containing Shapely
geometry objects and a CRS property. The primary vector data structure in
GeoPandas and GEO-INFER. See also: *geometry*, *CRS*.

**geohash**: A geocoding system that encodes a location into a short string of
letters and digits, representing a rectangular cell. Less used in GEO-INFER than
H3. See also: *H3*, *cell*.

**geometry**: A spatial object representing a geographic feature: Point,
LineString, Polygon, or their Multi- variants. Implemented using Shapely in
Python. See also: *vector*, *point*, *polygon*, *linestring*.

**H3**: A hierarchical hexagonal geospatial indexing system developed by Uber.
Provides 16 resolutions (0-15) of hexagonal cells covering the globe. GEO-INFER
uses H3 v4 API exclusively. See also: *cell*, *resolution*, *hexagonal grid*.

**hexagonal grid**: A tessellation of the plane using regular hexagons. Preferred
over square grids because hexagons have uniform adjacency (6 neighbors, all
equidistant from center) and less sampling bias. See also: *H3*, *tessellation*.

**intersection**: A spatial operation producing the area shared by two
geometries. See also: *union*, *difference*, *clip*.

**linestring**: A geometry type representing an ordered sequence of connected
points forming a line. Used for roads, rivers, trails. See also: *geometry*,
*point*, *polygon*.

**overlay**: A spatial operation that combines two geometry layers, computing
intersections, unions, or differences between features. See also: *intersection*,
*union*, *difference*.

**point**: The simplest geometry type: a single coordinate pair (x, y). Used for
sensor locations, addresses, landmarks. See also: *geometry*, *linestring*,
*polygon*.

**polygon**: A geometry type representing a closed area bounded by a ring of
coordinates. May contain holes (interior rings). See also: *geometry*, *point*,
*linestring*.

**projection**: A mathematical transformation from the Earth's curved surface
to a flat plane. Different projections preserve different properties: area
(equal-area), shape (conformal), or distance (equidistant). See also: *CRS*,
*EPSG*.

**raster**: A spatial data model representing the world as a regular grid of
cells (pixels), each with a value. Used for elevation, satellite imagery,
temperature fields. See also: *vector*, *GeoTIFF*.

**resolution**: In H3, a level from 0 (coarsest, ~4.3M km^2/cell) to 15
(finest, ~0.0009 m^2/cell) that determines cell size. See also: *H3*, *cell*.

**spatial index**: A data structure (R-tree, QuadTree, H3 grid) that accelerates
spatial queries by organizing geometries for fast retrieval based on location.
See also: *H3*, *spatial join*.

**spatial join**: An operation that combines attributes from two datasets based
on spatial relationships (contains, intersects, within). See also:
*spatial index*, *overlay*.

**union**: A spatial operation producing a geometry that covers the area of both
input geometries combined. See also: *intersection*, *difference*.

**vector**: A spatial data model representing geographic features as discrete
objects with geometry (point, line, polygon) and attributes. The complement of
raster. See also: *raster*, *geometry*, *GeoDataFrame*.

**WGS84**: World Geodetic System 1984, the standard geographic coordinate system
(EPSG:4326). Uses latitude/longitude in decimal degrees on a reference ellipsoid.
The default CRS in GEO-INFER. See also: *CRS*, *EPSG*.

---

## Section C: GEO-INFER-Specific Terms

**Active Inference Agent**: A software agent in GEO-INFER-ACT that implements
the Active Inference perception-action loop. Maintains beliefs, receives
observations, updates beliefs via free energy minimization, and selects actions
via expected free energy. See: `ActiveInferenceModel` class. See also:
*active inference*, *generative model*.

**AGENTS.md**: A file present in each module describing the module's capabilities
for multi-agent orchestration, including input/output formats and communication
patterns. See also: *SKILL.md*, *module*.

**belief updating (GEO-INFER)**: In GEO-INFER-ACT, the `perceive()` method on
`ActiveInferenceModel` that updates the agent's beliefs given a new observation.
Uses the `BayesianBeliefUpdate` class internally. See also: *belief*,
*perceptual inference*.

**conftest.py**: Pytest configuration file containing shared fixtures used across
test files. Present at the repository root and optionally within each module's
`tests/` directory. See also: *unified test runner*.

**free energy minimization (geospatial)**: Applying the free energy minimization
algorithm to spatial data: the agent's beliefs are distributed over H3 cells or
geographic features, and observations are spatial measurements (NDVI, temperature,
traffic flow). See also: *free energy*, *spatial prior*.

**generative model (GEO-INFER)**: An instance of the `GenerativeModel` class in
GEO-INFER-ACT, configured with an observation model (A matrix), transition model
(B matrix), state prior (D vector), and preferences (C vector). See also:
*generative model* (Active Inference term).

**graceful degradation**: The GEO-INFER pattern where `__init__.py` files wrap
imports in `try/except` blocks so that modules remain importable even when
optional dependencies are missing. Unavailable classes are omitted from
`__all__` rather than raising ImportError at import time. See also:
*try/except import*.

**H3 backend**: The spatial indexing backend in GEO-INFER-SPACE that uses the
H3 library for hexagonal grid operations. The default backend. See also:
*SRAI backend*, *H3*.

**module**: A self-contained package within the GEO-INFER monorepo. Each module
resides in a directory named `GEO-INFER-XXXX/`, contains its own `pyproject.toml`,
`src/`, `tests/`, `README.md`, `SKILL.md`, and `AGENTS.md`. There are 45 modules.

**module orchestrator**: A component in GEO-INFER-EXAMPLES that chains operations
across multiple modules in a pipeline. Defined in
`geo_infer_examples.core.module_orchestrator`. See also: *module*.

**policy (GEO-INFER)**: In GEO-INFER-ACT, a sequence of actions evaluated by the
`PolicySelector` class. The agent computes expected free energy for each policy
and selects via precision-weighted softmax. See also: *policy selection*,
*expected free energy*.

**precision-weighted prediction error**: Prediction error multiplied by the
precision of the observation channel. High-precision errors drive larger belief
updates; low-precision errors are attenuated. Central to how GEO-INFER-ACT
modulates the influence of noisy observations. See also: *precision*,
*prediction error*.

**pyproject.toml**: The standard Python project configuration file. Each GEO-INFER
module has one specifying its name, version, dependencies, and build settings.
See also: *module*, *uv*.

**SKILL.md**: A file in each module that Claude Code auto-discovers. Contains
YAML front matter (name, description, prerequisites, difficulty, estimated_time)
and markdown instructions for working with the module. See also: *AGENTS.md*,
*module*.

**spatial prior**: A prior distribution over hidden states that incorporates
spatial structure, such as distance decay from a center point or correlation
between adjacent H3 cells. See also: *prior*, *belief*, *H3*.

**SRAI backend**: An alternative spatial indexing backend in GEO-INFER-SPACE that
uses the SRAI library. Provides additional regionalization methods beyond H3.
See also: *H3 backend*.

**temporal prior**: A prior distribution that encodes temporal structure, such as
seasonal cycles, diurnal patterns, or temporal autocorrelation. See also:
*spatial prior*, *prior*.

**try/except import**: The coding pattern used in GEO-INFER `__init__.py` files
to handle optional dependencies gracefully:
`try: from .core.x import X; except ImportError: pass`. See also:
*graceful degradation*.

**unified test runner**: The script `GEO-INFER-TEST/run_unified_tests.py` that
discovers and runs tests across all 45 modules. Supports filtering by module,
category, and pytest markers. See also: *conftest.py*.

**uv**: The Python package manager used by GEO-INFER. Manages the root workspace
(`pyproject.toml`, `uv.lock`, `.venv/`); `uv sync --all-packages --all-extras`
installs every module. Install via
`curl -LsSf https://astral.sh/uv/install.sh | sh`.
See also: *pyproject.toml*.

---

## Abbreviations Table

| Abbreviation | Full Form |
|-------------|-----------|
| ACT | Active Inference (module) |
| AG | Agriculture (module) |
| AI | Artificial Intelligence (module) |
| AIC | Akaike Information Criterion |
| ANT | Ant Colony Optimization (module) |
| API | Application Programming Interface |
| APP | Application (module) |
| ART | Art (module) |
| BIC | Bayesian Information Criterion |
| BIO | Biology / Ecology (module) |
| CIV | Civic / Urban Planning (module) |
| CLI | Command Line Interface |
| COG | Cognitive (module) |
| COG | Cloud Optimized GeoTIFF (context-dependent) |
| COMMS | Communications (module) |
| CRS | Coordinate Reference System |
| DIC | Deviance Information Criterion |
| ECON | Economics (module) |
| EDU | Education (module) |
| EFE | Expected Free Energy |
| ELBO | Evidence Lower Bound |
| EPSG | European Petroleum Survey Group (geodetic codes) |
| FEP | Free Energy Principle |
| GIS | Geographic Information System |
| GP | Gaussian Process |
| H3 | Hierarchical Hexagonal spatial index (version 4) |
| HMC | Hamiltonian Monte Carlo |
| IOT | Internet of Things (module) |
| KL | Kullback-Leibler (divergence) |
| LOG | Logistics (module) |
| LOO | Leave-One-Out (cross-validation) |
| MATH | Mathematics (module) |
| MCMC | Markov Chain Monte Carlo |
| MDP | Markov Decision Process |
| NDVI | Normalized Difference Vegetation Index |
| OGC | Open Geospatial Consortium |
| OPS | Operations (module) |
| ORG | Organizational (module) |
| PEP | People (module) |
| REQ | Requirements (module) |
| RISK | Risk Assessment (module) |
| SEC | Security (module) |
| SIM | Simulation (module) |
| SMC | Sequential Monte Carlo |
| SPM | Statistical Parametric Mapping (module) |
| SRAI | Spatial Representations for Artificial Intelligence |
| STAC | SpatioTemporal Asset Catalog |
| TDD | Test-Driven Development |
| UTM | Universal Transverse Mercator |
| VFE | Variational Free Energy |
| WAIC | Widely Applicable Information Criterion |
| WFS | Web Feature Service |
| WGS84 | World Geodetic System 1984 |
| WKT | Well-Known Text |
| WMS | Web Map Service |

---

## Related Documentation

- [Active Inference Guide](active_inference_guide.md) -- mathematical details
- [Geospatial Standards](geospatial_standards.md) -- H3, CRS, format specs
- [Data Dictionary](data_dictionary.md) -- data structure definitions
- [Overview](overview.md) -- module descriptions and architecture
