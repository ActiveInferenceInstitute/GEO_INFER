# System Context {#sec:system_context}

## Project Boundary

GEO-INFER is a `{{MODULE_COUNT}}`-module geospatial inference framework for
spatial analysis, active inference, domain modeling, agent workflows, and
repository validation. The current checkout reports
`{{MODULES_WITH_TESTS_COUNT}}` modules with test files and
`{{DOCUMENTATION_PAGE_COUNT}}` cross-module documentation pages under
`GEO-INFER-INTRA/docs/`. H3-backed spatial indexing is treated as a
hierarchical geospatial evidence surface rather than a claim of exact
geometric containment [@h3_docs].

## Module Inventory

A module is a top-level directory named `GEO-INFER-<NAME>` that ships one
installable package under `src/`, its own `tests/` tree, and its own
`README.md` and `AGENTS.md`. The inventory discovers these source directories;
discovery is distinct from integration. A new module also needs a declared
theme, packaging and dependency configuration, an agent catalog entry, and
any adapters required by its consumers. The measured inventory therefore
establishes presence, not automatic interoperability.

The generator groups the measured modules into `{{MODULE_THEME_COUNT}}`
themes. The grouping is an editorial one, declared in the generator beside the
research-focus list and mirroring the module-theme table in `README.md`; it is
not derived from imports or from a dependency graph. It is nonetheless
enforced: the build refuses to run if a measured module is unthemed, if a
theme names a module that is not in the checkout, or if a module is declared
twice, so a module cannot silently disappear from this table.

{{MODULE_TABLE}}

: The measured module inventory at commit `{{RESEARCH_COMMIT}}`, grouped by the
declared themes the generator enforces. Each row names one module, the
installable package directory it ships under `src/`, and its counted Python
source and test files; the counts are measured by the same inventory pass that
produces every other quantity in this manuscript, and the build refuses to
render if any measured module is unthemed, absent, or declared twice. Read the
table as a map of where implementation and verification surface live — the
largest module is `GEO-INFER-SPACE`, and several modules carry more test files
than source files — not as a statement about correctness or maturity.
{#tbl:module_inventory}

The themes divide the framework by the kind of claim each module is
responsible for. *Spatial and place-based* modules own geometry, indexing,
time, and the environmental domains layered on them. *Bayesian and active
inference* modules own the probabilistic machinery: numerical and statistical
primitives in `GEO-INFER-MATH`, variational and expected free energy in
`GEO-INFER-ACT`, samplers and posterior diagnostics in `GEO-INFER-BAYES`,
statistical parametric mapping in `GEO-INFER-SPM`, cognitive modeling in
`GEO-INFER-COG`, and scenario simulation in `GEO-INFER-SIM`. *Agents and AI
orchestration* modules turn those methods into acting systems — autonomous and
swarm agents, machine-learning services, operations orchestration, and
messaging. *Governance, risk and domain* modules apply them to decisions with
consequences: hazard and risk modeling, insurance underwriting, economics,
norms and compliance, security, and health. *Data, API and applications*
modules own ingest, storage, service interfaces, and the human-facing
surfaces. *Infrastructure and validation* modules own the repository's own
evidence: the unified test framework in `GEO-INFER-TEST`, cross-module
documentation in `GEO-INFER-INTRA`, worked integration examples in
`GEO-INFER-EXAMPLES`, and repository tooling.

Theme membership is a statement about intent, not about size or maturity.
[@tbl:module_inventory] shows the spread directly: the largest module in the
checkout is `GEO-INFER-SPACE`, and several modules carry more test files than
source files.

## Theme Map

[@tbl:theme_map] states the declared grouping compactly. Short names omit
the shared `GEO-INFER-` prefix, exactly as the generator's figures do.

| Theme | Modules | Responsibility |
|---|---|---|
| Spatial and place-based | `SPACE`, `PLACE`, `TIME`, `MARINE`, `WATER`, `FOREST`, `CLIMATE`, `ENERGY`, `TRANSPORT`, `EMERGENCY` | Geometry, indexing, time, and the environmental and infrastructural domains layered on them |
| Bayesian and active inference | `BAYES`, `SIM`, `SPM`, `COG`, `ACT`, `MATH` | The probabilistic machinery: numerical primitives, variational and expected free energy, samplers and diagnostics, cognitive modeling, simulation |
| Agents and AI orchestration | `AGENT`, `AG`, `AI`, `ANT`, `OPS`, `COMMS` | Autonomous and swarm agents, machine-learning services, operations orchestration, and messaging |
| Governance, risk and domain | `INSURANCE`, `RISK`, `METAGOV`, `NORMS`, `ECON`, `PEP`, `REQ`, `SEC`, `CIV`, `HEALTH`, `ORG` | Decisions with consequences: hazard and risk, insurance, economics, norms and compliance, security, health, and organizational structure |
| Data, API and applications | `API`, `APP`, `DATA`, `IOT`, `ART`, `EDU` | Ingest, storage, service interfaces, and the human-facing surfaces |
| Infrastructure and validation | `INTRA`, `TEST`, `LOG`, `GIT`, `EXAMPLES`, `BIO` | The repository's own evidence: unified tests, cross-module documentation, repository tooling, and worked examples |

: The declared theme map, mirroring the generator's enforced grouping. Which
theme a module belongs to is an editorial declaration enforced by the build:
a measured module that is unthemed, absent, or declared twice fails the
build. The table carries no counts — per-module measured surfaces are in
[@tbl:module_inventory], and per-module prose entries are in the
Supplemental Module Catalog ([@sec:module_catalog]).
{#tbl:theme_map}

## Spatial Substrate

The framework's spatial substrate supplies indexing, place, and time contracts
through three modules. The theme map does not establish an import dependency
from every module to each member of this substrate.

`GEO-INFER-SPACE` owns indexing and geometry. Its
`SpatialIndexingInterface` is a backend-dispatching facade rather than a
direct H3 binding, so H3 is one implementation of a spatial index and not a
hard dependency of the framework's spatial API. H3 resolution choice is itself
a guarded operation: `geo_infer_space.core.h3_policy` estimates cell counts
from the published average cell area at each resolution 0 through 15, selects
the finest resolution whose estimated grid stays within a soft target, and
raises `H3HardCapExceededError` rather than allocating a grid beyond a hard
cell cap. The policy module deliberately imports no H3 runtime, so the guard
remains importable in an environment where the H3 backend is absent. Above the
flat grid, `geo_infer_space.nested` maintains parent-child hexagon hierarchies
with boundary detection, message passing across boundaries, and lumping,
splitting, and aggregation operations — the structure that lets an inference
run at one resolution be related to a coarser or finer one.

`GEO-INFER-PLACE` binds analyses to named places. `PlaceInterface` is a single
entry point that orchestrates location-specific analyzers, data acquisition,
and temporal analysis for a configured region, so a place-scoped study is a
configuration of shared machinery rather than a fork of it.

`GEO-INFER-TIME` owns the temporal axis under the same discipline. Its
`inference_schedule` validates that a sequence of timestamps is timezone-aware,
ordered, and exactly aligned to the declared model step, and raises on a
missing interval instead of interpolating one: a gap in observations is
treated as something the model must be asked to predict, not something the
scheduler may fill. Offsets are normalized to UTC before instants are
compared.

These checks address representational alignment. A cell identifier and UTC
timestamp do not specify whether a value is a point measurement, an area
average, a population count, or an estimate over a time interval. Regridding
and aggregation can change the estimand and its uncertainty
[@gotway_spatial_support_2002]. In particular, H3's exact logical hierarchy
must be distinguished from approximate geometric parent-child containment
[@h3_docs]. A study must declare spatial and temporal support, units, and
aggregation assumptions in addition to satisfying the software contract.

## Source Surfaces

- `GEO-INFER-*/src/` and `GEO-INFER-*/tests/`
- `GEO-INFER-TEST/`
- `GEO-INFER-INTRA/docs/`
- `README.md`
- `AGENTS.md`
- `pyproject.toml`
- `uv.lock`

The source surface contains `{{SOURCE_FILE_COUNT}}` Python files and
`{{SOURCE_LINE_COUNT}}` non-empty source lines. The three research-focus
modules contain `{{ACT_SOURCE_FILE_COUNT}}`,
`{{BAYES_SOURCE_FILE_COUNT}}`, and
`{{RISK_SOURCE_FILE_COUNT}}` source files respectively, with corresponding
test surfaces of `{{ACT_TEST_FILE_COUNT}}`, `{{BAYES_TEST_FILE_COUNT}}`, and
`{{RISK_TEST_FILE_COUNT}}` files.

## Template Boundary

The manuscript is rendered locally from `output/manuscript/` after the project
pipeline resolves its variables. The tracked `manuscript/` directory remains
the authored source; `output/` contains disposable figures, JSON evidence, and
resolved copies generated from this checkout.
