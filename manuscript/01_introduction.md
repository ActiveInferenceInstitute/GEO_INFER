# Introduction {#sec:introduction}

## Motivation

Geospatial inference combines observations, spatial representations,
probabilistic models, and decision rules developed under different assumptions.
A shared array shape or spatial identifier makes data exchange possible, but
does not resolve differences in measurement scale, spatial support, or the
population to which an inference applies [@gotway_spatial_support_2002]. A
modular framework therefore needs an inspectable account of both its software
interfaces and the scientific obligations left to the study designer.
GEO-INFER addresses the software part by keeping spatial and probabilistic
methods in owned modules while linking tests, manifests, generated figures,
and documentation to the checkout that produced them.

The framework-level claim behind that discipline is architectural: geospatial
inference can be composed through explicit interfaces. A spatial
index, a variational free-energy calculation, and a hazard return level are
useful separately, but the framework's reason to exist is that a value can
cross from one to the next through a declared adapter: a place-scoped
observation entering an active inference step, or posterior draws being
prepared as inputs to a risk model. These examples describe an architectural
path, not a universally valid conversion between scientific quantities. Typed
results, declared shapes, seeded generators, and explicit refusal behavior
make the crossings inspectable; the choice of likelihood, aggregation rule,
and decision objective still requires scientific justification. This
manuscript is the account of those boundaries, published from the same
checkout that implements them.

## Architecture at a Glance

The module set composes in layers that the theme map in
[@sec:system_context] makes explicit. The *spatial substrate* — indexing,
nested hierarchies, named places, and the temporal axis — owns geometry and
provides the representations used by spatial workflows. The *probabilistic machinery* —
Bayesian inference, active inference, statistical parametric mapping,
cognitive modeling, and simulation — owns the inference quantities that
consume spatial evidence. The *domain engines* — agents and orchestration,
governance and risk, and the applied sciences — turn those quantities into
decisions and domain claims. *Data, API and application* modules carry the
ingest, storage, service, and human-facing surfaces, and *infrastructure and
validation* modules own the repository's evidence loop: the unified test
framework, cross-module documentation, and repository tooling.

Three design rules organize the layers. First, each module owns its package,
tests, and public contract. Directory discovery supplies the measured
inventory, while declared themes, packaged agent catalogs, dependencies, and
cross-module adapters must be updated when a module is integrated. Second, the research
spine — Active Inference through Bayesian inference to risk — is one
composable path through the module set, stated as a contract in
[@sec:methods] rather than implied by adjacency. Third, every claim the
framework makes about itself is bound to a measured surface: the per-module
catalog in [@sec:module_catalog] enumerates what each module exposes and
verifies, and the evidence sections report what was actually run.

## Contributions

This manuscript makes five concrete contributions:

1. A measured inventory of `{{MODULE_COUNT}}` modules, `{{SOURCE_FILE_COUNT}}`
   source files, and `{{TEST_FILE_COUNT}}` test files at source fingerprint
   `{{RESEARCH_SOURCE_HASH}}`, published per module and per theme rather than
   as a single total.
2. A composable research spine connecting Active Inference, Bayesian
   inference, and risk analysis through independently testable module
   boundaries, with the interface obligations at those boundaries stated
   explicitly rather than asserted.
3. A reproducible artifact pipeline that generates `{{FIGURE_COUNT}}`
   publication figures, registers captions, generator paths, and a per-figure
   content digest, and injects all volatile values into resolved manuscript
   copies.
4. A fail-closed evidence record that distinguishes available source surfaces
   from commands that were actually executed and passed, and a publication
   mode that fails rather than certify a build whose record is empty or
   contains a failed group.
5. A per-module catalog that enumerates every module's purpose, public
   interface, and test surface in one place, indexed by the generated module
   table and organized by the declared themes, so the module contract can be
   read module by module rather than inferred from the tree.

## Scholarly Context

The active-inference spine draws on a general probabilistic account of
perception and action [@friston_free_energy_2010; @parr_active_inference_2022]
and an existing discrete-state software implementation [@heins_pymdp_2022].
The contribution assessed here is the repository's composition and evidence
contract; citing those works does not establish that every local scoring
convention is equivalent to their formulations.

Recording software versions, commands, parameters, and intermediate artifacts
follows established guidance for reproducible computation
[@sandve_reproducible_2013]. Machine-readable metadata and qualified links
between artifacts also address concerns articulated by the FAIR principles
[@wilkinson_fair_2016]. These are design motivations rather than a FAIR
certification: local hashes and JSON records do not by themselves provide
persistent public identifiers, searchable deposition, or long-term access.
Successful software checks likewise provide bounded evidence about execution;
the adequacy of an earth-science model remains conditional on observations
and assumptions [@oreskes_model_validation_1994].

## Reader Orientation

The manuscript should be read with the repository commit and source hash in
view. The module inventory and validation surface describe what is present in
the checkout; they do not substitute for domain-specific empirical validation.
The measured surface of every module appears in [@tbl:module_inventory], the
interface obligations that make those modules composable appear in the
Composition Contract, the implementation and verification surfaces for the
Active Inference, Bayesian, and RISK modules are summarized in
[@fig:research_spine], and the Supplemental Module Catalog expands the same
inventory into one prose entry per module.
