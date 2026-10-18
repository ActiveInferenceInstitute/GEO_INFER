# Methods {#sec:methods}

## Research Construction Method

The manuscript pipeline performs a deterministic source-first pass. It
discovers every `GEO-INFER-*` directory containing `src/`, measures Python
source and test surfaces, reads the project version from `pyproject.toml`, and
records the current Git commit, branch, commit date, working-tree dirtiness,
and source fingerprint. The inventory is written to
`output/data/research_inventory.json` before any prose is resolved.

The research focus is the composable path from Active Inference through
Bayesian inference to risk analysis [@friston_free_energy_2010;
@parr_active_inference_2022; @heins_pymdp_2022]. The pipeline does not
synthesize results or infer performance from file counts; it reports
implementation and evidence surfaces separately, then delegates behavioral
claims to the repository's validators and test suites.

## Composition Contract

The composition contract is the set of interface obligations that let the
three spine modules be developed and tested independently and still be run in
sequence. It is authored, not inferred: the obligations below are recorded in
`GEO-INFER-INTRA/docs/research_grade_inference_contracts.md`, and the module
source, tests, and strict validators remain authoritative where that page
drifts. The contract is auditable in the sense that every clause names a
symbol in the checkout that can be read, called, and tested.

### Typed results

The ACT part of the contract exposes typed return objects, so a caller can
inspect declared fields before adapting them to another module's inputs.
`geo_infer_act.core.types` declares three of them.
`FreeEnergyBreakdown` carries a scalar free energy together with its
`accuracy`, `complexity`, `entropy`, `pragmatic_value`, `epistemic_value`,
`risk`, and `ambiguity` components. `PolicyEvaluation` carries a policy, its
expected free energy, its selection probability and index, and the same
component decomposition. `ActiveInferenceStepResult` carries the beliefs, the
chosen action, the realized free energy, the optional expected free energy,
and the `PolicyEvaluation` that produced the action. Each exposes `to_dict()`,
which converts NumPy scalars, arrays, and nested typed results into JSON-safe
values, so a diagnostic can be serialized into an evidence record without a
bespoke encoder.

### Active Inference

For a categorical model with belief $q(s)$, reference state distribution
$r(s)$, and a coerced observation likelihood $\hat{p}(o \mid s)$,
`FreeEnergyCalculator.compute_categorical_free_energy()` computes complexity
minus accuracy in \eqref{eq:free_energy},

\begin{equation}\label{eq:free_energy}
F = \sum_{s} q(s)\,\bigl[\log q(s) - \log r(s)\bigr] \;-\; \sum_{s} q(s)\,\log \hat{p}(o \mid s),
\end{equation}

and reports the belief entropy $-\sum_s q(s) \log q(s)$ alongside it. Terms
are computed with a small additive constant inside each logarithm for
numerical stability, and the vectors are normalized before use. The current
API populates $r(s)$ from its argument named `preferences`, or uses a uniform
reference when none is supplied. In the usual variational interpretation,
$r(s)$ would represent a generative state prior; a desired outcome
distribution has a different modeling role [@parr_active_inference_2022].
The API's argument name does not establish that those roles are scientifically
interchangeable. This equation describes the local categorical calculator;
factored and external-backend paths must be interpreted using their own
generative-model contracts.

The local `compute_policy_expected_free_energy()` helper implements a
configurable policy score. For its preference-bearing branch, let $c(s)$ be
the normalized preference vector aligned to the predictive state vector
$\tilde{q}$, with an expected posterior $\tilde{q}^{+}$ where supplied,
a temporal discount $\gamma$, and an exploration weight $\beta$:

\begin{equation}\label{eq:expected_free_energy}
G(\pi) = \gamma \Bigl(-\sum_{s} \tilde{q}(s) \log c(s)\Bigr) \;-\; \beta\, D_{\mathrm{KL}}\bigl(\tilde{q}^{+} \,\|\, \tilde{q}\bigr) \;+\; \text{risk} \;+\; \text{ambiguity},
\end{equation}

where the risk term is a declared risk preference scaled by the variance of
the predictive distribution and the ambiguity term is supplied by the policy.
When no expected posterior is supplied, predictive entropy replaces the KL
term as an exploration proxy. Without configured preferences, the pragmatic
term is uniform-reference cross-entropy when an expected observation is
supplied, and zero otherwise. A policy may instead supply a finite score
directly; the returned metadata identifies that case and does not reconstruct
an unavailable decomposition. Computed terms are exposed in
`FreeEnergyBreakdown`. This operational convention, including its
variance-based risk and entropy proxy, is not asserted to be a general
derivation of expected free energy or identical to the external `pymdp`
backend [@heins_pymdp_2022].

Three further obligations bind the module:

- `ActiveInferenceModel.act()` accepts scalar or sequence-valued control
  configuration and validates action availability before selecting a policy.
- Categorical H3 inference uses the real `inferactively-pymdp` adapter when its
  matrix contract is present. A local fallback is opt-in through
  `allow_local_pymdp_fallback`, and a failure of the optional backend is never
  reported as a backend result. The same rule governs the optional
  RxInfer and Bayeux integrations: they either return posterior data produced
  by the external backend or use an explicitly labelled local sampler.
- Read-only H3 grid inference snapshots and restores the policy-selector RNG,
  so running a diagnostic does not change a later stochastic decision. H3
  observations, transition matrices, and belief vectors are shape-checked
  before Bayesian updates, and nested H3 results retain typed cell, level, and
  parent-child diagnostics.

### Bayesian inference

`GEO-INFER-BAYES` supplies the posterior machinery the spine consumes
[@gelman_bda_2013]. Its obligations are about reproducibility and about
refusing to report a diagnostic that is not defined:

- ABC, SMC, MCMC, HMC, variational inference, and spatio-temporal Gaussian
  process sampling use instance-local NumPy generators. A caller's global RNG
  state is not part of the sampler contract.
- MCMC and HMC flatten scalar and array-valued parameters internally and
  restore the declared parameter shapes in returned draws. The HMC no-U-turn
  path performs recursive tree construction with a genuine no-U-turn check; it
  is not an alias for a fixed-step sampler.
- Variational inference accepts vector mean-field parameters and preserves the
  caller's initial variational parameters during updates. Full-rank covariance
  currently carries an explicit scalar-parameter boundary.
- Diagnostics report effective sample size, Monte Carlo standard error,
  $\hat{R}$, and Geweke statistics. $\hat{R}$ is reported as `NaN` for a single
  chain, because between-chain convergence is undefined there; it is not
  replaced with a misleading scalar.
- `PosteriorAnalysis.credible_interval()` returns the equal-tailed interval
  between the $100\alpha/2$ and $100(1-\alpha/2)$ percentiles of the posterior
  draws. It requires $0 < \alpha < 1$ and rejects missing, empty, or
  non-finite draws.
- `SpatioTemporalGP.predict()` accepts only an $(n, 3)$ matrix of $x$, $y$,
  and time; split spatial/temporal arguments are rejected. Posterior
  predictions fit a concrete spatial GP for each draw without mutating the
  fitted model.

These diagnostics summarize the supplied draws. Finite effective sample size
or a between-chain statistic does not establish identifiability, posterior
calibration, or suitability of a likelihood. A study still needs checks of
sampling behavior, prior sensitivity, and predictions against its stated
observational target [@gelman_bda_2013]. The manuscript does not attribute a
particular modern diagnostic estimator to a method merely because it returns
a field named effective sample size or $\hat{R}$.

### Risk modeling

`GEO-INFER-RISK` is where the spine's output becomes a decision quantity, and
its obligations are correspondingly about declared bounds
[@coles_extremes_2001]:

- `RiskParameters` validates its numerical contract at construction: the
  confidence level must be finite and in $(0, 1)$, the time horizon and Monte
  Carlo iteration count must be positive integers, the spatial resolution must
  be finite and positive, and the seed is resolved eagerly so a bad seed
  surfaces at construction rather than at the first draw. Component outputs
  must be aligned, finite, non-negative, and within their declared probability
  bounds.
- Point-estimate uncertainty bounds come from declared component bounds. When
  no component uncertainty is declared, the lower and upper bounds equal the
  point estimate and the result identifies that source explicitly; arbitrary
  fixed percentage bands are not used.
- Monte Carlo runs use a local seeded generator and expose the configured
  percentile keys alongside the established result names. Hazard event
  timestamps are reproducible for seeded runs.
- Return-period intensities require a fitted, calibrated model. The Gumbel
  return level is derived by moment matching from the fitted mean $\bar{x}$
  and standard deviation $s$ of the intensity sample: the scale is
  $\sigma = s\sqrt{6}/\pi$, the location is $\mu = \bar{x} - \gamma_{E}\sigma$
  with $\gamma_{E}$ the Euler-Mascheroni constant, and the $T$-period return
  level is
  $z_{T} = \mu + \sigma\bigl(-\log(-\log(1 - 1/T))\bigr)$.
- `EnhancedRiskEngine` validates configuration before creating output
  resources, checks available integration APIs, and supports `close()` plus
  context-manager lifecycle for its executor and file handler. RiskEngine
  calibration rejects unsupported maximum-likelihood and underspecified
  Bayesian requests explicitly; only validated cross-validation inputs produce
  calibration results.

Interpreting a return period in years requires observations with the declared
annual support and a defensible extreme-value sampling model. The algebraic
quantile alone does not establish stationarity, independence, tail fit, or
uncertainty in extrapolation [@coles_extremes_2001]. Likewise, a model's
internal fitted/calibrated state is an execution prerequisite rather than
evidence of calibration against independent field data.

### What the contract does and does not establish

The contract constrains interfaces, provenance, and refusal behavior. It does
not establish that any model in the checkout is well-calibrated against field
data, and it makes no claim about predictive skill. Its purpose is narrower
and checkable: a value crossing a module boundary carries a declared shape, a
declared source, and declared bounds, and a module that cannot honor one of
those should fail explicitly within the checked path. These obligations are
scoped to the named contracts, not a claim that every method in the framework
checks every scientific assumption.

### Scientific obligations across a boundary

The following review questions accompany a composed study. They distinguish
software checks already named above from scientific decisions that cannot be
inferred from serialization or successful execution.

| Boundary | Software evidence | Study-level obligation |
|---|---|---|
| Observation to spatial state | H3 cell order, shape, duplicate and allocation checks; explicit missing values | Declare coordinate reference, units, observation error, spatial support, and aggregation weights. |
| Observation to time step | UTC alignment and declared scheduling interval | Distinguish an instant from an interval total or mean; justify handling of gaps and temporal dependence. |
| State to inference | Model dimensions, typed components, explicit backend identity | Specify state and observation meanings, priors, likelihoods, preference interpretation, and approximation assumptions. |
| Posterior to risk | Finite draws, declared parameter shapes and component bounds | Preserve dependence and distinguish parameter, observation, and decision uncertainty when propagating draws. |
| Risk to decision | Declared horizons, probability bounds and return-period inputs | Justify exposure, vulnerability, loss units, decision costs, and external calibration. |

: Review obligations for a composed analysis. The software column names
inspectable contracts; the study column is not represented as an automated
guarantee. Shared indexing does not solve the statistical change-of-support
problem [@gotway_spatial_support_2002]. {#tbl:composition_obligations}

## Script, CLI, and Agent Composition

`geo_infer_api.AgentSession` is a common execution layer for Python scripts,
the persistent JSONL CLI, and the stdio Model Context Protocol server.
`agent_catalog.py` discovers public definitions without importing each
scientific backend, while describing or calling a target resolves its owning
module in the active environment. Catalog presence therefore establishes
discoverability, not availability of every optional dependency.

The session validates operation fields and selectors before dispatch. JSON
requests can construct scientific values through declared codecs and pass
native results onward through process-local references. For example, a
completed OPS task's `get_task_result()` retains its native value even though
the workflow's historical status summary renders results as strings. Typed
codecs and native handles prevent an array's shape or an object's identity
from being reduced to its printed representation during composition.

Lifecycle is explicit: retained references belong to one live session, context
operations enter and exit native context managers, and releasing a reference
does not imply calling an arbitrary object's cleanup method. Calls execute
with the authority of the local Python process. A batch stops on failure but
does not roll back preceding effects. These are execution interfaces, not a
sandbox, a scientific workflow specification, or a guarantee of recovery from
partial external effects.

## Performance Measurement Boundary

Execution profiling and timing experiments answer different questions. Python
3.15's `profiling.tracing` records deterministic call events;
`profiling.sampling` estimates stack occupancy. Instrumentation changes the
execution being measured, and sampled counts are not exact call counts
[@python315_profiling_docs]. A comparative speed claim requires a separate
timing design with a declared workload, warmup, repetitions, environment,
cache state, and uncertainty; profile durations alone do not establish a
speedup.

Repository profile receipts should identify the interpreter, profiler,
command, source fingerprint, dependency environment, and produced artifacts.
A successful Python 3.15 profile qualifies only the named workload in that
environment. It does not establish that the full scientific dependency stack
or every module is supported on Python 3.15. The asynchronous timing helpers
in `geo_infer_test.core.performance_monitor` await work to completion;
constructing a coroutine is not counted as completing an inference task.

## Figure and Caption Method

The figure producer builds `{{FIGURE_COUNT}}` figures from the same measured
inventory used for the manuscript variables. PNG figures, SVG companions,
JSON and CSV datasets, captions, labels, generator paths, commit, source hash,
and content digests are recorded in the figure registry, `figure_registry.json`, under
`output/figures`. Each caption is a complete sentence that names the
population, encoding, provenance, and interpretation boundary. Figures are
drawn at the printed size of the render's text block, so the type in a
figure is set at the point size it is authored at rather than being scaled
down by the renderer. This medium-aware layout and explicit captioning follow
scientific figure-design guidance [@rougier_figures_2014]. The companion data
expose the values and encodings used to draw each figure; they are checked
against recorded hashes along with the rendered representations.

The first figure, the graphical abstract, is additionally wired into the
title page: `config.yaml` points the renderer's cover slot at the same
generated PNG, so the cover page and the Abstract section carry one artifact
whose panel content and caption come from the declared themes, the Methods
section, and the published verification record rather than from hand-drawn
artwork.

![{{MODULE_INVENTORY_CAPTION}}](../output/figures/module_inventory.png){#fig:module_inventory width=100%}

![{{RESEARCH_SPINE_CAPTION}}](../output/figures/research_spine.png){#fig:research_spine width=100%}

![{{VALIDATION_SURFACE_CAPTION}}](../output/figures/validation_surface.png){#fig:validation_surface width=100%}

## Module Catalog Method

The per-module entries of the Supplemental Module Catalog
([@sec:module_catalog]) are authored, one file per module under
`manuscript/sections/`, named by the module's lowercase slug. Each entry is
verified against three surfaces rather than written from memory: the public
interface is read from package exports in `__init__.py` and, where imports are
submodule-specific, the owning implementation. Verification surfaces are checked
against the module's test tree, and the theme role restates the declared grouping
the generator enforces.
Entries are plain prose — one `##` heading naming the module, then
paragraphs — and carry no tokens, labels, figures, or raw LaTeX, because
the render combines them from the tracked source after the generator's
token substitution has run; a quantity that belongs in the catalog belongs
in its generated index table instead. Authored entries do not duplicate source
or test counts, rank suite quality by file counts, or treat local software
fixtures as field validation.

The render wiring is deliberate and fail-closed: `S02_module_catalog.md`
is a published section whose position in the section order is fixed in
`scripts/render_manuscript_pdf.py`, and the combine step appends every
`manuscript/sections/*.md` in alphabetical order after it, refusing to
render if the catalog has no entries. The catalog's authoritative module
set stays the generator's measured inventory — the index table is generated
from it — so a prose entry can elaborate a module but cannot add one the
build does not measure.

## Evidence Promotion and Verification

Research claims are promoted in this order:

1. Discover the source surface and write a provenance-bearing inventory.
2. Generate figures and captions from measured data, failing if a declared
   figure is absent, unregistered, or larger than the printable box.
3. Run the strict repository, documentation, skills, model, reproducibility,
   and source checks; optionally run the full unit, integration, performance,
   and H3 suites.
4. Inject every volatile value into `output/manuscript/` and reject unresolved
   uppercase tokens.
5. Review the resolved manuscript and registry together before publication.

## Template Compliance

Each publication section has one H1 label, every figure uses labeled Pandoc
image syntax, captions are sourced from the figure registry, and generated
copies contain no unresolved variable tokens.
