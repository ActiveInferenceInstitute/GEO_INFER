# ACT 0.4.0 method review and interface migration

The source inventory covers all 48 Python files in `src/geo_infer_act`, including
657 functions and methods, internal helpers, asynchronous methods, and runner
orchestration. Definitions and callers were reviewed together with configuration,
tests, public exports, and documentation. This scope count is an inventory of
reviewed source, not a claim that every branch has an independent numerical oracle.

The proof levels remain separate:

- **Source review:** definitions, call sites, shapes, ownership, configuration use,
  failure reporting, optional imports, and orchestration boundaries.
- **Analytical regression:** independently calculated posterior, probability,
  Gaussian KL/measurement likelihood, continuous-time derivative/filter, resource
  transport, and real pentagon/topology expectations in
  [test_method_scan_contracts.py](../tests/unit/test_method_scan_contracts.py).
- **Module acceptance:** canonical unit, slow, and integration selections retain
  collected and executed node inventories plus fresh JUnit. The module has no
  dedicated performance test roots; an empty ACT performance selection is not a
  performance pass. Repository performance gates run their registered roots.
- **Backend execution:** [test_bayeux_backend.py](../tests/integration/test_bayeux_backend.py)
  runs real NumPyro NUTS through Bayeux on a scalar Gaussian target, with 100 warmup
  iterations and 100 retained draws. The Monte Carlo tolerances verify this small
  target; they do not certify all densities or convergence.
  A separate cold subprocess uses a new bytecode lookup prefix, strict warnings,
  an external temporary working directory, and both public interfaces. The owning
  import helper covers only the exact jaxopt 0.8.3 OSQP source hash and its known
  invalid `\mu` docstring escape at the CPython 3.11/3.12 compiler locations.
  Other compile/runtime warnings and backend import failures retain caller policy;
  changed dependency source is outside this exception. The helper neither retries
  an interrupted import nor replaces the declared backend.
- **Artifact acceptance:** `verify_comprehensive.py` executes ten representative
  method, model, runner, visualization, API, and documentation sections. It does
  not exhaust every callable in the inventory.

Julia/RxInfer execution, GPU/multi-device acceleration, an externally deployed
ACT HTTP model service, and arbitrary caller-supplied PyMC/Pyro source models
remain outside this local acceptance evidence. Availability discovery alone
never establishes backend behavior. `local_gaussian` is a separate conjugate
model and carries that backend label.

## Configuration and model contracts

Flat categorical matrices use A `(observation, state)` and B
`(next_state, current_state[, action])`. A valid supplied zero remains zero.
Matrix, prior, observation, preference, and control dimensions must agree;
normalization cannot replace a supplied matrix, prior, or control axis. Explicit
positive observation weights retain the frequency semantics of the pymdp adapter.
Empty, nonfinite, negative, or mismatched observation vectors fail before inference.
A missing observation requires a separate explicit prediction interval.

`ActiveInferenceModel.step` conditions one observation and evaluates prospective
policies. `predict_beliefs(action_index)` advances a flat posterior once using the
executed action; action-independent B uses `predict_beliefs()`. The caller owns
an interval/action history when composing these methods. Returned beliefs and
public snapshots own their data, and invalid outcomes do not append history.

H3 `GenerativeModel` scoring preserves the configured **states per cell** in
`state_dim`, along with the original A/B matrix dimensions. `h3_cells` is the
separate spatial axis. `update_h3_beliefs` takes identified observations;
unlabelled `update_beliefs` arrays on an H3-enabled model are rejected. Per-cell
scoring/diffusion is a different declared model from the global SPACE/TIME→ACT
state-space composition, whose H3 cells are actual global states with explicit
sparse transitions. Cell aggregation computes equal-cell posterior means;
refinement, invalid identifiers, mixed input resolutions, and incompatible
belief dimensions are errors. Diffusion convexly blends observed neighbors,
keeps absent observations absent, and accepts only rates in [0, 1]. H3 update
`aggregate_free_energy` is the mean actual observed-cell perception VFE;
nested updates retain that measurement diagnostic before spatial blending.
Nested level `mean_reference_kl_to_uniform` describes the aggregate posterior's
uniform-reference KL, since parent cells have no independent observed likelihood.
Read-only grid queries restore joint priors and Gaussian measurement diagnostics
even when a cell fails. The legacy polygon integration helper's `max_cells`
limits accepted output cardinality after native H3 coverage; it does not bound
that native polygon allocation.

Research profiles specify A/B/C and initial beliefs; they do not guarantee
multiple selected actions for arbitrary observations. A nearby environmental
sample can share one analytically optimal policy. Diversity acceptance uses a
declared categorical fixture covering three known policy regions, checked with
independent posterior, negative-EFE, softmax and selection calculations. Gallery
outputs separately verify the scores, softmax and selection against the recorded
policy input. Retained `policy_beliefs` metadata identifies the local posterior used
for policy evaluation before later spatial or nested belief blending.

Climate and ecological models consume explicit integer modality indices through
the joint factored runtime. The correlated joint posterior is retained rather
than reconstructed from marginal products, policy temperature and zero policy
priors are effective, and selected joint controls produce the next prior.
Factored summaries report entropy of the correlated joint, including ragged
factor dimensions. Their standalone `free_energy` is explicitly labelled
`reference_kl_to_uniform_joint`; the active agent's observed VFE is separate.
Urban `planning_horizon` is used by bounded exact evaluation of default stationary
policies; its default is two. Callers can supply explicit factored policy
sequences when controls must change within the horizon.

Gaussian VFE uses KL(q||p) plus the expected normalized measurement likelihood,
including the configured rectangular observation matrix and covariance. Core
`ActiveInferenceModel.perceive` exposes the generative model's actual retained
measurement VFE. Generic Gaussian `act` has no declared control dynamics and
raises explicitly; use
[ContinuousPOMDPActiveInference](../src/geo_infer_act/models/continuous_pomdp.py)
or the [Gaussian GNN runner](gnn_interchange.md#discrete-time-linear-gaussian-v2)
for control evaluation. DCM A/B parameters describe continuous-time derivatives;
parameter fitting and filtering use the actual increasing model-step axis.

## Removed implicit behavior

- `GenerativeModel` defaults `obs_dim` to the declared `state_dim`; pass an
  explicit `obs_dim` and A for a rectangular observation model.
- H3 enablement no longer multiplies `state_dim` while leaving A/B unchanged.
  Inspect `h3_cells` for the cell count and use identified observation mappings.
- `NestedH3LevelSummary.mean_free_energy` was renamed to
  `mean_reference_kl_to_uniform`, including its serialized field. H3 update
  aggregate VFE now comes from actual measurements rather than that reference KL.
- Pymdp adapters no longer substitute identity likelihood/transitions, truncate or
  cycle a control axis, resize preferences, or replace a mismatched prior.
- API `update_beliefs` requires `{"observations": ordered_vector}`; numeric
  dictionary keys are not sorted into an invented observation order. Model IDs
  are nonempty and unique, and creating an existing ID is an error.
- `integrate_rxinfer` requires explicit `data` and, for Julia, explicit
  `model_specification`. Choose `backend="local_gaussian"` deliberately;
  `allow_local_fallback` was removed. No observations are generated by the helper.
- `GenerativeModel.integrate_bayeux` defaults to the explicitly named
  `numpy_metropolis` chain. Select `backend="bayeux"` for actual NumPyro NUTS;
  density callables accept keyword parameters. The integration hub's density and
  transform callables accept a parameter mapping. Neither path invents evidence
  or effective sample size. Python source strings require the literal
  `allow_dynamic_code=True`, use per-call namespaces, and propagate failures.
- Unsupported `spatial_temporal` model types and unconfigured policy evaluation
  fail instead of returning fabricated uniform diagnostics.

## Owning source surfaces

Each file below was included in source review. Test links in the method inventory
and the regression files identify executable acceptance surfaces; source review
of an internal helper alone does not establish exhaustive runtime coverage.

| Owning Python surface | Callable definitions |
| --- | ---: |
| [__init__.py](../src/geo_infer_act/__init__.py) | 0 |
| [api/__init__.py](../src/geo_infer_act/api/__init__.py) | 0 |
| [api/client.py](../src/geo_infer_act/api/client.py) | 4 |
| [api/endpoints.py](../src/geo_infer_act/api/endpoints.py) | 1 |
| [api/interface.py](../src/geo_infer_act/api/interface.py) | 6 |
| [core/__init__.py](../src/geo_infer_act/core/__init__.py) | 0 |
| [core/active_inference.py](../src/geo_infer_act/core/active_inference.py) | 38 |
| [core/belief_updating.py](../src/geo_infer_act/core/belief_updating.py) | 6 |
| [core/civic_intel.py](../src/geo_infer_act/core/civic_intel.py) | 15 |
| [core/dynamic_causal_model.py](../src/geo_infer_act/core/dynamic_causal_model.py) | 9 |
| [core/factored_runtime.py](../src/geo_infer_act/core/factored_runtime.py) | 3 |
| [core/free_energy.py](../src/geo_infer_act/core/free_energy.py) | 9 |
| [core/generative_model.py](../src/geo_infer_act/core/generative_model.py) | 59 |
| [core/gnn_contract.py](../src/geo_infer_act/core/gnn_contract.py) | 11 |
| [core/gnn_factored_contract.py](../src/geo_infer_act/core/gnn_factored_contract.py) | 16 |
| [core/gnn_gaussian_contract.py](../src/geo_infer_act/core/gnn_gaussian_contract.py) | 3 |
| [core/markov_decision_process.py](../src/geo_infer_act/core/markov_decision_process.py) | 16 |
| [core/policy_selection.py](../src/geo_infer_act/core/policy_selection.py) | 11 |
| [core/spatial_agent.py](../src/geo_infer_act/core/spatial_agent.py) | 29 |
| [core/types.py](../src/geo_infer_act/core/types.py) | 14 |
| [core/variational_inference.py](../src/geo_infer_act/core/variational_inference.py) | 28 |
| [models/__init__.py](../src/geo_infer_act/models/__init__.py) | 0 |
| [models/base.py](../src/geo_infer_act/models/base.py) | 22 |
| [models/climate.py](../src/geo_infer_act/models/climate.py) | 8 |
| [models/continuous_pomdp.py](../src/geo_infer_act/models/continuous_pomdp.py) | 14 |
| [models/ecological.py](../src/geo_infer_act/models/ecological.py) | 6 |
| [models/multi_agent.py](../src/geo_infer_act/models/multi_agent.py) | 19 |
| [models/resource.py](../src/geo_infer_act/models/resource.py) | 5 |
| [models/urban.py](../src/geo_infer_act/models/urban.py) | 5 |
| [runners/__init__.py](../src/geo_infer_act/runners/__init__.py) | 0 |
| [runners/cli.py](../src/geo_infer_act/runners/cli.py) | 4 |
| [runners/contracts.py](../src/geo_infer_act/runners/contracts.py) | 4 |
| [runners/gallery.py](../src/geo_infer_act/runners/gallery.py) | 5 |
| [runners/h3.py](../src/geo_infer_act/runners/h3.py) | 6 |
| [runners/io.py](../src/geo_infer_act/runners/io.py) | 20 |
| [runners/scenarios.py](../src/geo_infer_act/runners/scenarios.py) | 68 |
| [runners/wrapper.py](../src/geo_infer_act/runners/wrapper.py) | 1 |
| [utils/__init__.py](../src/geo_infer_act/utils/__init__.py) | 0 |
| [utils/analysis.py](../src/geo_infer_act/utils/analysis.py) | 42 |
| [utils/bayeux_backend.py](../src/geo_infer_act/utils/bayeux_backend.py) | 1 |
| [utils/config.py](../src/geo_infer_act/utils/config.py) | 5 |
| [utils/h3_adapter.py](../src/geo_infer_act/utils/h3_adapter.py) | 16 |
| [utils/integration.py](../src/geo_infer_act/utils/integration.py) | 24 |
| [utils/math.py](../src/geo_infer_act/utils/math.py) | 24 |
| [utils/pymdp_adapter.py](../src/geo_infer_act/utils/pymdp_adapter.py) | 20 |
| [utils/spatial_diagnostics.py](../src/geo_infer_act/utils/spatial_diagnostics.py) | 25 |
| [utils/spatial_research.py](../src/geo_infer_act/utils/spatial_research.py) | 18 |
| [utils/visualization.py](../src/geo_infer_act/utils/visualization.py) | 17 |

The canonical export and runner artifact list is maintained in
[method_inventory.md](method_inventory.md). Cross-module migration and release
acceptance evidence is maintained in the repository release documentation.
