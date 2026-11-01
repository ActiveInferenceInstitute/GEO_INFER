# E4 fixture admission: GEO-INFER proposal

**Status:** proposed for companion-owner agreement (2026-11-01). This declares one
admitted interchange fixture and its consumer semantics before implementation. It
does not claim formal-to-numerical equivalence, refresh a native receipt or extend
any published release.

E4 is a GNN roadmap item: bind formal statements to numerical semantics through
independent source extraction and numerical witnesses. GEO-INFER consumes GNN
artifacts, so its part is to admit one fixture explicitly and state the source,
quantities, axes, conditioning, time, units and precision both sides implement
against. Ownership and the acceptance boundary are recorded in the
[compatibility guide](companion_releases.md#next-cross-repository-handoff-e4-and-s1)
and [TODO.md](../../../TODO.md).

## Proposed fixture

| Item | Declaration |
| --- | --- |
| GNN source | `tests/export/gaussian_rectangular.md` |
| GNN revision | `8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1` (v5.0.3) |
| Source SHA-256 | `f8e52c35ef8795f90c848f1db0fc1c2f7e1dbed2017dce9b46f55a5dadf8bd12` |
| Artifact contract | `gnn-geo-infer/2`, `model_type="linear_gaussian"` |
| Producer | `gnn.export.geo_infer` with `--model-type linear_gaussian --step-seconds 2 --units units.json` |
| Consumer | `geo_infer_act.core.gnn_gaussian_contract.GaussianGNNArtifact`, `run_gaussian_gnn_inference` |

The fixture is deliberately asymmetric: three state coordinates, two observations
and one control. It carries a nonidentity rectangular `H`, a rectangular `G`, a
nonzero prior mean and correlated prior and noise covariances, so transposed
axes, shifted controls and conditioning-order mistakes remain visible.

## Declared semantics

### Quantities and axes

- `F` `(state, state)`: one declared interval of autonomous dynamics.
- `G` `(state, control)`; `H` `(observation, state)`.
- `Q` `(state, state)`, symmetric positive semidefinite: per-interval process
  covariance. `R` `(observation, observation)`, symmetric positive definite.
- `initial_belief.mean` `(state,)` and `.covariance` `(state, state)`, symmetric
  positive definite.
- JSON arrays are row-major `[row][column]`; no implicit transpose is applied.
  Units follow the declared coordinate ratios; covariance units follow coordinate
  products, and no unit conversion is implicit.

### Conditioning and time

- The initial belief describes the state at the first measurement timestamp. That
  record is conditioned once with a linear solve and a Joseph covariance update.
- The record's control is applied exactly once to obtain the next prior:
  `F @ posterior_mean + G @ control` and
  `F @ posterior_covariance @ F.T + Q`.
- `F`, `G` and `Q` already represent the declared physical interval
  (`time.step_seconds = 2`); they are never Euler-discretized or rescaled.
- Timestamps are timezone-aware, unique, ordered and regularly spaced by
  `step_seconds`; naive, duplicate, reversed and gapped schedules are rejected,
  and observations are never sorted, resampled or filled.
- Every step records prior, posterior and next-prior mean and full covariance plus
  negative log evidence. This is filtering with supplied controls; the runner does
  not select policies and does not smooth.

### Units

`states = ["m", "m/s", "K"]`, `observations = ["m", "m/s"]`, `controls = ["N"]`,
one nonempty label per coordinate, as in the exporter's `units.json` for this
fixture.

### Precision

IEEE-754 float64 throughout; JSON numbers are parsed as float64 and no float32
substitution is accepted. Companion witness comparisons for these small
well-conditioned fixtures use absolute tolerance `1e-12` and relative tolerance
`1e-10` (GNN E4 foundation and v2 receipts). GEO's consumer-side analytic
reference for this fixture compares at the same tolerances.

## GEO-side evidence available now

- `GEO-INFER-ACT/tests/unit/test_gnn_gaussian_contract.py` instantiates exactly
  this fixture's parameters and compares the trace against an independently
  computed closed form: posterior mean `[2/3, 4/3, 0]`, posterior covariance
  `diag(1/3, 4/3, 9)`, next-prior mean `[19/12, 11/6, 0]`, next-prior covariance
  `diag(43/30, 43/30, 47/20)` and the negative log evidence, at the declared
  tolerances.
- The same module rejects a continuous-time domain, singular `R`, negative `Q`,
  asymmetric covariance, a transposed `G` axis, missing units, boolean leaves and
  oversized dimensions; it rejects irregular, naive, reversed-shape, non-finite,
  incomplete and over-budget records; it proves shape rejection precedes dense
  allocation, and that indefinite covariances cannot pass through NaN comparisons.
- `GEO-INFER-TEST/validate_gnn_interchange.py` exports the tracked Gaussian and
  categorical fixtures in a separate locked GNN environment and replays GEO
  inference in two Python versions. GNN's
  `tests/export/test_geo_infer_consumer_compat.py` mirrors GEO's consumer rules
  producer-side. Neither is an E4 witness: both execute real producer and consumer
  code instead of independently reading the source.

## Precise gap

1. The semantic witness reads its own authored fixtures
   (`tests/render/test_source_numerical_witnesses.py`) through
   `src/gnn/validation/witness_source.py`, which admits only its own heading set
   (including a versioned `## NumericalWitness`) and JSON-valued assignment
   literals. The interchange fixture declares tuple-set
   `InitialParameterization` literals and no witness section, so no single source
   identity is currently both exported and independently witnessed.
2. The witness's Gaussian contract admits one passive stationary LGSSM with no
   authored control; this fixture declares `G` and GEO applies supplied controls.
   The admitted control semantics must be fixed before implementation.
3. No joint receipt binds the exporter artifact, the independent reference and the
   GEO consumer trace for one source digest. GNN and GEO each hold separate tests.
4. FEP_Formal's formal statement is not yet bound to this fixture; its
   `fep-semantic-witness-audit-v1` lane is independently governed.

## Options for the companion owners

- **A. Widen the witness grammar to this fixture.** Extend the restricted reader
  (or add a JSON-valued variant with the witness section) and declare a fixed
  control sequence for the admitted horizon, so one source is exported and
  independently witnessed.
- **B. Admit a passive restriction first.** Use the same state/observation
  dimensions with `G` fixed to a declared constant (or zero) and treat authored
  controls as a separate extension. Smallest first joint receipt.
- **C. Bind two sources by declaration.** Keep the witness fixture separate and
  declare parameter identity with the interchange source. Weakest option: identity
  would be assumed rather than witnessed, and the declaration must say so.

GEO's consumer supports options A and B without contract changes; option A also
requires the exporter to accept the witness fixture's assignment grammar, which is
GNN-owned.

## Acceptance boundary

Agreement requires independent source extraction and numerical references, mismatch
counterexamples (source, axis, action-time, conditioning, covariance), and separate
current native, proof and custody receipts. Matching values, shapes or successful
runners do not prove equivalence, and this proposal changes no pin, formal statement
or runtime behavior. A later GNN pin bump must re-declare the fixture against its
new source digest.
