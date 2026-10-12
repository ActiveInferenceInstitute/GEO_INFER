# GNN and GEO-INFER interchange

GNN owns model notation, extraction and matrix provenance. GEO-INFER-SPACE owns
H3 topology and ordered spatial states; TIME owns observation scheduling; ACT
owns inference. The repositories exchange JSON artifacts and run in separate
Python environments. No generated source, pickle or cross-checkout imports are
needed. `gnn-geo-infer/1` supports one categorical state factor, one observation
modality and one control factor, with one-step policies.

The producer and normative format description are in GNN's
`src/gnn/export/geo_infer.py` and `src/gnn/export/geo_infer_contract.md` (GNN revisions before its 2026-09 package reorganization, including pinned pairing commit 903b9c339, use `src/export/geo_infer.py`). ACT's
`geo_infer_act.core.gnn_contract.GNNArtifact` validates the consumer boundary.
The contract version must change when axis meanings or inference timing change.

## Matrix and state contract

| Field | Shape / meaning |
| --- | --- |
| A | `[observation, state]`; each state column sums to one |
| B | `[next_state, current_state, action]`; axis zero sums to one |
| C | `[observation]`; log preferences, not normalized probabilities |
| D | `[state]`; initial prior at the first observation timestamp |
| E | `[action]`; prior over the one-step policies in action order |
| space.state_ids | Unique strings in matrix state order; never sorted on import |
| space.kind | `categorical`, or `h3` with canonical cells at one resolution |
| time.step_seconds | Explicit positive physical interval represented by B |
| provenance | Producer label and SHA-256 of the exact UTF-8 GNN source |

A, B, D and E are finite nonnegative probabilities; C is finite. Normalization
uses absolute tolerance 1e-8. Missing matrices, inferred defaults, unknown fields,
unsupported versions and mismatched dimensions are rejected. Artifact files are
bounded to 32 MiB, total dense matrix entries to 1,000,000, and runner input to
10,000 steps by default. Source hashes establish identity, not authenticity.

The artifact's digest hashes Python's sorted compact JSON representation in
UTF-8, without a trailing newline. This is not an RFC 8785 canonicalization claim.

## Run across separate environments

In the GNN checkout, install its optional H3 support when exporting spatial IDs:

```bash
uv sync --extra dev --extra geo-infer
uv run --no-sync python -m gnn.export.geo_infer \  # `gnn` package at src/gnn/, installed editable; pre-reorg revisions use `PYTHONPATH=src python -m export.geo_infer`
  input/gnn_files/pomdp_gridworld/pomdp_gridworld_3x3.md \
  /tmp/gridworld.geo-infer.json --step-seconds 60
```

In a GEO environment containing ACT and TIME (`geo-infer-act[gnn]`):

```python
from geo_infer_act.core.gnn_contract import GNNArtifact, run_gnn_inference

artifact = GNNArtifact.load('/tmp/gridworld.geo-infer.json')
trace = run_gnn_inference(artifact, [
    {'timestamp': '2026-01-01T00:00:00Z', 'observation': 0},
    {'timestamp': '2026-01-01T00:01:00Z', 'observation': 1},
], random_seed=42)
```

At time t, ACT conditions once using A and the current prior, evaluates policies
through real pymdp 1.0.3 including E, and propagates once with the selected B
slice to obtain the prior at t+1. Exact categorical conditioning retains
structural zeros. Backend posterior agreement is checked at 1e-5 tolerance;
zero-probability observations fail instead of receiving invented likelihoods.
Trace free energy is negative log model evidence. Backend diagnostics retain
pymdp's own precision and free-energy reporting.

Timestamps must include timezone offsets. TIME normalizes to UTC and rejects
naive timestamps, duplicates, reversals and missing intervals. It does not sort,
resample or fill observations. Timestep precision is limited to microseconds.

## H3 composition and reproducibility

`geo_infer_space.core.state_space.H3StateSpace` preserves supplied cell order.
Its sparse stay/diffuse operators conserve probability at domain boundaries by
retaining the mass of excluded neighbors at the source. Real pentagon topology
sets the neighbor degree. Dense tensor conversion has an explicit entry budget.
A geographic point outside the domain raises; it is never snapped to a cell.

Run the executable cross-repository check from GEO-INFER, after preparing GNN's
separate environment with its `geo-infer` extra:

```bash
uv run --no-sync python GEO-INFER-TEST/validate_gnn_interchange.py \
  --gnn-repo ../GeneralizedNotationNotation \
  --gnn-python ../GeneralizedNotationNotation/.venv/bin/python
```

It exports the real 3×3 GNN example, checks source provenance, executes repeatable
pymdp traces, and builds a seven-cell H3 model in SPACE. It passes that model
through the GNN exporter and verifies exact state order and transition values
before executing spatial inference. Both paths use real code in each environment.

## Worked H3 example: Cascadia river homing

[Cascadia river homing](cascadia_river_homing.md) exercises this contract on
real geometry: GEO-INFER-ACT compiles a typed 26-cell H3 network from the
retained NHDPlus HR lower Smith River lines, renders exact dyadic A–E as GNN
v1 Markdown, exports it through GNN's strict `gnn.export.geo_infer` with
`--space-kind h3`, and runs the closed loop through `run_gnn_inference`. The
same artifact is the input of a Lean 4 instance in the fep_lean checkout.

## Continuous inference correction

`ContinuousPOMDPActiveInference` now reports `F = complexity - accuracy`, where
complexity is `KL(q || predictive prior)` and accuracy is expected log likelihood.
The no-argument diagnostic uses the latest measurement and saved predictive prior;
after exact Kalman conditioning, F agrees with negative log evidence. Entropy is
Gaussian differential entropy. Covariances and shapes are validated; updates use
linear solves and Joseph covariance form.

`time_domain='continuous'` retains Euler propagation with `I + dt*A`, `dt*B`
and `dt*Q`. `time_domain='discrete'` uses A, B and Q directly per step. This
explicit option does not import continuous GNN models. The control objective is
expected squared observation error plus effort minus weighted Gaussian mutual
information. Repeated sensing conditions uncertainty to avoid double-counting
information, while expected error uses unconditional trajectory uncertainty.

## Contract boundaries

Version 1 remains limited to single-factor categorical models. Gaussian v2 below
and the separate [factored interchange](gnn_factored_interchange.md) provide
explicit additional model types. Factored artifacts require structured JSON with
dependency axes and enumerated policies; they do not claim Markdown extraction
or Step 7 support. Nonlinear models and implicit resampling remain unsupported.

## Discrete-time linear Gaussian v2

`geo_infer_act.core.gnn_gaussian_contract` provides `GaussianGNNArtifact` and
`run_gaussian_gnn_inference`. This separate model type preserves categorical v1.
The artifact declares `gnn-geo-infer/2`, `linear_gaussian`, dimensions
`states`, `observations`, `controls`; explicit F/G/H/Q/R matrices; initial belief;
physical coordinate units; discrete step seconds; and original-source digest.
GNN owns extraction and serialization in `gnn.export.geo_infer_gaussian` (pre-reorg revisions: `export.geo_infer_gaussian`).

F is `(state,state)`, G `(state,control)`, H `(observation,state)`; Q and R are
process and observation covariance. Q may be positive semidefinite; R and the
initial covariance must be positive definite. All arrays are finite, explicit
and bounded. Continuous-time generators, missing units, malformed axes,
duplicate JSON keys and implicit noise defaults are rejected.

```python
from geo_infer_act.core.gnn_gaussian_contract import (
    GaussianGNNArtifact, run_gaussian_gnn_inference,
)

artifact = GaussianGNNArtifact.load("gaussian.geo-infer.json")
trace = run_gaussian_gnn_inference(artifact, [
    {"timestamp": "2026-09-04T00:00:00Z", "observation": [1.0, 2.0], "control": [0.25]},
    {"timestamp": "2026-09-04T00:00:02Z", "observation": [2.0, 1.0], "control": [-0.5]},
])
```

The initial prior belongs to the first timestamp. Each record conditions once
using a linear solve and Joseph covariance update, records negative log evidence,
and applies the supplied control once to obtain the next prior. F/G/Q already
represent the declared physical interval; they are not Euler-discretized again.
This filtering runner evaluates supplied controls and does not select a policy.
The trace records prior/posterior/next-prior mean and covariance plus controls,
observations and timestamps, supporting direct analytic replay.

The GNN fixture `src/tests/export/gaussian_rectangular.md` uses three state
coordinates, two observations and one control. Corresponding ACT regressions in
`test_gnn_gaussian_contract.py` prove exact initial conditioning, unequal axes,
control timing, evidence and covariance, immutable JSON snapshots and rejection
before malformed dense allocation. Cross-repository validation uses separate
locked Python environments.
