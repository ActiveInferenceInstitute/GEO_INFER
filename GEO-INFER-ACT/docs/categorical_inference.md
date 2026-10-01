# Categorical inference contract

`GenerativeModel.update_beliefs` and `CategoricalModel` use a
predict-then-update filter:

1. Predict the next state with `B @ beliefs`, where `B` is column-stochastic and
   `B[next_state, current_state] = P(next_state | current_state)`.
2. Update that prior with the observation model `A`, where
   `A[observation, state] = P(observation | state)`.

Categorical likelihoods are evaluated in log space. Observation vectors may be
non-negative hard counts or soft counts, including totals large enough that a
linear-space likelihood product would underflow. Every successful update returns
finite, non-negative beliefs that sum to one. Invalid shapes, negative values,
zero-mass priors, and observations with no posterior support raise `ValueError`.

`GenerativeModel` uses one `B` matrix for a single-level model. A hierarchical
`GenerativeModel` accepts a mapping such as
`{"level_0": B0, "level_1": B1}`; a single matrix may be broadcast only when
all levels have the same state dimension. If no `B` is supplied, the categorical
default is the identity transition, preserving state continuity.

The categorical model retains its row-stochastic matrix orientation:
`transition_matrix[current_state, next_state]`.
It applies `transition_matrix.T @ beliefs` before each observation update.
`BayesianBeliefUpdate.update_categorical` shares the same stable log-space
normalization helper but remains a pure Bayes update over the prior supplied by
the caller.

## Perception and policy timing

The simple categorical `ActiveInferenceModel` backend has a separate contract:
`perceive` conditions on the current beliefs once, and `act` evaluates policies
from that posterior. `act` passes the posterior directly to pymdp's policy
inference; it does not call state inference again. The returned beliefs, policy
diagnostics, and recorded free energy therefore describe the same observation
update. Free energy is retained from perception when action candidates or
preferences change. Repeated `act` calls neither advance time nor alter beliefs.

This backend normalizes observation vectors to frequencies before
conditioning, so `[8, 2]` and `[0.8, 0.2]` have the same effect. Unlike
`GenerativeModel.update_beliefs`, it does not propagate B before the next
observation. B supplies prospective transitions for policy evaluation. Use the
[GNN interchange runner](gnn_interchange.md) when each selected action must
propagate the posterior exactly once on an explicit observation schedule.

The tuple return from `step`, typed step results, arbitrary action labels,
scalar/list control counts, history, and optional local fallback remain
supported. Replacing the generative model clears observations and inference
diagnostics from the old model; `reset` also restores the initial beliefs.
Grid inference saves and restores the perception diagnostic with the rest of
the agent state. A backend that recovers after local perception receives the
local posterior directly, without repeating the observation update.
The opt-in local fallback retains hard/soft count semantics; its saved free
energy uses that count likelihood and the original perception prior. It does
not substitute the generative model's unrelated reference diagnostic.

At the adapter level, `run_model_step` and `run_pymdp_step` accept an explicit
`posterior` and finite `perception_free_energy` for policy-only inference.
The posterior must be a normalized, finite, nonnegative state vector. Metadata
identifies `inference_mode` as `perception_policy` or `policy_only`.

## Validation

Run the focused regression suite from the repository root:

```bash
PYTHONPATH=GEO-INFER-ACT/src uv run pytest \
  GEO-INFER-ACT/tests/unit/test_categorical_regressions.py -q
uv run pytest GEO-INFER-ACT/tests/unit/test_perception_policy_timing.py -q
```

Then run the ACT unit suite and the repository contract checks:

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module ACT
uv run python GEO-INFER-TEST/validate_repo_contracts.py --strict-source-language
```
