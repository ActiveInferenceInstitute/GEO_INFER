# Active Inference interfaces and spatial composition

ACT owns GEO-INFER's Active Inference models, perception updates, free-energy
calculations, policy evaluation and typed diagnostics. Application scripts supply
observations, model assumptions, configuration and output destinations through
these package interfaces. They do not implement a second inference engine.

## What the model declares

For a categorical model, likelihood `A` is indexed `[observation, state]` and each
state column sums to one. Controlled transitions `B` are indexed
`[next_state, current_state, action]`, with each current-state column summing to
one for each action. The prior is a probability vector in that same state order.
Preferences and available actions must match the declared model dimensions.
Rectangular likelihoods require an explicit `obs_dim`.

Variational free energy is model-conditioned evidence:

```text
F(q, o) = KL(q(s) || p(s)) - E_q[log p(o | s)]
```

Its complexity and accuracy terms concern a supplied likelihood and prior.
Expected free energy evaluates declared predictions and preferences under a
policy. In ACT's pymdp diagnostics, `negative_expected_free_energy` is a score
where higher is better; the corresponding expected free energy has the opposite
sign. Policy probabilities also depend on precision and policy priors. These
quantities are separate from forecast error, geographic distance and field
calibration. A low free energy on an assumed model does not demonstrate that the
model represents an ecosystem or civic system accurately.

## An actual categorical perception-action step

This tiny analytical fixture uses the installed ACT runtime. Its categories and
actions are declared symbols, without an implied physical deployment. Bayes'
rule independently gives posterior `[0.8, 0.2]` for the first observation.

```python
import numpy as np
from geo_infer_act import ActiveInferenceModel, ActiveInferenceStepResult, GenerativeModel

model = GenerativeModel("categorical", {"state_dim": 2, "obs_dim": 2})
model.observation_model = np.array([[0.8, 0.2], [0.2, 0.8]])
model.transition_model = np.stack([
    np.eye(2), np.array([[0.0, 1.0], [1.0, 0.0]]),
], axis=2)
model.beliefs["states"] = np.array([0.5, 0.5])
agent = ActiveInferenceModel(
    model_type="categorical", policy_selection_mode="deterministic", random_seed=7,
)
agent.set_generative_model(model)
result = agent.step(
    np.array([1.0, 0.0]), available_actions=["stay", "flip"], return_result=True,
)
assert isinstance(result, ActiveInferenceStepResult)
np.testing.assert_allclose(result.beliefs["states"], [0.8, 0.2], atol=1e-6)
assert result.action in {"stay", "flip"}
assert np.isfinite(result.free_energy)
assert np.isclose(np.sum(result.beliefs["states"]), 1.0)
```

`return_result=True` returns `ActiveInferenceStepResult`; the default `step`
returns the `(beliefs, action)` tuple. The categorical belief payload retains the
model's `{"states": vector}` shape. The real categorical backend is
`inferactively-pymdp==1.0.3`, accessed through ACT's owning adapter. Invalid model
shapes, observations or backend failures cannot justify invented uniform
results. Soft categorical observations, categorical modality indices in factored
models, and continuous measurements have different input contracts.

`PolicySelector` can select policies with explicitly supplied expected-free-energy
values or evaluate its declared predictive policy fields. Its generic helper's
predictive-entropy exploration term is identified in breakdown metadata; it
must not be described as a full future-observation mutual-information calculation.
The pymdp runtime evaluates its actual generative matrices. Preserve the recorded
`policy_beliefs` when auditing scores computed before later spatial blending.

## Two spatial interpretations

ACT's distributed H3 inference attaches a categorical model to each cell. Its
per-cell state dimension does not become the number of cells. H3 neighborhood
blending and nested aggregation follow their declared spatial contracts;
parent-level uniform-reference KL is not a measurement likelihood or observed
variational free energy.

The SPACE/TIME state-space composition instead treats the supplied H3 cells as
one model's ordered global states. Every prior, likelihood column and transition
axis must retain that order. SPACE supplies actual sparse movement operators,
including pentagon degree and reflecting domain boundaries.

```python
import h3
from geo_infer_space import H3StateSpace

center = h3.latlng_to_cell(41.75, -124.2, 8)
neighbor = sorted(set(h3.grid_disk(center, 1)) - {center})[0]
state_space = H3StateSpace([neighbor, center])
stay, diffuse = state_space.transitions()
assert state_space.cells == (neighbor, center)
np.testing.assert_allclose(np.asarray(stay.sum(axis=0)).ravel(), [1.0, 1.0])
np.testing.assert_allclose(np.asarray(diffuse.sum(axis=0)).ravel(), [1.0, 1.0])
assert diffuse[0, 1] > 0.0
```

To execute that global-state interpretation, bind the same state order and a
physical step to ACT's categorical GNN contract. In this fixture, the sensor
with the larger reading reports category 0 (neighbor) or 1 (center). This
classification rule, the likelihood, and the center preference are declared
assumptions. The two readings are complete; classification rejects a gap rather
than converting it to a category.

```python
import hashlib
import json
import pandas as pd
from geo_infer_space import align_h3_observations
from geo_infer_act.core.gnn_contract import GNNArtifact, run_gnn_inference

times = ["2024-01-01T00:00:00Z", "2024-01-01T00:01:00Z"]
records = pd.DataFrame([
    {"cell": neighbor, "timestamp": times[0], "value": 0.0},
    {"cell": center, "timestamp": times[0], "value": 3.0},
    {"cell": neighbor, "timestamp": times[1], "value": 4.0},
    {"cell": center, "timestamp": times[1], "value": 6.0},
])
series = align_h3_observations(
    records, state_space=state_space, timestamps=times, max_entries=4,
)
assert not series.data.isna().any().any()
observations = [
    {"timestamp": instant.isoformat(), "observation": int(np.argmax(row))}
    for instant, row in series.data.iterrows()
]
transition = state_space.dense_transition_tensor(max_entries=8)
artifact = GNNArtifact.from_dict({
    "schema_version": "gnn-geo-infer/1",
    "model_type": "categorical", "model_name": "two-cell analytical fixture",
    "dimensions": {"states": 2, "observations": 2, "actions": 2},
    "matrices": {
        "A": [[0.9, 0.2], [0.1, 0.8]], "B": transition.tolist(),
        "C": [0.0, 2.0], "D": [0.99, 0.01], "E": [0.5, 0.5],
    },
    "space": {"kind": "h3", "state_ids": list(state_space.cells)},
    "time": {"step_seconds": 60},
    "provenance": {
        "producer": "geo-infer-documentation",
        "source_sha256": hashlib.sha256(
            json.dumps(records.to_dict("records"), sort_keys=True).encode()
        ).hexdigest(),
    },
})
trace = run_gnn_inference(artifact, observations, random_seed=42)
assert trace["state_ids"] == list(state_space.cells)
assert [step["action"] for step in trace["steps"]] == [1, 0]
np.testing.assert_allclose(trace["steps"][0]["posterior"], [99 / 107, 8 / 107])
first = trace["steps"][0]
np.testing.assert_allclose(first["next_prior"], [503 / 642, 139 / 642])
np.testing.assert_allclose(
    trace["steps"][1]["prior"], transition[:, :, first["action"]] @ first["posterior"],
)
assert (pd.Timestamp(trace["steps"][1]["timestamp"]) -
        pd.Timestamp(first["timestamp"])).total_seconds() == 60
```

The first posterior follows Bayes' rule from likelihood `[0.1, 0.8]` and prior
`[0.99, 0.01]`; ACT's real backend evaluates the policies, then the selected H3
operator advances the prior once. `run_gnn_inference` requires contiguous model
steps and rejects missing intervals. A trace records model decisions; it does
not establish that an action was performed by a sensor or a physical system.

Dense transition materialization requires an explicit allocation budget. Spatial
observations should first cross `align_h3_observations` with a declared UTC axis.
An observed sensor value is not automatically a likelihood or a posterior;
classify or model it through a stated observation mapping. Missing pairs remain
missing, and zero remains an observation. Never insert zero or reuse the previous
posterior as if it were a newly received measurement.

For a model step measured in seconds, TIME's `inference_schedule` validates every
step. If observations have gaps, `action_observation_schedule` requires every
intervening action. It validates a history and does not select actions, interpolate
measurements or infer hidden policy execution. DCM parameters describe
continuous-time derivatives; DCM fitting and filtering use the actual increasing
model-step axis. Generic Gaussian `act` has no declared control dynamics and
raises; choose `ContinuousPOMDPActiveInference` or the declared Gaussian GNN
runner for control evaluation.

## Configuration, runners and artifacts

Use `GenerativeModel` and `ActiveInferenceModel` for package calls, or the
`geo-infer-act-run` entrypoint for declared runner scenarios. The CLI parses
configuration and delegates execution to the owning runners; configuration does
not authorize an external deployment or hazard-response action.

H3 and nested runner results retain typed `H3GridInferenceResult`,
`NestedH3GridInferenceResult`, `SpatialInferenceTrace` and related diagnostic
records. Manifests bind data, CSV/GeoJSON outputs, figures and plotted-data
sidecars. Read [the runner contract](../../GEO-INFER-ACT/docs/geospatial_applications.md)
before interpreting the artifacts. A browser-rendered map, a valid artifact hash,
a numerical oracle and a calibrated field result establish different claims.

Research profiles deliberately specify likelihoods, preferences and controlled
transitions. They do not guarantee diverse selected actions on every input:
nearby observations may share the same optimal policy. Diversity tests need a
fixture crossing independently known policy regions. For reproducible plots,
use the package gallery and its recorded score/softmax/selection checks:

```bash
uv run python GEO-INFER-ACT/examples/spatial_active_inference_gallery.py --json
```

Choose external integrations by their explicit backend names. Bayeux's NumPyro
NUTS path and the local `numpy_metropolis` sampler are distinct algorithms;
Julia/RxInfer requires its own configured runtime and model specification.
Backend absence and runtime failure remain errors or explicitly unavailable
results. Dynamic source execution requires the explicit opt-in documented by
its owning interface; caller-supplied strings are not a safe persistence format.

## Verification and further contracts

Run focused analytical and real-backend regressions before fleet gates:

```bash
uv run python GEO-INFER-TEST/validate_active_inference_contract.py
uv run python GEO-INFER-TEST/validate_h3_active_inference_contract.py
uv run python -m pytest GEO-INFER-TEST/tests/integration/test_space_time_composition_contract.py
uv run python GEO-INFER-TEST/run_unified_tests.py --module ACT
uv run python GEO-INFER-TEST/validate_doc_examples.py
```

See [ACT's method review](../../GEO-INFER-ACT/docs/0_4_method_review.md),
[the complete method inventory](../../GEO-INFER-ACT/docs/method_inventory.md),
[SPACE/TIME composition](../../GEO-INFER-SPACE/docs/CROSS_MODULE_COMPOSITION.md),
[the temporal guide](temporal_analysis_guide.md), and
[the release evidence boundaries](releases/0.4.0_readiness.md). Source review and
small numerical fixtures do not establish all data distributions, live external
services, hardware acceleration, licensed data, historical PROJ causality or
advisor authentication. Those claims require separate acceptance evidence.
