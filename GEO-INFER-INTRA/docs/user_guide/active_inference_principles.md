# Active Inference Principles

Active inference connects perception and action through an explicit generative
model. A state posterior reflects prior beliefs and the likelihood of an observation;
a policy compares predicted outcomes using declared preferences and uncertainty.
Learning updates model parameters and requires its own data and assumptions.

For latent state `s` and observation `o`, variational free energy is
`F = E_Q[log Q(s) - log P(o,s)]`. It bounds negative log evidence when the
probability model and approximation are well defined. Expected free energy is a
policy-dependent quantity; its decomposition depends on the implementation and
preference representation. It is not a universal name for every prediction score.

## Check categorical belief updating

The exact discrete update is `posterior ∝ prior × A[observed, :]`.
Prediction applies `B[:, :, action]` to that posterior. The example validates a
real `GNNArtifact` and computes these analytical identities explicitly. The
[custom-model example](../advanced/custom_models.md) also executes the ACT runner.

```python
import hashlib
import numpy as np
from geo_infer_act.core.gnn_contract import GNNArtifact

source = b"documented two-state categorical fixture"
artifact = GNNArtifact.from_dict({
    "schema_version": "gnn-geo-infer/1", "model_type": "categorical",
    "model_name": "documented categorical model",
    "dimensions": {"states": 2, "observations": 2, "actions": 1},
    "matrices": {
        "A": [[0.9, 0.2], [0.1, 0.8]],
        "B": [[[1.0], [0.0]], [[0.0], [1.0]]],
        "C": [0.0, 1.0], "D": [0.6, 0.4], "E": [1.0],
    },
    "space": {"kind": "categorical", "state_ids": ["state_0", "state_1"]},
    "time": {"step_seconds": 60},
    "provenance": {"producer": "documentation", "source_sha256": hashlib.sha256(source).hexdigest()},
})
matrices = {name: np.asarray(value) for name, value in artifact.to_dict()["matrices"].items()}
posterior = matrices["A"][1] * matrices["D"]
posterior /= posterior.sum()
np.testing.assert_allclose(posterior, [3 / 19, 16 / 19], atol=1e-12)
next_prior = matrices["B"][:, :, 0] @ posterior
np.testing.assert_allclose(next_prior, posterior, atol=1e-12)
assert artifact.to_dict()["space"]["state_ids"] == ["state_0", "state_1"]
```

## Spatial and temporal interpretation

Geospatial states can use `H3StateSpace.cells` in explicit caller order. An H3
movement operator represents a chosen abstract transition model; it does not
establish road access, intervention causality, or sensor likelihoods. Those require
domain-specific calibration. Fixed-step inference must also preserve physical
model intervals and explicit actions through observation gaps.

Uncertainty representation must be inspected in each method's output. A linear
trend slope, heuristic interpolation, or risk score does not inherently include
posterior uncertainty. Evaluate posterior calibration and action consequences
against held-out observations or a validated simulator before operational use.

See [custom models](../advanced/custom_models.md),
[SPACE composition](../../../GEO-INFER-SPACE/docs/CROSS_MODULE_COMPOSITION.md), and
[research inference contracts](../research_grade_inference_contracts.md).
