# Agricultural Intelligence

Sequential agricultural decisions require a generative model linking hidden crop
or soil states to measurements and actions. A multi-season study must also declare
phenology, weather drivers, crop rotations, costs, and observation support. These
are modeling choices rather than automatic capabilities of a facade class.

## A verifiable categorical perception step

The two hidden states below can represent two declared moisture regimes after
calibrating an observation model. The example validates a real `GNNArtifact` and
explicitly computes the Bayesian conditioning and identity prediction. The
[custom-model example](../advanced/custom_models.md) executes the ACT runner;
this synthetic model does not recommend irrigation.

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

To study interventions, add explicit action-conditioned transitions and justify
preference scores with domain objectives. Preserve units and state order when
mapping NDVI, soil sensors, or yield measurements into categorical observations.
Keep missing observations explicit. An unobserved model interval requires a declared
action history, not repeated evidence or an invented measurement.

Validate posterior calibration and management outcomes on held-out seasons before
interpreting a policy as useful. Do not infer causality from a fitted seasonal
correlation alone. See [agricultural data alignment](agricultural_applications.md),
[custom models](../advanced/custom_models.md), and
[action/observation schedules](../../../GEO-INFER-TIME/docs/action_observation_schedule.md).
