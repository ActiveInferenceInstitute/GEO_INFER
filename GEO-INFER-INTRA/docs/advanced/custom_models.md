# Custom Models

Custom models should declare the observation model, latent state meaning, transition
rule, preferences, units, and fitted parameters before inference. A useful model
extension has an independent numerical oracle and a clear owner; inheriting a
class alone does not establish a correct generative model.

## A bounded categorical model

The maintained ACT interchange surface is `GNNArtifact` and `run_gnn_inference`.
`A` uses [observation, state], `B` uses [next state, current state, action],
`D` is the initial state prior, `E` is the action prior, and `C` contains finite
observation preference scores. Probability columns must already sum to one.
This one-action fixture verifies a Bayesian posterior and its identity transition.

```python
import hashlib
import numpy as np
from geo_infer_act.core.gnn_contract import GNNArtifact, run_gnn_inference

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
trace = run_gnn_inference(artifact, [{"timestamp": "2024-01-01T00:00:00Z", "observation": 1}], random_seed=42)
expected = np.array([0.6 * 0.1, 0.4 * 0.8])
expected /= expected.sum()
np.testing.assert_allclose(trace["steps"][0]["posterior"], expected, atol=1e-12)
np.testing.assert_allclose(trace["steps"][0]["next_prior"], expected, atol=1e-12)
assert trace["steps"][0]["action"] == 0
```

## A Gaussian-process model

`SpatialGP.fit` conditions on declared kernel hyperparameters; `predict` returns
conditional means and optionally standard deviations. The inputs below are
normalized coordinates, not latitude/longitude distances. For physical space-time
inputs, declare a projection or geographic metric and a separate temporal scale.

```python
import numpy as np
from geo_infer_bayes.models.spatial_gp import SpatialGP

X = np.array([[0.0], [0.5], [1.0]])
y = np.array([0.0, 1.0, 0.0])
model = SpatialGP(kernel="rbf", lengthscale=0.3, noise=0.01)
model.fit(X, y)
mean, std = model.predict(np.array([[0.25], [0.75]]), return_std=True)
assert mean.shape == std.shape == (2,)
assert np.isfinite(mean).all() and np.isfinite(std).all()
assert (std >= 0).all()
np.testing.assert_allclose(mean[0], mean[1], atol=1e-12)
```
Model comparison should use held-out predictions or an appropriate predictive
score with the same observation likelihood across candidates. Fit quality and
calibration are separate: a good training fit does not establish transferable
uncertainty. Do not tune on the final holdout. Treat finite numerical checks,
parameter-domain checks, and repeatable seeds as prerequisites for comparison.

See [ACT](../modules/geo-infer-act.md), [BAYES](../modules/geo-infer-bayes.md),
and the [research inference contracts](../research_grade_inference_contracts.md).
