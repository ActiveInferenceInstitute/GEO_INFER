# Urban Planning

Planning combines land-use permissions, mobility, green space, infrastructure,
and community objectives. Metrics inform a decision; a weighted score does not
establish consensus, causality, equity, or the effects of an intervention.

## A transparent land-use mix metric

This synthetic three-category distribution belongs to a real H3 planning cell.
The entropy is normalized by the number of declared categories. A balanced mix
is mathematically maximal under this metric, without implying a planning preference.

```python
import h3
import numpy as np
from geo_infer_space import H3StateSpace

cell = h3.latlng_to_cell(45.52, -122.67, 9)
space = H3StateSpace([cell])
fractions = np.array([0.5, 0.3, 0.2])
assert (fractions >= 0).all() and np.isclose(fractions.sum(), 1)
mix = -np.dot(fractions, np.log(fractions)) / np.log(len(fractions))
assert 0 < mix < 1
balanced = np.ones(3) / 3
assert np.isclose(-np.dot(balanced, np.log(balanced)) / np.log(3), 1)
assert space.locate(45.52, -122.67) == 0
```

A real zoning layer may permit uses that differ from observed use. Record that
choice and use projected overlap areas when assigning polygon fractions. Transport
analysis needs an actual network, edge costs, and connectivity; straight-line H3
adjacency is not a travel-time model. Site comparisons should expose objective
weights, hard constraints, missing-data coverage, and sensitivity to those choices.

For a sequential policy, build and validate an ACT model rather than labeling a
score's argmax as active inference. Keep participatory preferences traceable and
review predictions against observed outcomes. See [urban analytics](urban_analytics.md),
[active inference principles](../user_guide/active_inference_principles.md), and
[custom models](../advanced/custom_models.md).
