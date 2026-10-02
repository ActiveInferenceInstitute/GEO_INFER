# GEO-INFER-AI: model training and evaluation

AI owns spatial feature engineering, model training, interpolation and evaluation.
Its installed Python package is `geo_infer_ai`; inspect the module README and
SKILL for the current exported trainer, feature, interpolation and explanation
surfaces. Optional experiment tracking is declared separately from core numeric
operations. Dataset provenance, coordinate reference system, target units and
train/test separation remain caller responsibilities.

## A value-error example

The value-tolerance evaluator measures target-value errors. It does not estimate
geographic displacement. These three predictions have absolute errors 0, 1 and 2;
a tolerance of 1 accepts two of them.

```python
import math
import numpy as np
from geo_infer_ai import GeospatialModelEvaluator

result = GeospatialModelEvaluator().evaluate_value_tolerance(
    np.array([0., 2., 4.]), np.array([0., 3., 6.]), tolerance=1.
)
assert result["mean_absolute_error"] == 1.
assert math.isclose(result["within_tolerance_percentage"], 200 / 3)
```

In 0.4.0 this method replaces `evaluate_spatial_accuracy`; coordinates and
`buffer_distance` were removed from that value metric. Use a separate geographic
error calculation when displacement is the quantity of interest. Spatial block
cross-validation and held-out data require a declared partition rather than a
random row split that allows nearby samples into both folds.

## Verification

Run `uv run python GEO-INFER-TEST/run_unified_tests.py --module AI`. The owning
suite includes model evaluation, feature engineering, training, interpolation and
serialization cases. This example exercises the numeric evaluator; it establishes
no hardware or external model-service acceptance. See the
[0.4.0 migration guide](../releases/0.4.0_migration.md).
