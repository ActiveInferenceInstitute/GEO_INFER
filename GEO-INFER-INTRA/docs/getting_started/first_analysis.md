# Your First Spatial Inference Analysis

This walkthrough uses only current public exports from GEO-INFER-SPACE and
GEO-INFER-ACT. It converts a WGS84 location into an H3 v4 cell, performs a
categorical Active Inference step, and inspects a free-energy decomposition.

## Prerequisites

```bash
uv sync --all-packages
```

Both SPACE and ACT are needed throughout this walkthrough. The workspace sync
installs their base dependencies together; use `uv run --all-packages` when
running the snippets from the repository root.

## 1. Index a location with H3 v4

```python
from geo_infer_space import cell_to_latlng, latlng_to_cell

latitude, longitude = 37.7749, -122.4194  # WGS84 degrees
cell = latlng_to_cell(latitude, longitude, resolution=9)
center = cell_to_latlng(cell)

print(cell)
print(f"cell center: {center[0]:.5f}, {center[1]:.5f}")
```

`latlng_to_cell` expects latitude followed by longitude. Keep this order
explicit at application boundaries and document any projected CRS before using
metric operations.

## 2. Build a categorical generative model

```python
import numpy as np

from geo_infer_act import ActiveInferenceModel, GenerativeModel

generative_model = GenerativeModel(
    "categorical",
    {"state_dim": 3, "obs_dim": 3},
)
agent = ActiveInferenceModel(
    model_type="categorical",
    policy_selection_mode="deterministic",
    random_seed=42,
)
agent.set_generative_model(generative_model)

result = agent.step(
    np.array([1.0, 0.0, 0.0]),
    available_actions=["survey", "wait"],
    return_result=True,
)

print(result.beliefs)
print(result.action)
print(result.free_energy)
```

`return_result=True` returns an `ActiveInferenceStepResult`. Callers that need
the established tuple shape can omit it and receive `(beliefs, action)`.

## 3. Inspect the free-energy terms

```python
from geo_infer_act import FreeEnergyCalculator

calculator = FreeEnergyCalculator()
breakdown = calculator.compute_categorical_free_energy(
    beliefs=np.array([0.7, 0.2, 0.1]),
    observations=np.array([0.8, 0.15, 0.05]),
    preferences=np.array([0.6, 0.25, 0.15]),
    return_breakdown=True,
)

assert np.isfinite(breakdown.free_energy)
assert np.isclose(
    breakdown.free_energy,
    breakdown.complexity - breakdown.accuracy,
)
```

For categorical models ACT reports `F = complexity - accuracy` after finite
normalization. The typed result also exposes entropy and metadata.

## 4. Make the result spatial

An application can associate the inference result with the H3 cell created in
step 1:

```python
cell_result = {
    "cell": cell,
    "center": center,
    "beliefs": result.beliefs["states"].tolist(),
    "action": result.action,
    "free_energy": result.free_energy,
}
print(cell_result)
```

For multi-cell work use the ACT H3 methods documented in
[GEO-INFER-ACT/SKILL.md](../../../GEO-INFER-ACT/SKILL.md). Nested hierarchies
are opt-in and must use real H3 cells and ordered resolutions.

## 5. Validate the workflow

```bash
uv run python GEO-INFER-TEST/validate_active_inference_contract.py
uv run python GEO-INFER-TEST/validate_h3_active_inference_contract.py
uv run python -m pytest GEO-INFER-ACT/tests/unit/test_h3_active_inference.py -q
```

Generated examples should write artifacts to a temporary or explicit output
directory. Do not leave maps, model reports, or caches in the repository root.

## Next steps

- [Active Inference basics](active_inference_basics.md)
- [H3 v4 guide](../geospatial/data_formats/h3/index.md)
- [ACT module documentation](../modules/geo-infer-act.md)
- [Visualization guidance](../geospatial/visualization/index.md)
