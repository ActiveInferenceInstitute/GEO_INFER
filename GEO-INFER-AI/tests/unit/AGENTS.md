# Agent Instructions: GEO-INFER-AI/tests/unit

## Scope

- Owning module: `GEO-INFER-AI`
- Python package: `geo_infer_ai`
- Directory role: Unit workspace within `GEO-INFER-AI`.

## Capabilities

- Maintains the tracked files and subdirectories listed below for this workspace.
- Validates behavior with the command in the Validation section.
- Integrates through `geo_infer_ai` and the owning module's public contracts.

## Working Rules

- Keep changes scoped to this directory unless an import, test, or documented command requires a coordinated edit.
- Prefer existing module patterns and public exports over new orchestration layers.
- Do not add planned, fake, mock, stub, or placeholder behavior to user-facing docs.
- If external services are involved, keep deterministic local validation available.

## Local Contents

- `test_cross_validation.py`
- `test_explainability.py`
- `test_explainability_determinism.py`
- `test_feature_engineering.py`
- `test_feature_engineering_coverage.py`
- `test_geospatial_ai.py`
- `test_geospatial_ai_coverage.py`
- `test_idw_interpolation.py`
- `test_image_classifier.py`
- `test_kriging.py`
- `test_mlflow_integration.py`
- `test_model_evaluation.py`
- `test_model_evaluation_coverage.py`
- `test_permutation_ownership.py`
- `test_rng.py`
- `test_spatial_lag_features.py`
- `test_spatial_predictor.py`
- `test_training.py`
- `test_training_coverage.py`
- `test_training_wave2_fixes.py`

## Validation

```bash
uv run python -m pytest GEO-INFER-AI/tests/unit
```


## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
