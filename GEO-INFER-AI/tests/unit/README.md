# GEO-INFER-AI/tests/unit

Unit workspace within `GEO-INFER-AI`.

## Contents

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
- `test_rng.py`
- `test_spatial_lag_features.py`
- `test_spatial_predictor.py`
- `test_training.py`
- `test_training_coverage.py`
- `test_training_wave2_fixes.py`

## Public Interface

- No public Python symbols are defined directly in this directory.

## Module Metadata

- Module: `GEO-INFER-AI`
- Package: `geo_infer_ai`
- Version: `0.3.0`
- Install: `uv sync --package geo-infer-ai`
- Tests: `uv run python -m pytest GEO-INFER-AI/tests/unit`

## Dependencies

- `numpy>=1.20.0`
- `pandas>=1.3.0`
- `scikit-learn>=1.0.0`
- `h3>=4.5.0,<5`


## Validation

```bash
uv run python -m pytest GEO-INFER-AI/tests/unit
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
