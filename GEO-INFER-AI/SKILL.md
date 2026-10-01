---
name: geo-infer-ai
description: Machine learning pipelines and model selection for geospatial AI. Use when training spatial ML models, building prediction pipelines, performing feature engineering on geographic data, spatial interpolation/kriging, or evaluating spatial model performance.
prerequisites:
  recommended:
    - geo-infer-space
    - geo-infer-data
    - geo-infer-act
    - geo-infer-bayes
difficulty: advanced
estimated_time: 60min
examples_dir: ../GEO-INFER-EXAMPLES/examples/
---

# GEO-INFER-AI

## Instructions

### Core Capabilities

- **ML pipelines**: Spatial feature engineering → training → evaluation with scikit-learn
- **Geospatial active inference**: H3-grid environmental modeling and belief
  updating via `EnvironmentalActiveInferenceEngine`; multi-scale hierarchical
  spatial analysis via `MultiScaleHierarchicalAnalyzer` with Gaussian-process
  and DBSCAN spatial methods
- **Model selection**: Cross-validation, hyperparameter search, spatial block cross-validation
- **Spatial features**: Coordinate-based features (distances, angles, spatial lag)
- **Spatial interpolation**: IDW and Ordinary Kriging with prediction variance
- **Explainability**: Permutation importance, SHAP-like values, partial dependence
- **MLOps**: Optional MLflow experiment tracking (graceful degradation without it)

### Key Imports

```python
from geo_infer_ai import (
    ModelTrainer,
    TrainingConfig,
    GeospatialModelEvaluator,
    ImageClassifier,
    SpatialPredictor,
    IDWInterpolator,
    OrdinaryKriging,
    GeospatialFeatureEngineer,
    ModelExplainer,
    MLflowPipeline,
    EnvironmentalActiveInferenceEngine,
    MultiScaleHierarchicalAnalyzer,
```

## Examples

Train and spatially cross-validate a model:

```python
from sklearn.ensemble import RandomForestRegressor
from geo_infer_ai import GeospatialModelEvaluator

model = RandomForestRegressor(n_estimators=50, random_state=42)
evaluator = GeospatialModelEvaluator()

results = evaluator.cross_validate_spatial(model, X, y, coordinates, n_splits=3)
print(results["mean_score"], results["std_score"])
```

Fit an IDW interpolator:

```python
import numpy as np
from geo_infer_ai import IDWInterpolator

coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
values = np.array([10.0, 20.0, 30.0])

interpolator = IDWInterpolator(power=2.0).fit(coords, values)
pred = interpolator.predict(np.array([[0.5, 0.5]]))
```

Spatial feature engineering:

```python
from geo_infer_ai import GeospatialFeatureEngineer

engineer = GeospatialFeatureEngineer(normalize=True)
features = engineer.create_spatial_features(coordinates, include_distances=True)
```

## Guidelines

- Use `cross_validate_spatial` (not shuffled K-fold) when samples are spatially
  autocorrelated — it blocks contiguous regions to prevent spatial leakage.
- Kriging `predict` returns `(values, variances)`; use the variances for
  prediction-uncertainty estimates.
- `GeospatialFeatureEngineer` does not correct for spatial autocorrelation;
  it is handled by spatial models and block CV.

### Integrations

- Standalone: depends only on numpy, pandas, scikit-learn, h3
- Optional MLflow tracking via the `mlops` extra (`MLflowPipeline` degrades
  gracefully to disabled mode when MLflow is absent)
- GEO-INFER-MATH provides the config-driven general-purpose interpolation
  counterparts (see `geo_infer_ai.models.predictive` docstring for the
  deliberate API differences)
- Test: `uv run python GEO-INFER-TEST/run_unified_tests.py --module AI`
