#!/usr/bin/env python3
"""GEO-INFER-AI module orchestrator.

Runs one documented end-to-end AI operation on synthetic data: fit the
module's ``SpatialPredictor`` (ridge regression with automatic spatial
features) on synthetic terrain observations, predict held-out targets, and
score the predictions with ``GeospatialModelEvaluator``. All work goes
through the real ``geo_infer_ai`` public API.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

_ORCHESTRATORS_DIR = Path(__file__).resolve().parents[2]
if str(_ORCHESTRATORS_DIR) not in sys.path:
    sys.path.insert(0, str(_ORCHESTRATORS_DIR))

from _lib import run_module_orchestrator  # noqa: E402


def _operation() -> dict[str, Any]:
    from geo_infer_ai import GeospatialModelEvaluator, SpatialPredictor

    rng = np.random.default_rng(42)

    # Synthetic study region: 120 samples of terrain features over a
    # fictional Willamette-Valley-like box.
    n_samples = 120
    coordinates = np.column_stack(
        [
            rng.uniform(-123.4, -122.8, n_samples),
            rng.uniform(44.8, 45.2, n_samples),
        ]
    )
    elevation = rng.uniform(0.0, 800.0, n_samples)
    slope = rng.uniform(0.0, 35.0, n_samples)
    ndvi = rng.uniform(0.0, 1.0, n_samples)
    features = np.column_stack([elevation, slope, ndvi])

    target = (
        12.0
        + 0.045 * elevation
        + 1.3 * slope
        + 42.0 * ndvi
        + rng.normal(0.0, 3.0, n_samples)
    )

    split = 90
    (
        X_train,
        X_test,
        y_train,
        y_test,
        coords_train,
        coords_test,
    ) = (
        features[:split],
        features[split:],
        target[:split],
        target[split:],
        coordinates[:split],
        coordinates[split:],
    )

    predictor = SpatialPredictor(model_type="ridge", include_spatial_features=True)
    predictor.fit(X_train, y_train, coordinates=coords_train)
    predictions = predictor.predict(X_test, coordinates=coords_test)

    metrics = GeospatialModelEvaluator().evaluate_regression(y_test, predictions)

    return {
        "operation": "spatial_regression_train_predict_evaluate",
        "model_type": predictor.model_type,
        "n_train": int(split),
        "n_test": int(n_samples - split),
        "feature_names": predictor.feature_names_,
        "metrics": metrics,
        "sample_predictions": [round(float(p), 4) for p in predictions[:5]],
        "sample_true_values": [round(float(v), 4) for v in y_test[:5]],
    }


if __name__ == "__main__":
    sys.exit(run_module_orchestrator("AI", _operation))
