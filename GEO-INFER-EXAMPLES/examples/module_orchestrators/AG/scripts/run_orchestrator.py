#!/usr/bin/env python3
"""GEO-INFER-AG module orchestrator.

Runs one documented end-to-end AG operation on synthetic data: train the
``CropYieldModel`` in machine-learning mode on a synthetic set of field
observations, predict yields for a validation season, and rank the driving
environment/management factors by feature importance. All work goes through
the real ``geo_infer_ag`` public API.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_ORCHESTRATORS_DIR = Path(__file__).resolve().parents[2]
if str(_ORCHESTRATORS_DIR) not in sys.path:
    sys.path.insert(0, str(_ORCHESTRATORS_DIR))

from _lib import run_module_orchestrator  # noqa: E402


def _operation() -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    from geo_infer_ag import CropYieldModel

    rng = np.random.default_rng(42)

    feature_columns = [
        "rainfall_mm",
        "growing_degree_days",
        "soil_organic_matter_pct",
        "ndvi_peak",
        "irrigation_hours",
    ]

    def _synthetic_fields(count: int) -> pd.DataFrame:
        """Generate one season of synthetic field observations with yields."""
        frame = pd.DataFrame(
            {
                "rainfall_mm": rng.uniform(350.0, 800.0, size=count),
                "growing_degree_days": rng.uniform(1200.0, 2200.0, size=count),
                "soil_organic_matter_pct": rng.uniform(1.0, 4.5, size=count),
                "ndvi_peak": rng.uniform(0.45, 0.90, size=count),
                "irrigation_hours": rng.uniform(0.0, 120.0, size=count),
            }
        )
        frame["yield"] = (
            2.0
            + 0.004 * frame["rainfall_mm"]
            + 0.002 * frame["growing_degree_days"]
            + 0.6 * frame["soil_organic_matter_pct"]
            + 6.0 * frame["ndvi_peak"]
            + 0.01 * frame["irrigation_hours"]
            + rng.normal(0.0, 0.4, size=count)
        )
        return frame

    training_fields = _synthetic_fields(120)
    validation_fields = _synthetic_fields(30)

    model = CropYieldModel(crop_type="wheat", model_type="machine_learning")
    model.fit(
        {"field_data": training_fields},
        target_column="yield",
        feature_columns=feature_columns,
    )
    prediction = model.predict({"field_data": validation_fields})
    importances = model.get_feature_importance()
    summary = prediction["summary"]

    return {
        "operation": "crop_yield_train_predict_and_rank",
        "crop_type": model.crop_type,
        "model_type": model.model_type,
        "n_training_fields": int(len(training_fields)),
        "n_validation_fields": int(len(validation_fields)),
        "predicted_mean_yield_t_ha": round(float(summary["mean_yield"]), 4),
        "predicted_min_yield_t_ha": round(float(summary["min_yield"]), 4),
        "predicted_max_yield_t_ha": round(float(summary["max_yield"]), 4),
        "predicted_yield_std": round(float(summary["std_yield"]), 4),
        "top_feature_importances": {
            name: round(float(score), 4)
            for name, score in list(importances.items())[:3]
        },
    }


if __name__ == "__main__":
    sys.exit(run_module_orchestrator("AG", _operation))
