#!/usr/bin/env python3
"""GEO-INFER-TEST module orchestrator.

Runs one documented end-to-end TEST operation: exercise the module's own
synthetic validation helpers (finite-array, probability, stochastic-matrix,
and seed-replay assertions) and the real validator suite — data-quality,
spatial, IoT, Bayesian, and performance validators coordinated by the
``QualityController`` — against deterministic synthetic data.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_ORCHESTRATORS_DIR = Path(__file__).resolve().parents[2]
if str(_ORCHESTRATORS_DIR) not in sys.path:
    sys.path.insert(0, str(_ORCHESTRATORS_DIR))

from _lib import run_module_orchestrator  # noqa: E402


def _operation() -> dict[str, Any]:
    import h3

    from geo_infer_test import (
        QualityController,
        assert_probability,
        assert_seed_replay,
        assert_stochastic_matrix,
    )
    from geo_infer_test import as_finite_array

    rng = np.random.default_rng(42)

    # 1. Module's own synthetic validation helpers.
    logits = np.round(rng.normal(0.0, 1.0, 5), 4)
    probabilities = assert_probability(
        np.exp(logits) / np.exp(logits).sum(), name="synthetic_posterior"
    )
    transitions = assert_stochastic_matrix(
        rng.dirichlet(np.ones(4), size=4), axis=1, name="synthetic_transitions"
    )

    def _seeded_factory(seed: int) -> np.ndarray:
        draw_rng = np.random.default_rng(seed)
        return as_finite_array(
            draw_rng.normal(0.0, 1.0, 16), name="seeded_model_output"
        )

    assert_seed_replay(_seeded_factory, seed=42)
    helpers_passed = True

    # 2. Synthetic IoT sensor panel: 30 recent readings, one planted anomaly.
    n_rows = 30
    now = pd.Timestamp.now(tz="UTC")
    sensor_df = pd.DataFrame(
        {
            "sensor_id": [f"sensor_{i % 3 + 1:03d}" for i in range(n_rows)],
            "timestamp": [now - pd.Timedelta(minutes=5 * i) for i in range(n_rows)],
            "radiation_level": np.round(
                np.clip(rng.normal(40.0, 5.0, n_rows), 0.0, 100.0), 3
            ),
        }
    )
    sensor_df.loc[7, "radiation_level"] = 99.0  # planted statistical anomaly

    # 3. Synthetic spatial records with real H3 cells.
    lats = np.round(44.0 + rng.uniform(0.0, 0.4, 12), 6)
    lons = np.round(-124.0 + rng.uniform(0.0, 0.4, 12), 6)
    spatial_df = pd.DataFrame(
        {
            "latitude": lats,
            "longitude": lons,
            "h3_index": [
                h3.latlng_to_cell(lat, lon, 8) for lat, lon in zip(lats, lons)
            ],
        }
    )

    # 4. Synthetic Bayesian inference results.
    inference_results = {
        "converged": True,
        "predictions": np.round(rng.normal(10.0, 2.0, 40), 3).tolist(),
        "uncertainty": np.abs(np.round(rng.normal(1.0, 0.3, 40), 3)).tolist(),
        "processing_time": 1.7,
    }

    # 5. Synthetic performance metrics.
    performance_metrics = {
        "inference_time": 2.4,
        "accuracy": 0.93,
        "memory_usage": 512 * 1024 * 1024,
    }

    # 6. Real comprehensive validation through the QualityController.
    controller = QualityController()
    summary = controller.run_comprehensive_validation(
        sensor_data=sensor_df,
        spatial_results={"h3_aggregated_data": spatial_df},
        inference_results=inference_results,
        performance_metrics=performance_metrics,
    )

    iot_results = summary["iot_validation"]
    anomaly_counts = iot_results["sensor_validation"]["anomaly_detection"].get(
        "anomaly_counts", {}
    )
    spatial_results = summary["spatial_validation"]["spatial_validation"]
    bayes_quality = summary["bayesian_validation"]["overall_quality"]
    perf_overall = summary["performance_validation"]["performance_validation"][
        "overall_performance"
    ]
    overall = summary["overall_results"]
    system_quality = (
        overall.get("system_quality") if isinstance(overall, dict) else str(overall)
    )

    return {
        "operation": "synthetic_validation_suite",
        "helpers_passed": helpers_passed,
        "posterior_sum": round(float(probabilities.sum()), 6),
        "posterior_max": round(float(probabilities.max()), 6),
        "transition_matrix_rows": int(transitions.shape[0]),
        "iot_total_sensors": iot_results["total_sensors"],
        "iot_anomaly_counts": anomaly_counts,
        "spatial_valid_coordinates": spatial_results["coordinate_validity"].get(
            "valid_coordinates"
        ),
        "spatial_valid_h3_indices": spatial_results["h3_validation"].get(
            "valid_h3_indices"
        ),
        "bayesian_overall_quality": bayes_quality,
        "performance_overall": perf_overall,
        "components_validated": summary["components_validated"],
        "system_quality": system_quality,
        "total_validation_time_seconds": round(summary["total_validation_time"], 4),
    }


if __name__ == "__main__":
    sys.exit(run_module_orchestrator("TEST", _operation))
