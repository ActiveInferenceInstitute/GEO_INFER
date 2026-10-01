#!/usr/bin/env python3
"""GEO-INFER-DATA module orchestrator.

Runs one documented end-to-end DATA operation on synthetic data: register a
synthetic sensor panel with the ``DataQualityManager``, execute the
comprehensive quality-validation workflow (completeness, accuracy,
consistency, validity, temporal, spatial, format, schema checks), and
compare coordinate-quality scores between a clean panel and a corrupted
one. All work goes through the real ``geo_infer_data`` public API.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ORCHESTRATORS_DIR = Path(__file__).resolve().parents[2]
if str(_ORCHESTRATORS_DIR) not in sys.path:
    sys.path.insert(0, str(_ORCHESTRATORS_DIR))

from _lib import run_module_orchestrator  # noqa: E402


def _operation() -> dict[str, Any]:
    import pandas as pd

    from geo_infer_data import DataQualityManager
    from geo_infer_data.models.schemas import DataLineage, DatasetMetadata

    rng = np.random.default_rng(42)

    # Synthetic sensor panel: 4 sensors, 12 hourly readings each, with a
    # controlled amount of missing data so the completeness check has
    # something real to score.
    n_rows = 48
    base = pd.Timestamp.now(tz=None).floor("h") - pd.Timedelta(hours=n_rows)
    frame = pd.DataFrame(
        {
            "sensor_id": [f"sensor_{i % 4 + 1:03d}" for i in range(n_rows)],
            "timestamp": [base + pd.Timedelta(hours=i) for i in range(n_rows)],
            "latitude": np.round(44.0 + rng.uniform(0.0, 0.5, n_rows), 6),
            "longitude": np.round(-124.0 + rng.uniform(0.0, 0.5, n_rows), 6),
            "temperature_c": np.round(10.0 + rng.normal(0.0, 2.0, n_rows), 3),
            "radiation_level": np.round(
                np.clip(rng.normal(45.0, 8.0, n_rows), 0.0, 100.0), 3
            ),
        }
    )
    missing_idx = rng.choice(n_rows, size=4, replace=False)
    frame.loc[missing_idx, "temperature_c"] = np.nan

    quality_manager = DataQualityManager(
        validation_rules="comprehensive",
        quality_threshold=0.8,
        real_time_monitoring=False,
    )
    metadata = DatasetMetadata(
        title="synthetic_sensor_panel",
        description="Synthetic hourly sensor readings for orchestrator demo",
        keywords=["synthetic", "sensors", "demo"],
        lineage=DataLineage(
            source="synthetic_rng_seed_42",
            process="orchestrator_synthetic_generation",
            created_by="run_orchestrator.py",
        ),
    )
    quality_manager.register_dataset("synthetic_sensor_panel", frame, metadata)

    report = asyncio.run(quality_manager.validate_dataset("synthetic_sensor_panel"))
    check_scores = {
        name: {"score": round(check.score, 4), "status": check.status.value}
        for name, check in report.checks.items()
    }

    # Corrupted counterpart: one clearly out-of-range coordinate, to show the
    # coordinate validator producing a genuinely different score.
    corrupted = frame.copy()
    corrupted.loc[0, "latitude"] = 123.4
    clean_coords = quality_manager.validator.validate_coordinates(frame)
    corrupted_coords = quality_manager.validator.validate_coordinates(corrupted)

    recommendations = quality_manager.get_improvement_recommendations(report)

    return {
        "operation": "data_quality_validation",
        "dataset_id": report.dataset_id,
        "rows": int(len(frame)),
        "rows_with_missing_values": int(frame.isna().any(axis=1).sum()),
        "overall_quality_score": round(report.overall_score, 4),
        "check_scores": check_scores,
        "recommendation_count": len(recommendations),
        "clean_coordinate_score": round(clean_coords.score, 4),
        "clean_coordinate_status": clean_coords.status.value,
        "corrupted_coordinate_score": round(corrupted_coords.score, 4),
        "corrupted_coordinate_status": corrupted_coords.status.value,
    }


if __name__ == "__main__":
    sys.exit(run_module_orchestrator("DATA", _operation))
