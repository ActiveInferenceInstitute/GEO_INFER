#!/usr/bin/env python3
"""GEO-INFER-RISK module orchestrator.

Runs one documented end-to-end RISK operation on synthetic data: assemble a
synthetic H3 cell set over a coastal study area, then assess single-hazard
flood risk through the exported core risk API — ``FloodHazardModel``,
``BuildingVulnerabilityModel``, and ``PopulationExposureModel`` combined by
``RiskModel.calculate_risk`` and stress-tested with the seeded Monte Carlo
simulation. All work goes through the real ``geo_infer_risk`` public API.
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
    import geopandas as gpd
    import h3
    import numpy as np
    from shapely.geometry import Polygon

    from geo_infer_risk import RiskModel
    from geo_infer_risk.core.risk_models import (
        BuildingVulnerabilityModel,
        FloodHazardModel,
        PopulationExposureModel,
        RiskParameters,
    )

    rng = np.random.default_rng(42)

    # Synthetic H3 cells at resolution 8 over a fictional coastal district.
    resolution = 8
    lats = np.linspace(41.72, 41.78, 6)
    lngs = np.linspace(-124.22, -124.16, 6)
    cells = sorted(
        {
            h3.latlng_to_cell(float(lat), float(lng), resolution)
            for lat in lats
            for lng in lngs
        }
    )
    polygons: list[Polygon] = []
    for cell in cells:
        boundary = h3.cell_to_boundary(cell)
        polygons.append(Polygon([(lng, lat) for lat, lng in boundary]))

    frame = gpd.GeoDataFrame(
        {
            "h3_index": cells,
            "population": rng.integers(200, 2000, size=len(cells)),
            "building_material": [
                str(material)
                for material in rng.choice(
                    ["wood", "masonry", "reinforced_concrete"], size=len(cells)
                )
            ],
            "building_age": rng.integers(5, 80, size=len(cells)),
            "building_condition": rng.integers(1, 6, size=len(cells)),
        },
        geometry=polygons,
        crs="EPSG:4326",
    )

    parameters = RiskParameters(
        confidence_level=0.95, monte_carlo_iterations=400, random_seed=42
    )
    risk_model = RiskModel(parameters)
    risk_model.set_hazard(FloodHazardModel(return_period=100, base_depth_m=2.5))
    risk_model.set_vulnerability(BuildingVulnerabilityModel())
    risk_model.set_exposure(PopulationExposureModel(income_per_capita=42000.0))

    assessed = risk_model.calculate_risk(frame)
    monte_carlo = risk_model.run_monte_carlo(frame)

    risk_scores = assessed["risk_score"].to_numpy(dtype=float)
    worst_position = int(risk_scores.argmax())
    worst_cell = str(assessed.iloc[worst_position]["h3_index"])

    return {
        "operation": "single_hazard_flood_risk_assessment_on_h3_cells",
        "hazard_type": "flood",
        "return_period_years": 100,
        "h3_resolution": resolution,
        "n_cells": len(cells),
        "mean_risk_score": round(float(risk_scores.mean()), 6),
        "max_risk_score": round(float(risk_scores.max()), 6),
        "highest_risk_cell": worst_cell,
        "monte_carlo_iterations": int(parameters.monte_carlo_iterations),
        "monte_carlo_mean_risk": round(
            float(np.asarray(monte_carlo["mean"]).mean()), 6
        ),
        "monte_carlo_ci95_upper_mean": round(
            float(np.asarray(monte_carlo["upper_bound"]).mean()), 6
        ),
    }


if __name__ == "__main__":
    sys.exit(run_module_orchestrator("RISK", _operation))
