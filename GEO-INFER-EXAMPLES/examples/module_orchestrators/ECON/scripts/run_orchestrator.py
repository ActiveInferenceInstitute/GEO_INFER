#!/usr/bin/env python3
"""GEO-INFER-ECON module orchestrator.

Runs one documented end-to-end ECON operation on synthetic data: value the
ecosystem-service portfolios of three synthetic bioregions with the
benefit-transfer ``EcosystemServicesValuation`` engine (per-hectare unit
values adjusted for biome quality), then compare annual values and net
present values across regions. All work goes through the real
``geo_infer_econ`` public API.
"""

from __future__ import annotations

import sys
from typing import Any

from geo_infer_examples.orchestration import run_module_orchestrator


def _operation() -> dict[str, Any]:
    import numpy as np
    from geo_infer_econ import EcosystemServicesValuation

    rng = np.random.default_rng(11)
    valuator = EcosystemServicesValuation(
        config={"discount_rate": 0.03, "time_horizon": 30}
    )

    # Same TEEB service catalog assessed over three synthetic bioregions;
    # areas and biome-quality factors differ per region.
    service_catalog: list[tuple[str, str]] = [
        ("provisioning", "water"),
        ("regulating", "climate_regulation"),
        ("regulating", "water_purification"),
        ("supporting", "habitat"),
        ("cultural", "recreation"),
    ]
    regions = ("fir_river_watershed", "oak_savanna", "coastal_dunes")

    regional_results: dict[str, dict[str, float]] = {}
    for region in regions:
        services = [
            {
                "category": category,
                "type": service_type,
                "area_ha": float(rng.uniform(500.0, 4000.0)),
                "quality_factor": float(rng.uniform(0.6, 1.0)),
            }
            for category, service_type in service_catalog
        ]
        regional_results[region] = valuator.value_services(services)

    annual_by_region = {
        region: round(float(values["total_annual"]), 2)
        for region, values in regional_results.items()
    }
    npv_by_region = {
        region: round(float(values["total_npv"]), 2)
        for region, values in regional_results.items()
    }
    best_region = max(annual_by_region, key=lambda name: annual_by_region[name])

    return {
        "operation": "ecosystem_service_valuation_across_regions",
        "discount_rate": valuator.discount_rate,
        "time_horizon_years": valuator.time_horizon,
        "n_regions": len(regions),
        "annual_value_usd_per_region": annual_by_region,
        "npv_usd_per_region": npv_by_region,
        "highest_value_region": best_region,
        "highest_value_region_breakdown": regional_results[best_region],
    }


if __name__ == "__main__":
    sys.exit(run_module_orchestrator("ECON", _operation))
