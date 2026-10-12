#!/usr/bin/env python3
"""GEO-INFER-HEALTH module orchestrator.

Runs one documented end-to-end HEALTH operation on synthetic data: build a
synthetic disease-report surveillance set with two spatial clusters, detect
case hotspots, compute the local incidence rate against a population
estimate, and run the module's SIR outbreak simulation. All work goes
through the real ``geo_infer_health`` public API.
"""

from __future__ import annotations

import sys
from datetime import datetime
from typing import Any

from geo_infer_examples.orchestration import run_module_orchestrator


def _operation() -> dict[str, Any]:
    from geo_infer_health import (
        DiseaseHotspotAnalyzer,
        DiseaseReport,
        Location,
        PopulationData,
    )

    # Synthetic surveillance set: a dense harbor-district cluster and a
    # sparse inland cluster, reported over June 2026 with fixed offsets.
    cluster_harbor: list[tuple[float, float, int, int]] = [
        (41.7542, -124.2010, 4, 1),
        (41.7551, -124.1994, 3, 2),
        (41.7533, -124.2022, 5, 4),
        (41.7560, -124.2001, 2, 6),
        (41.7528, -124.1988, 3, 8),
        (41.7547, -124.2031, 4, 11),
        (41.7555, -124.1979, 2, 13),
    ]
    cluster_inland: list[tuple[float, float, int, int]] = [
        (41.7731, -124.1655, 1, 3),
        (41.7748, -124.1621, 1, 9),
        (41.7712, -124.1689, 2, 15),
    ]

    reports: list[DiseaseReport] = []
    for report_id, (lat, lng, case_count, day) in enumerate(
        cluster_harbor + cluster_inland, start=1
    ):
        reports.append(
            DiseaseReport(
                report_id=f"RPT-{report_id:03d}",
                disease_code="A90",
                location=Location(latitude=lat, longitude=lng),
                report_date=datetime(2026, 6, day),
                case_count=case_count,
                source="synthetic-clinic",
            )
        )

    # A single unlocated population area is the module's documented coarse
    # regional estimate for incidence-rate denominators.
    population = [PopulationData(area_id="synthetic-district", population_count=25000)]

    analyzer = DiseaseHotspotAnalyzer(reports=reports, population_data=population)

    hotspots = analyzer.identify_simple_hotspots(
        threshold_case_count=10, scan_radius_km=2.5
    )
    harbor_center = Location(latitude=41.7547, longitude=-124.2007)
    incidence_rate, total_cases, est_population, pop_estimated = (
        analyzer.calculate_local_incidence_rate(
            harbor_center, radius_km=2.5, time_window_days=30
        )
    )
    cases_near_center = len(analyzer.get_cases_in_radius(harbor_center, radius_km=2.5))
    sir = analyzer.simulate_sir_model(
        initial_infected=12, population=25000, beta=0.3, gamma=0.1, days=60
    )

    return {
        "operation": "outbreak_hotspot_incidence_and_sir",
        "n_reports": len(reports),
        "n_hotspots": len(hotspots),
        "hotspot_case_counts": [int(hs["case_count"]) for hs in hotspots],
        "cases_within_2_5km_of_center": cases_near_center,
        "incidence_rate_per_100k": round(float(incidence_rate), 3),
        "total_cases_in_radius": int(total_cases),
        "estimated_population": int(est_population),
        "population_estimated": bool(pop_estimated),
        "sir_basic_reproduction_number": round(
            float(sir["basic_reproduction_number"]), 4
        ),
        "sir_peak_infected": round(float(sir["peak_infected"]), 2),
        "sir_peak_day": int(sir["peak_day"]),
    }


if __name__ == "__main__":
    sys.exit(run_module_orchestrator("HEALTH", _operation))
