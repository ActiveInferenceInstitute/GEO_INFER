"""Integration tests for GEO-INFER-LOG."""

from geo_infer_log import FacilityLocator, InventoryManager
from geo_infer_log.utils.geo import haversine_distance


class TestLogIntegration:
    """Cross-layer integration through the public package surface."""

    def test_module_integration(self) -> None:
        """Facility location feeds downstream inventory optimization."""
        candidates = [
            {"id": "FAC_A", "location": (2.3522, 48.8566)},  # Paris center
            {"id": "FAC_B", "location": (2.2945, 48.8584)},  # West Paris
            {"id": "FAC_C", "location": (4.8357, 45.7640)},  # Lyon
        ]
        demand_points = [
            {
                "id": f"D{i}",
                "location": (2.30 + i * 0.01, 48.85 + i * 0.01),
                "demand": 10 + i,
            }
            for i in range(4)
        ]

        locator = FacilityLocator()
        selected = locator.locate_facilities(
            candidates, demand_points, num_facilities=1, max_distance=50.0
        )

        assert len(selected) == 1
        # The exact p-median must pick a candidate
        # within the max_distance constraint of the clustered demand points.
        facility = selected[0]
        assert all(
            haversine_distance(tuple(demand["location"]), tuple(facility["location"]))
            <= 50.0
            for demand in demand_points
        )

        coverage = locator.analyze_coverage(selected, demand_points, max_distance=50.0)
        assert coverage["coverage_ratio"] > 0
        inventory = InventoryManager()
        result = inventory.optimize_inventory(
            facilities=selected,
            demand_data={
                "FAC_A": [40, 42, 38, 45],
                "FAC_B": [30, 28, 33],
                "FAC_C": [20, 22],
            },
            lead_times={"FAC_A": 3, "FAC_B": 5, "FAC_C": 4},
            service_level=0.95,
        )
        assert isinstance(result, dict)
