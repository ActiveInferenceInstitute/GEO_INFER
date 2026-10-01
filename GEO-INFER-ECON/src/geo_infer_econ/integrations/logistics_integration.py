"""
Logistics integration for GEO-INFER-ECON.

Bridges economic supply/demand models with the logistics capabilities
in GEO-INFER-LOG (supply chain optimization, facility location, and
inventory management).

Gracefully degrades when GEO-INFER-LOG is not installed.
"""

import logging
from typing import Any, cast

logger = logging.getLogger(__name__)

# Optional dependency on GEO-INFER-LOG
try:
    from geo_infer_log.core.supply_chain import (
        SupplyChainModel,
        NetworkOptimizer,
        FacilityLocator,
        InventoryManager,
    )
    from geo_infer_log.core.transport import EmissionsCalculator

    HAS_LOG = True
except ImportError:
    HAS_LOG = False
    logger.info("GEO-INFER-LOG not available; logistics integration disabled")


class LogisticsEconomicAnalyzer:
    """
    Connects ECON economic models with LOG logistics optimization.

    Provides combined cost analysis that accounts for both economic
    factors (demand, pricing, market structure) and logistics factors
    (transport cost, facility location, inventory holding costs).

    Falls back to simplified estimates when GEO-INFER-LOG is not
    installed.

    Example:
        >>> analyzer = LogisticsEconomicAnalyzer()
        >>> result = analyzer.total_cost_analysis(
        ...     facilities=[{"id": "f1", "location": (13.4, 52.5), "fixed_cost": 100000}],
        ...     demand_points=[{"id": "d1", "location": (13.5, 52.6), "demand": 500}],
        ...     lead_times={"f1": 3},
        ... )
        >>> result["logistics_cost"]
    """

    def __init__(self) -> None:
        self._supply_chain: Any | None = None
        self._facility_locator: Any | None = None
        self._inventory_mgr: Any | None = None
        self._emissions_calc: Any | None = None

        if HAS_LOG:
            self._supply_chain = SupplyChainModel()
            self._facility_locator = FacilityLocator()
            self._inventory_mgr = InventoryManager()
            self._emissions_calc = EmissionsCalculator()
            logger.info("LogisticsEconomicAnalyzer initialized with GEO-INFER-LOG")
        else:
            logger.warning(
                "LogisticsEconomicAnalyzer running without GEO-INFER-LOG; "
                "install with: uv pip install -e ../GEO-INFER-LOG"
            )

    @property
    def log_available(self) -> bool:
        """Whether GEO-INFER-LOG is available."""
        return HAS_LOG

    def total_cost_analysis(
        self,
        facilities: list[dict[str, Any]],
        demand_points: list[dict[str, Any]],
        lead_times: dict[str, int],
        demand_data: dict[str, list[float]] | None = None,
        service_level: float = 0.95,
        max_distance_km: float = 100.0,
    ) -> dict[str, Any]:
        """
        Compute total landed cost combining logistics and economic factors.

        Args:
            facilities: Facility locations with ``id``, ``location``,
                ``fixed_cost``, ``ordering_cost``, ``holding_cost``.
            demand_points: Customer demand points with ``id``, ``location``,
                ``demand``.
            lead_times: Mapping of facility ID → lead time in periods.
            demand_data: Historical demand series per facility for
                inventory optimization (optional).
            service_level: Target fill rate (default 95%).
            max_distance_km: Maximum service distance in km.

        Returns:
            Dictionary with ``logistics_cost``, ``inventory_cost``,
            ``coverage``, ``facility_cost``, ``total_landed_cost``.
        """
        result: dict[str, Any] = {
            "logistics_cost": 0.0,
            "inventory_cost": 0.0,
            "facility_cost": sum(f.get("fixed_cost", 0) for f in facilities),
            "coverage": {},
            "total_landed_cost": 0.0,
        }

        if not HAS_LOG or self._facility_locator is None:
            # Simplified fallback without LOG
            result["total_landed_cost"] = result["facility_cost"]
            result["_warning"] = (
                "GEO-INFER-LOG not available; logistics estimates omitted"
            )
            return result

        # Coverage analysis
        assert self._facility_locator is not None
        coverage = self._facility_locator.analyze_coverage(
            facilities=facilities,
            demand_points=demand_points,
            max_distance=max_distance_km,
        )
        result["coverage"] = coverage
        result["logistics_cost"] = coverage.get("average_distance", 0.0) * sum(
            dp.get("demand", 1) for dp in demand_points
        )

        # Inventory optimization (if data provided)
        if demand_data and self._inventory_mgr is not None:
            inv_result = self._inventory_mgr.optimize_inventory(
                facilities=facilities,
                demand_data=demand_data,
                lead_times=lead_times,
                service_level=service_level,
            )
            result["inventory_cost"] = inv_result.get("total_inventory_cost", 0.0)
            result["inventory_detail"] = inv_result

        result["total_landed_cost"] = (
            result["facility_cost"]
            + result["logistics_cost"]
            + result["inventory_cost"]
        )

        logger.info(
            "Total cost analysis: facilities=%d, demand=%d, total=%.0f",
            len(facilities),
            len(demand_points),
            result["total_landed_cost"],
        )
        return result

    def optimal_network_design(
        self,
        candidate_locations: list[dict[str, Any]],
        demand_points: list[dict[str, Any]],
        max_facilities: int = 5,
        budget: float = float("inf"),
        max_distance_km: float = 100.0,
    ) -> dict[str, Any]:
        """
        Design an optimal logistics network using p-median optimization.

        Delegates to LOG's ``NetworkOptimizer`` for facility location
        and then enriches the result with economic cost metrics.

        Args:
            candidate_locations: Potential facility sites.
            demand_points: Customer demand points.
            max_facilities: Maximum number of facilities to open.
            budget: Budget constraint for facility fixed costs.
            max_distance_km: Maximum service distance.

        Returns:
            Dictionary with selected facilities, links, service level,
            and total cost breakdown.
        """
        if not HAS_LOG:
            return {"error": "GEO-INFER-LOG required for network design"}

        optimizer = NetworkOptimizer()
        result = optimizer.optimize_network(
            locations=candidate_locations,
            demand_points=demand_points,
            constraints={
                "max_facilities": max_facilities,
                "budget": budget,
                "max_distance": max_distance_km,
            },
        )
        # Enrich with economic metrics
        result["demand_served"] = sum(dp.get("demand", 1) for dp in demand_points)
        result["cost_per_unit_demand"] = result.get("total_cost", 0) / max(
            result["demand_served"], 1
        )
        return cast(dict[str, Any], result)
