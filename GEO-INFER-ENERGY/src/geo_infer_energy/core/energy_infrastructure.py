"""Energy infrastructure planning module."""

import logging
import xarray as xr

logger = logging.getLogger(__name__)


class EnergyInfrastructurePlanner:
    """Plan energy infrastructure siting."""

    def __init__(self, config: dict | None = None):
        """Initialize infrastructure planner.

        Documented contract: ``config`` is accepted and stored for API
        stability but its contents are currently ignored by this planner
        (reserved argument; only ``WindAnalyzer`` consumes config keys).
        """
        self.config = config or {}

    def optimize_facility_siting(
        self,
        resource_potential: xr.DataArray,
        demand_centers: xr.DataArray,
        constraints: xr.DataArray | None = None,
    ) -> xr.Dataset:
        """
        Optimize energy facility siting.

        Args:
            resource_potential: Resource potential map
            demand_centers: Demand center locations

        Returns:
            Optimal siting analysis
        """
        # Resource suitability, guarded against an all-zero potential map
        # (the +1e-10 keeps the ratio finite instead of all-NaN).
        resource_suitability = resource_potential / (resource_potential.max() + 1e-10)

        # Demand density index: demand_centers normalized by its own maximum.
        # This is a demand-density proximity proxy, not a geographic distance;
        # true distance-to-demand requires lat/lon coordinates on the grid.
        demand_density = demand_centers / (demand_centers.max() + 1e-10)

        # Combined suitability
        suitability = resource_suitability * 0.6 + demand_density * 0.4

        if constraints is not None:
            suitability = suitability * constraints

        # Identify optimal sites (top 10%)
        threshold = suitability.quantile(0.9)
        optimal_sites = suitability >= threshold

        return xr.Dataset(
            {
                "suitability": suitability,
                "optimal_sites": optimal_sites,
                "resource_suitability": resource_suitability,
                "demand_density": demand_density,
            }
        )

    def assess_infrastructure_capacity(
        self,
        current_capacity: xr.DataArray,
        projected_demand: xr.DataArray,
        years: int = 10,
    ) -> xr.Dataset:
        """
        Assess infrastructure capacity needs.

        Args:
            current_capacity: Current infrastructure capacity
            projected_demand: Projected future demand
            years: Planning horizon in years

        Returns:
            Capacity assessment
        """
        demand_growth = (projected_demand - current_capacity) / years
        required_capacity = projected_demand
        capacity_gap = required_capacity - current_capacity

        return xr.Dataset(
            {
                "current_capacity": current_capacity,
                "required_capacity": required_capacity,
                "capacity_gap": capacity_gap,
                "annual_growth_needed": demand_growth,
            }
        )
