"""Water infrastructure planning module."""

import logging
import xarray as xr

logger = logging.getLogger(__name__)


class WaterInfrastructurePlanner:
    """Plan water infrastructure."""

    def __init__(self, config: dict | None = None):
        """Initialize infrastructure planner."""
        self.config = config or {}

    def optimize_water_allocation(
        self,
        water_supply: xr.DataArray,
        water_demand: xr.DataArray,
        priorities: xr.DataArray | None = None,
    ) -> xr.Dataset:
        """
        Optimize water allocation.

        Args:
            water_supply: Available water supply
            water_demand: Water demand
            priorities: Optional priority weights

        Returns:
            Allocation optimization results
        """
        # Calculate supply-demand ratio
        supply_demand_ratio = water_supply / (water_demand + 1e-10)

        # Priority-weighted allocation: when supply is scarce, each demander
        # receives a share of the supply proportional to its weighted need
        # (demand * priority), capped at its demand so no demander is
        # over-allocated. Default priority is 1.0 (equal weighting).
        if priorities is not None:
            weights = priorities
        else:
            weights = xr.ones_like(water_demand)

        weighted_need = water_demand * weights
        total_weighted_need = weighted_need.sum()
        # Allocate supply in proportion to weighted need.
        raw_allocation = water_supply * weighted_need / (total_weighted_need + 1e-10)
        # Cap at demand (no over-allocation); surplus is left unallocated.
        allocation = xr.where(
            raw_allocation > water_demand, water_demand, raw_allocation
        )
        allocated_surplus = water_supply - allocation.sum()
        # Redistribute any surplus left after capping proportionally to the
        # unmet weighted need of demanders that still have a shortfall.
        if float(allocated_surplus) > 1e-10:
            remaining_need = (water_demand - allocation) * weights
            remaining_total = remaining_need.sum()
            top_up = xr.where(
                remaining_total > 1e-10,
                allocated_surplus * remaining_need / (remaining_total + 1e-10),
                0.0,
            )
            allocation = allocation + xr.where(
                top_up > remaining_need, remaining_need, top_up
            )

        # Shortage is the unmet demand after allocation.
        shortage = xr.where(water_demand > allocation, water_demand - allocation, 0)

        return xr.Dataset(
            {
                "allocation": allocation,
                "shortage": shortage,
                "supply_demand_ratio": supply_demand_ratio,
                "adequacy": xr.where(supply_demand_ratio >= 1, 1, supply_demand_ratio),
            }
        )

    def assess_infrastructure_needs(
        self, current_capacity: xr.DataArray, projected_demand: xr.DataArray
    ) -> xr.Dataset:
        """
        Assess water infrastructure capacity needs.

        Args:
            current_capacity: Current infrastructure capacity
            projected_demand: Projected future demand

        Returns:
            Infrastructure needs assessment
        """
        capacity_gap = projected_demand - current_capacity
        capacity_gap = xr.where(capacity_gap < 0, 0, capacity_gap)

        adequacy = current_capacity / (projected_demand + 1e-10)

        return xr.Dataset(
            {
                "capacity_gap": capacity_gap,
                "adequacy": adequacy,
                "expansion_needed": capacity_gap > 0,
            }
        )
