"""Energy grid optimization module."""

import logging
import xarray as xr

logger = logging.getLogger(__name__)


class EnergyGridOptimizer:
    """Optimize energy grid networks."""

    def __init__(self, config: dict | None = None):
        """Initialize grid optimizer.

        Documented contract: ``config`` is accepted and stored for API
        stability but its contents are currently ignored by this optimizer
        (reserved argument; only ``WindAnalyzer`` consumes config keys).
        """
        self.config = config or {}

    def optimize_grid_network(
        self,
        demand: xr.DataArray,
        supply: xr.DataArray,
        transmission_capacity: xr.DataArray | None = None,
    ) -> xr.Dataset:
        """
        Optimize energy grid network.

        Args:
            demand: Energy demand
            supply: Energy supply
            transmission_capacity: Optional per-cell transmission capacity;
                when provided, surplus and deficit are clipped to this limit
                (power that cannot be transmitted over the grid).

        Returns:
            Grid optimization results
        """
        # Calculate supply-demand balance
        balance = supply - demand

        # Identify deficits and surpluses
        deficit = xr.where(balance < 0, -balance, 0)
        surplus = xr.where(balance > 0, balance, 0)

        # Apply transmission limits: flow beyond capacity cannot be used,
        # so both surplus export and deficit import are clipped.
        if transmission_capacity is not None:
            surplus = xr.where(
                surplus > transmission_capacity, transmission_capacity, surplus
            )
            deficit = xr.where(
                deficit > transmission_capacity, transmission_capacity, deficit
            )

        # Grid reliability
        reliability = supply / (demand + 1e-10)
        reliability = xr.where(reliability > 1, 1, reliability)

        return xr.Dataset(
            {
                "balance": balance,
                "deficit": deficit,
                "surplus": surplus,
                "reliability": reliability,
            }
        )

    def assess_grid_reliability(
        self,
        generation_capacity: xr.DataArray,
        peak_demand: xr.DataArray,
        reserve_margin: float = 0.15,
    ) -> xr.Dataset:
        """
        Assess grid reliability.

        Args:
            generation_capacity: Total generation capacity
            peak_demand: Peak demand
            reserve_margin: Required reserve margin (0-1)

        Returns:
            Reliability assessment
        """
        required_capacity = peak_demand * (1 + reserve_margin)
        adequacy = generation_capacity / (required_capacity + 1e-10)

        # Reliability index (1 = adequate, <1 = inadequate)
        reliability = xr.where(adequacy >= 1, 1, adequacy)

        return xr.Dataset(
            {
                "required_capacity": required_capacity,
                "adequacy": adequacy,
                "reliability_index": reliability,
                "capacity_deficit": xr.where(
                    adequacy < 1, required_capacity - generation_capacity, 0
                ),
            }
        )
