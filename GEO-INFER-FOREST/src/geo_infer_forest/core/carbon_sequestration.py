"""Carbon sequestration modeling."""

import logging
import xarray as xr

logger = logging.getLogger(__name__)


class CarbonSequestrationModeler:
    """Model carbon sequestration in forests."""

    def __init__(self, config: dict | None = None):
        """Initialize carbon sequestration modeler."""
        self.config = config or {}
        # Carbon content: ~50% of dry biomass
        self.carbon_fraction = 0.5

    def calculate_carbon_stock(self, biomass: xr.DataArray) -> xr.DataArray:
        """
        Calculate carbon stock from biomass.

        Args:
            biomass: Forest biomass (tons/ha)

        Returns:
            Carbon stock (tons C/ha)
        """
        carbon_stock = biomass * self.carbon_fraction
        return carbon_stock

    def estimate_sequestration_rate(
        self,
        biomass_growth: xr.DataArray,
        time_period: float = 1.0,  # years
    ) -> xr.DataArray:
        """
        Estimate carbon sequestration rate.

        Args:
            biomass_growth: Annual biomass growth (tons/ha/year)
            time_period: Time period in years

        Returns:
            Sequestration rate (tons C/ha/year)
        """
        carbon_sequestration = biomass_growth * self.carbon_fraction / time_period
        return carbon_sequestration

    def calculate_carbon_credits(
        self,
        carbon_sequestration: xr.DataArray,
        area: xr.DataArray,
        price_per_ton: float = 50.0,
    ) -> xr.DataArray:
        """
        Calculate carbon credit value.

        Args:
            carbon_sequestration: Carbon sequestration (tons C/ha/year)
            area: Area (ha)
            price_per_ton: Price per ton of CO2 equivalent

        Returns:
            Carbon credit value (USD/year)
        """
        # Convert C to CO2 (molecular weight ratio ~3.67)
        co2_equivalent = carbon_sequestration * 3.67
        total_co2 = co2_equivalent * area
        credit_value = total_co2 * price_per_ton
        return credit_value
