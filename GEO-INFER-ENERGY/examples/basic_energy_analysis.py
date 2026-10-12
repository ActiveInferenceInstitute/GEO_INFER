"""
Basic energy analysis example.

Demonstrates renewable resource assessment, energy demand forecasting,
and grid optimization using the real GEO-INFER-ENERGY public API.
"""

import numpy as np
import xarray as xr

from geo_infer_energy import (
    RenewableResourceAssessor,
    EnergyDemandForecaster,
    EnergyGridOptimizer,
)


def main():
    """Run basic energy analysis example."""
    print("GEO-INFER-ENERGY: Basic Energy Analysis Example")
    print("=" * 50)

    resource_assessor = RenewableResourceAssessor()
    demand_forecaster = EnergyDemandForecaster()
    grid_optimizer = EnergyGridOptimizer()

    # 1. Assess solar potential from an irradiance raster (kWh/m^2/day)
    print("\n1. Assessing Solar Potential")
    print("-" * 30)
    irradiance = xr.DataArray(
        np.linspace(4.0, 6.5, 100).reshape(10, 10),
        dims=("y", "x"),
    )
    solar_result = resource_assessor.assess_solar_potential(irradiance)
    print(
        f"Mean annual potential (kWh/m^2/yr): {float(solar_result['solar_potential'].mean()):.0f}"
    )

    # 2. Forecast energy demand from a historical series
    print("\n2. Forecasting Energy Demand")
    print("-" * 30)
    years = np.arange(2010, 2025)
    historical_demand = xr.DataArray(
        100.0 + 2.0 * (years - 2010),
        dims=["time"],
        coords={"time": years.astype(float)},
    )
    temperature = xr.DataArray(
        np.full(15, 18.0) + np.linspace(-1.0, 1.0, 15),
        dims=["time"],
        coords={"time": years.astype(float)},
    )
    population = xr.DataArray(
        800.0 * (1.01 ** np.arange(15)),
        dims=["time"],
        coords={"time": years.astype(float)},
    )
    demand_result = demand_forecaster.forecast_demand(
        historical_demand,
        temperature=temperature,
        population=population,
        forecast_years=5,
    )
    print(f"Forecast (5 years): {np.round(demand_result['demand_forecast'].values, 1)}")

    # 3. Grid balance between supply and demand rasters
    print("\n3. Optimizing Energy Grid")
    print("-" * 30)
    demand_map = xr.DataArray(np.full((10, 10), 80.0), dims=("y", "x"))
    supply_map = xr.DataArray(np.full((10, 10), 100.0), dims=("y", "x"))
    grid_result = grid_optimizer.optimize_grid_network(
        demand=demand_map, supply=supply_map
    )
    print(f"Mean surplus (MW): {float(grid_result['surplus'].mean()):.1f}")

    print("\n" + "=" * 50)
    print("Example completed successfully!")


if __name__ == "__main__":
    main()
