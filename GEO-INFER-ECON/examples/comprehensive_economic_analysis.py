"""
Comprehensive Economic Analysis Example

This example demonstrates the integration of microeconomic, macroeconomic,
and bioregional economic analysis capabilities within GEO-INFER-ECON.

The example models a regional economy with:
1. Microeconomic consumer and producer behavior
2. Macroeconomic growth dynamics
3. Bioregional ecosystem services and natural capital
"""

import geopandas as gpd


def create_sample_bioregion():
    """Create a sample bioregional economy for analysis"""

    # 1. Define bioregional boundary
    # In practice, this would be loaded from actual geographic data
    bioregion_coords = [
        (-120.5, 45.5),
        (-120.0, 45.5),
        (-120.0, 45.0),
        (-120.5, 45.0),
        (-120.5, 45.5),
    ]

    # Create sample GeoDataFrame for bioregion
    from shapely.geometry import Polygon

    bioregion_poly = Polygon(bioregion_coords)
    bioregion_gdf = gpd.GeoDataFrame(
        [{"region_id": "sample_bioregion"}], geometry=[bioregion_poly]
    )

    return bioregion_gdf


def analyze_consumer_behavior():
    """Demonstrate microeconomic consumer analysis"""

    print("=== MICROECONOMIC ANALYSIS: Consumer Behavior ===")

    # Sample consumer data
    consumers = {
        "urban_consumer": {
            "income": 75000.0,
            "location": (45.25, -120.25),
            "preferences": {
                "food": 0.3,
                "housing": 0.4,
                "transport": 0.2,
                "recreation": 0.1,
            },
        },
        "rural_consumer": {
            "income": 45000.0,
            "location": (45.1, -120.4),
            "preferences": {
                "food": 0.4,
                "housing": 0.3,
                "transport": 0.15,
                "recreation": 0.15,
            },
        },
    }

    # Market prices
    prices = {"food": 3.0, "housing": 1500.0, "transport": 0.5, "recreation": 25.0}

    # Analyze consumption patterns
    results = {}
    for consumer_id, consumer in consumers.items():
        # Simple Cobb-Douglas utility maximization
        expenditures = {}
        total_expenditure = 0

        for good, preference in consumer["preferences"].items():
            expenditure = preference * consumer["income"]
            quantity = expenditure / prices[good]
            expenditures[good] = {"expenditure": expenditure, "quantity": quantity}
            total_expenditure += expenditure

        results[consumer_id] = {
            "expenditures": expenditures,
            "total_expenditure": total_expenditure,
            "savings": consumer["income"] - total_expenditure,
        }

        print(f"\n{consumer_id}:")
        print(f"  Income: ${consumer['income']:,}")
        print(f"  Total expenditure: ${total_expenditure:,.2f}")
        print(f"  Savings: ${results[consumer_id]['savings']:,.2f}")

    return results


def analyze_regional_growth():
    """Demonstrate macroeconomic growth analysis"""

    print("\n=== MACROECONOMIC ANALYSIS: Regional Growth ===")

    # Regional parameters
    regions = {
        "urban_center": {
            "initial_gdp_per_capita": 50000,
            "population": 50000,
            "technology_level": 1.2,
            "institutions_quality": 0.8,
        },
        "rural_area": {
            "initial_gdp_per_capita": 30000,
            "population": 15000,
            "technology_level": 0.8,
            "institutions_quality": 0.6,
        },
    }

    # Solow growth model parameters
    alpha = 0.33  # Capital share
    s = 0.2  # Savings rate
    delta = 0.05  # Depreciation
    g = 0.02  # Technology growth

    results = {}
    years = 20

    for region_id, region in regions.items():
        # Adjust growth rate based on regional characteristics
        adjusted_s = s * region["institutions_quality"]
        adjusted_g = g * region["technology_level"]

        # Simple growth calculation
        growth_rate = adjusted_s * alpha + adjusted_g - delta
        final_gdp_per_capita = (
            region["initial_gdp_per_capita"] * (1 + growth_rate) ** years
        )

        results[region_id] = {
            "initial_gdp_per_capita": region["initial_gdp_per_capita"],
            "final_gdp_per_capita": final_gdp_per_capita,
            "growth_rate": growth_rate,
            "total_gdp": final_gdp_per_capita * region["population"],
        }

        print(f"\n{region_id}:")
        print(f"  Initial GDP per capita: ${region['initial_gdp_per_capita']:,}")
        print(f"  Final GDP per capita: ${final_gdp_per_capita:,.2f}")
        print(f"  Annual growth rate: {growth_rate * 100:.2f}%")
        print(f"  Total regional GDP: ${results[region_id]['total_gdp']:,.0f}")

    return results


def analyze_ecosystem_services():
    """Demonstrate bioregional ecosystem services analysis"""

    print("\n=== BIOREGIONAL ANALYSIS: Ecosystem Services ===")

    # Ecosystem assets
    assets = {
        "old_growth_forest": {
            "area_hectares": 5000,
            "carbon_storage_tons": 12500,
            "carbon_sequestration_per_year": 125,
            "biodiversity_index": 0.95,
            "recreation_value_per_hectare": 50,
        },
        "restored_wetland": {
            "area_hectares": 500,
            "carbon_storage_tons": 1500,
            "carbon_sequestration_per_year": 30,
            "water_purification_value": 300000,
            "flood_control_value": 1000000,
        },
        "regenerative_farm": {
            "area_hectares": 200,
            "carbon_storage_tons": 600,
            "carbon_sequestration_per_year": 20,
            "food_production_value": 100000,
            "pollination_value": 14000,
        },
    }

    # Ecosystem service prices
    carbon_price = 50  # $/ton CO2
    biodiversity_value = 100  # $/hectare/year

    total_annual_value = 0
    total_carbon_stock = 0

    for asset_id, asset in assets.items():
        # Calculate annual ecosystem service values
        carbon_flow_value = asset["carbon_sequestration_per_year"] * carbon_price

        if "biodiversity_index" in asset:
            biodiversity_value_annual = (
                asset["area_hectares"]
                * asset["biodiversity_index"]
                * biodiversity_value
            )
        else:
            biodiversity_value_annual = 0

        if "recreation_value_per_hectare" in asset:
            recreation_value = (
                asset["area_hectares"] * asset["recreation_value_per_hectare"]
            )
        else:
            recreation_value = 0

        # Direct economic values
        direct_values = sum(
            v
            for k, v in asset.items()
            if "value" in k and k != "recreation_value_per_hectare"
        )

        annual_value = (
            carbon_flow_value
            + biodiversity_value_annual
            + recreation_value
            + direct_values
        )
        total_annual_value += annual_value
        total_carbon_stock += asset["carbon_storage_tons"]

        print(f"\n{asset_id}:")
        print(f"  Area: {asset['area_hectares']} hectares")
        print(f"  Carbon sequestration value: ${carbon_flow_value:,.0f}/year")
        print(f"  Total annual ES value: ${annual_value:,.0f}")

    carbon_stock_value = total_carbon_stock * carbon_price

    print("\n=== Total Bioregional Values ===")
    print(f"Total annual ecosystem services: ${total_annual_value:,.0f}")
    print(f"Total carbon stock value: ${carbon_stock_value:,.0f}")

    return {
        "annual_service_value": total_annual_value,
        "carbon_stock_value": carbon_stock_value,
        "total_carbon_stock": total_carbon_stock,
    }


def integrated_analysis(consumer_results, growth_results, ecosystem_results):
    """Integrate all analyses for sustainability assessment"""

    print("\n=== INTEGRATED SUSTAINABILITY ANALYSIS ===")

    # Economic indicators
    total_regional_gdp = sum(r["total_gdp"] for r in growth_results.values())
    total_consumer_expenditure = sum(
        r["total_expenditure"] for r in consumer_results.values()
    )
    ecosystem_service_value = ecosystem_results["annual_service_value"]

    # Sustainability ratios
    ecosystem_gdp_ratio = ecosystem_service_value / total_regional_gdp
    natural_capital_ratio = ecosystem_results["carbon_stock_value"] / total_regional_gdp

    print(f"Regional GDP: ${total_regional_gdp:,.0f}")
    print(f"Consumer expenditure: ${total_consumer_expenditure:,.0f}")
    print(f"Ecosystem services value: ${ecosystem_service_value:,.0f}")
    print(f"Ecosystem services / GDP ratio: {ecosystem_gdp_ratio:.3f}")
    print(f"Natural capital / GDP ratio: {natural_capital_ratio:.3f}")

    # Rural-urban analysis
    rural_urban_gap = (
        growth_results["urban_center"]["final_gdp_per_capita"]
        / growth_results["rural_area"]["final_gdp_per_capita"]
    )

    print(f"Rural-urban income gap: {rural_urban_gap:.1f}:1")

    print("\nPolicy Recommendations:")
    if ecosystem_gdp_ratio < 0.1:
        print("- Invest in natural capital and ecosystem restoration")
    if rural_urban_gap > 2.0:
        print("- Implement payments for ecosystem services to support rural areas")
    if natural_capital_ratio < 1.0:
        print("- Develop sustainable economic strategies to protect natural capital")

    return {
        "ecosystem_gdp_ratio": ecosystem_gdp_ratio,
        "natural_capital_ratio": natural_capital_ratio,
        "rural_urban_gap": rural_urban_gap,
    }


def main():
    """Run comprehensive economic analysis"""

    print("GEO-INFER-ECON Comprehensive Analysis Example")
    print("=" * 50)

    # Create bioregion
    bioregion = create_sample_bioregion()
    print(
        f"Analyzing bioregion with area: {bioregion.geometry.area.iloc[0]:.2f} square degrees"
    )

    # Run analyses
    consumer_results = analyze_consumer_behavior()
    growth_results = analyze_regional_growth()
    ecosystem_results = analyze_ecosystem_services()
    integrated_results = integrated_analysis(
        consumer_results, growth_results, ecosystem_results
    )

    print("\n" + "=" * 50)
    print("Analysis demonstrates integration of:")
    print("- Microeconomic consumer behavior analysis")
    print("- Macroeconomic regional growth modeling")
    print("- Bioregional ecosystem services valuation")
    print("- Integrated sustainability assessment")

    return {
        "consumer": consumer_results,
        "growth": growth_results,
        "ecosystem": ecosystem_results,
        "integrated": integrated_results,
    }


if __name__ == "__main__":
    results = main()
