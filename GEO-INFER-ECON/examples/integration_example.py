"""
Integration Example: Using GEO-INFER-ECON with SPACE, TIME, and DATA modules

This example demonstrates how to use the integration adapters to combine
economic analysis with spatial, temporal, and data management capabilities.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Import integration adapters
from geo_infer_econ.integrations import (
    SpaceIntegration,
    TimeIntegration,
    DataIntegration,
)

# Import economic analysis tools
from geo_infer_econ.utils.indicators import EconomicIndicators


def example_spatial_economic_analysis():
    """Example: Spatial economic analysis using GEO-INFER-SPACE integration."""

    print("=== Spatial Economic Analysis Example ===")

    # Initialize space integration
    space = SpaceIntegration(backend="h3")

    if not space.is_available():
        print("⚠️  GEO-INFER-SPACE not available. Using fallback methods.")

    # Create sample economic data with spatial coordinates
    regions = gpd.GeoDataFrame(
        {
            "region_id": ["A", "B", "C", "D", "E"],
            "gdp_per_capita": [50000, 45000, 55000, 40000, 60000],
            "unemployment": [5.2, 6.1, 4.8, 7.2, 4.5],
            "education_index": [0.85, 0.78, 0.92, 0.72, 0.95],
        },
        geometry=[
            Point(-122.4, 37.8),  # San Francisco
            Point(-122.3, 37.7),  # Oakland
            Point(-122.5, 37.9),  # Berkeley
            Point(-122.2, 37.6),  # San Leandro
            Point(-122.6, 38.0),  # Richmond
        ],
    )

    # Convert to spatial cells
    print("\nConverting regions to spatial cells:")
    for _, row in regions.iterrows():
        centroid = row.geometry.centroid
        cell = space.latlng_to_cell(centroid.y, centroid.x, resolution=9)
        print(f"  Region {row['region_id']}: {cell}")

    # Calculate distances between regions
    print("\nCalculating distances between regions:")
    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            p1 = (regions.iloc[i].geometry.y, regions.iloc[i].geometry.x)
            p2 = (regions.iloc[j].geometry.y, regions.iloc[j].geometry.x)
            distance = space.calculate_distance(p1, p2)
            if distance:
                print(
                    f"  {regions.iloc[i]['region_id']} to {regions.iloc[j]['region_id']}: {distance / 1000:.2f} km"
                )

    # Analyze hotspots
    print("\nAnalyzing economic hotspots:")
    hotspots = space.analyze_hotspots(regions, "gdp_per_capita")
    if hotspots is not None:
        print(f"  Found {len(hotspots)} hotspot regions")
    else:
        print("  Hotspot analysis not available")


def example_temporal_economic_analysis():
    """Example: Temporal economic analysis using GEO-INFER-TIME integration."""

    print("\n=== Temporal Economic Analysis Example ===")

    # Initialize time integration
    time = TimeIntegration()

    if not time.is_available():
        print("⚠️  GEO-INFER-TIME not available. Using fallback methods.")

    # Create sample economic time series
    dates = pd.date_range(start="2020-01-01", periods=48, freq="M")
    gdp_growth = pd.Series(
        np.random.randn(48).cumsum() * 0.5 + 2.0,  # Simulated GDP growth
        index=dates,
        name="gdp_growth",
    )

    # Detect trends
    print("\nDetecting trends in GDP growth:")
    trend = time.detect_trend(gdp_growth, method="linear")
    if trend:
        print(f"  Trend: {trend.get('trend', 'unknown')}")
        print(f"  Slope: {trend.get('slope', 0):.4f}")
        print(f"  R-squared: {trend.get('r_squared', 0):.4f}")

    # Analyze seasonality
    print("\nAnalyzing seasonality:")
    seasonality = time.analyze_seasonality(gdp_growth, period=12)
    if seasonality:
        print(
            f"  Seasonal pattern detected: {seasonality.get('has_seasonality', False)}"
        )

    # Decompose time series
    print("\nDecomposing time series:")
    decomposition = time.decompose_time_series(gdp_growth, model="additive")
    if decomposition:
        print("  Decomposition components:")
        print(
            f"    Trend range: {decomposition['trend'].min():.2f} to {decomposition['trend'].max():.2f}"
        )
        print(
            f"    Seasonal range: {decomposition['seasonal'].min():.2f} to {decomposition['seasonal'].max():.2f}"
        )

    # Forecast
    print("\nForecasting future GDP growth:")
    forecast = time.forecast(gdp_growth, horizon=12, method="arima")
    if forecast:
        print(f"  Forecast available: {forecast.get('forecast', 'N/A')}")
    else:
        print("  Forecasting not available")


def example_data_integration():
    """Example: Data loading using GEO-INFER-DATA integration."""

    print("\n=== Data Integration Example ===")

    # Initialize data integration
    data = DataIntegration()

    if not data.is_available():
        print("⚠️  GEO-INFER-DATA not available. Using fallback methods.")

    # List available datasets
    print("\nListing available economic datasets:")
    datasets = data.list_datasets(dataset_type="vector", tags=["economic"])
    if datasets:
        print(f"  Found {len(datasets)} economic datasets")
        for ds in datasets[:5]:  # Show first 5
            print(f"    - {ds.get('id', 'unknown')}: {ds.get('name', 'unnamed')}")
    else:
        print("  No datasets found or data service not available")

    # Load data from file (fallback method)
    print("\nLoading economic data from file:")
    # This would work with actual file paths
    print("  (File loading would work with actual data files)")


def example_integrated_analysis():
    """Example: Integrated spatial-temporal-economic analysis."""

    print("\n=== Integrated Analysis Example ===")

    # Initialize all integrations
    space = SpaceIntegration()
    time = TimeIntegration()

    # Create sample regional economic panel data
    regions = ["A", "B", "C"]
    dates = pd.date_range(start="2020-01-01", periods=24, freq="M")

    panel_data = []
    for region in regions:
        for date in dates:
            panel_data.append(
                {
                    "region": region,
                    "date": date,
                    "gdp": np.random.randn() * 1000 + 10000,
                    "unemployment": np.random.randn() * 2 + 5,
                    "lat": 37.7 + np.random.randn() * 0.1,
                    "lon": -122.4 + np.random.randn() * 0.1,
                }
            )

    df = pd.DataFrame(panel_data)

    # Spatial analysis
    print("\nPerforming spatial analysis:")
    gdf = gpd.GeoDataFrame(
        df, geometry=[Point(row["lon"], row["lat"]) for _, row in df.iterrows()]
    )
    cells = {space.latlng_to_cell(pt.y, pt.x, resolution=7) for pt in gdf.geometry}
    print(f"  {len(gdf)} observations across {len(cells)} H3 cells")

    # Temporal analysis per region
    print("\nPerforming temporal analysis per region:")
    for region in regions:
        region_data = df[df["region"] == region].set_index("date")["gdp"]
        trend = time.detect_trend(region_data)
        if trend:
            print(f"  Region {region}: {trend.get('trend', 'unknown')} trend")

    # Economic indicators
    print("\nCalculating economic indicators:")
    indicators = EconomicIndicators()
    gdp_values = df.groupby("region")["gdp"].mean()
    gini = indicators.calculate_gini_coefficient(gdp_values.values)
    print(f"  Gini coefficient: {gini:.3f}")


def main():
    """Run all integration examples."""

    print("GEO-INFER-ECON Integration Examples")
    print("=" * 50)

    try:
        example_spatial_economic_analysis()
        example_temporal_economic_analysis()
        example_data_integration()
        example_integrated_analysis()

        print("\n" + "=" * 50)
        print("✅ Integration examples completed successfully!")

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
