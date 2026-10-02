"""
Integration tests for cross-module workflows in the GEO-INFER framework.

This module tests end-to-end workflows that span multiple GEO-INFER modules,
ensuring proper data flow, API compatibility, and system integration.
"""

import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
import h3
from shapely.geometry import Point, Polygon


# Test data fixtures


def _linear_trend(series: pd.Series) -> float:
    """Least-squares slope of a series against its sample index."""
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) < 2:
        return 0.0
    return float(np.polyfit(np.arange(len(values), dtype=float), values, 1)[0])


@pytest.fixture
def sample_spatial_temporal_data():
    """Sample spatial-temporal data for integration testing."""
    # Create time series with spatial components
    dates = pd.date_range("2023-01-01", periods=30, freq="D")
    np.random.seed(42)

    # Generate spatial points
    points = []
    for _ in range(10):
        lat = 37.7749 + np.random.normal(0, 0.01)
        lng = -122.4194 + np.random.normal(0, 0.01)
        points.append(Point(lng, lat))

    # Create GeoDataFrame with temporal data
    data = []
    for date in dates:
        for i, point in enumerate(points):
            data.append(
                {
                    "timestamp": date,
                    "geometry": point,
                    "sensor_id": f"sensor_{i:03d}",
                    "temperature": 20
                    + 5 * np.sin(2 * np.pi * date.dayofyear / 365)
                    + np.random.normal(0, 2),
                    "humidity": 60
                    + 10 * np.sin(2 * np.pi * date.dayofyear / 365 + np.pi / 4)
                    + np.random.normal(0, 5),
                    "air_quality": np.random.exponential(20),
                }
            )

    return gpd.GeoDataFrame(data, crs="EPSG:4326")


@pytest.fixture
def sample_agricultural_workflow_data():
    """Sample data for agricultural workflow testing."""
    # Create field boundaries
    fields = [
        Polygon(
            [
                (-122.5, 37.7),
                (-122.3, 37.7),
                (-122.3, 37.9),
                (-122.5, 37.9),
                (-122.5, 37.7),
            ]
        ),
        Polygon(
            [
                (-122.4, 37.6),
                (-122.2, 37.6),
                (-122.2, 37.8),
                (-122.4, 37.8),
                (-122.4, 37.6),
            ]
        ),
    ]

    field_gdf = gpd.GeoDataFrame(
        {
            "geometry": fields,
            "field_id": ["field_A", "field_B"],
            "crop_type": ["corn", "wheat"],
            "area_ha": [50, 75],
        },
        crs="EPSG:4326",
    )

    # Create time series data for each field
    dates = pd.date_range("2023-01-01", periods=120, freq="D")
    sensor_data = []

    for field_id in ["field_A", "field_B"]:
        for date in dates:
            day_of_year = date.dayofyear
            _seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * day_of_year / 365)

            sensor_data.append(
                {
                    "timestamp": date,
                    "field_id": field_id,
                    "temperature": 15
                    + 10 * np.sin(2 * np.pi * day_of_year / 365)
                    + np.random.normal(0, 3),
                    "rainfall": max(0, np.random.exponential(2)),
                    "soil_moisture": 0.3
                    + 0.2 * np.sin(2 * np.pi * day_of_year / 365)
                    + np.random.normal(0, 0.05),
                    "ndvi": 0.2
                    + 0.6 * np.clip(day_of_year / 120, 0, 1)
                    + np.random.normal(0, 0.05),
                    "yield_estimate": 0.8
                    + 0.4 * np.clip(day_of_year / 120, 0, 1)
                    + np.random.normal(0, 0.1),
                }
            )

    return {"fields": field_gdf, "sensor_data": pd.DataFrame(sensor_data)}


@pytest.fixture
def sample_health_epidemiology_data():
    """Sample data for health epidemiology workflow testing."""
    # Create regions
    regions = [
        Polygon(
            [
                (-122.5, 37.7),
                (-122.3, 37.7),
                (-122.3, 37.9),
                (-122.5, 37.9),
                (-122.5, 37.7),
            ]
        ),
        Polygon(
            [
                (-122.4, 37.6),
                (-122.2, 37.6),
                (-122.2, 37.8),
                (-122.4, 37.8),
                (-122.4, 37.6),
            ]
        ),
        Polygon(
            [
                (-122.6, 37.5),
                (-122.4, 37.5),
                (-122.4, 37.7),
                (-122.6, 37.7),
                (-122.6, 37.5),
            ]
        ),
    ]

    region_gdf = gpd.GeoDataFrame(
        {
            "geometry": regions,
            "region_id": ["region_A", "region_B", "region_C"],
            "population": [100000, 150000, 80000],
            "healthcare_facilities": [5, 8, 3],
        },
        crs="EPSG:4326",
    )

    # Create epidemiological data
    dates = pd.date_range("2023-01-01", periods=365, freq="D")
    health_data = []

    for region_id in ["region_A", "region_B", "region_C"]:
        # Base rates with seasonal patterns
        base_cases = 10 + 5 * np.sin(2 * np.pi * np.arange(365) / 365)

        for i, date in enumerate(dates):
            day_of_year = date.dayofyear
            seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * day_of_year / 365)

            health_data.append(
                {
                    "date": date,
                    "region_id": region_id,
                    "cases": max(
                        0, int(base_cases[i] * seasonal_factor + np.random.poisson(2))
                    ),
                    "hospitalizations": max(
                        0, int(np.random.poisson(0.1 * base_cases[i] * seasonal_factor))
                    ),
                    "deaths": max(
                        0,
                        int(np.random.poisson(0.01 * base_cases[i] * seasonal_factor)),
                    ),
                    "vaccinations": np.random.poisson(50),
                    "testing_rate": np.random.uniform(0.1, 0.3),
                }
            )

    return {"regions": region_gdf, "health_data": pd.DataFrame(health_data)}


class TestSpatialTemporalIntegration:
    """Test integration between SPACE and TIME modules."""

    def test_spatial_temporal_data_fusion(self, sample_spatial_temporal_data):
        """Test fusion of spatial and temporal data."""
        gdf = sample_spatial_temporal_data

        # Test spatial indexing with temporal data
        _spatial_index = gdf.sindex

        # Test temporal aggregation with spatial grouping
        temporal_stats = gdf.groupby("sensor_id").agg(
            {
                "temperature": ["mean", "std", "min", "max"],
                "humidity": ["mean", "std"],
                "air_quality": ["mean", "std"],
            }
        )

        assert len(temporal_stats) == 10  # 10 sensors
        assert "temperature" in temporal_stats.columns
        assert "humidity" in temporal_stats.columns
        assert "air_quality" in temporal_stats.columns

    def test_h3_temporal_analysis(self, sample_spatial_temporal_data):
        """Test H3 spatial indexing with temporal analysis."""
        gdf = sample_spatial_temporal_data

        # Add H3 indices
        resolution = 10
        gdf["h3_index"] = gdf.apply(
            lambda row: h3.latlng_to_cell(row.geometry.y, row.geometry.x, resolution),
            axis=1,
        )

        # Group by H3 index and time
        gdf["date"] = gdf["timestamp"].dt.date
        h3_temporal_stats = (
            gdf.groupby(["h3_index", "date"])
            .agg({"temperature": "mean", "humidity": "mean", "air_quality": "mean"})
            .reset_index()
        )

        assert len(h3_temporal_stats) > 0
        assert "h3_index" in h3_temporal_stats.columns
        assert "date" in h3_temporal_stats.columns

    def test_spatial_temporal_interpolation(self, sample_spatial_temporal_data):
        """Test spatial-temporal interpolation."""
        gdf = sample_spatial_temporal_data

        # Create regular grid for interpolation
        x_coords = np.linspace(-122.5, -122.3, 20)
        y_coords = np.linspace(37.7, 37.9, 20)
        grid_x, grid_y = np.meshgrid(x_coords, y_coords)

        # Extract coordinates and values for interpolation
        points = np.array([[point.x, point.y] for point in gdf.geometry])
        values = gdf["temperature"].values

        # Simple nearest neighbor interpolation
        from scipy.spatial import cKDTree

        tree = cKDTree(points)
        distances, indices = tree.query(
            np.column_stack([grid_x.ravel(), grid_y.ravel()])
        )
        interpolated = values[indices].reshape(grid_x.shape)

        assert interpolated.shape == grid_x.shape
        assert not np.isnan(interpolated).all()


class TestAgriculturalWorkflow:
    """Test agricultural workflow involving SPACE, TIME, and AI modules."""

    def test_precision_agriculture_workflow(self, sample_agricultural_workflow_data):
        """Test end-to-end precision agriculture workflow."""
        fields = sample_agricultural_workflow_data["fields"]
        sensor_data = sample_agricultural_workflow_data["sensor_data"]

        # 1. Spatial analysis of fields
        metric_fields = fields.to_crs("EPSG:3857")
        _field_areas = metric_fields.geometry.area
        _field_centroids = metric_fields.geometry.centroid

        # 2. Temporal analysis of sensor data
        temporal_trends = sensor_data.groupby("field_id").agg(
            {
                "temperature": ["mean", _linear_trend],
                "rainfall": "sum",
                "soil_moisture": "mean",
                "ndvi": "mean",
                "yield_estimate": "mean",
            }
        )

        # 3. Spatial-temporal correlation
        field_stats = []
        for field_id in fields["field_id"]:
            field_data = sensor_data[sensor_data["field_id"] == field_id]
            field_stats.append(
                {
                    "field_id": field_id,
                    "avg_temperature": field_data["temperature"].mean(),
                    "total_rainfall": field_data["rainfall"].sum(),
                    "avg_soil_moisture": field_data["soil_moisture"].mean(),
                    "avg_ndvi": field_data["ndvi"].mean(),
                    "yield_prediction": field_data["yield_estimate"].iloc[-1],
                }
            )

        field_analysis = pd.DataFrame(field_stats)

        assert len(field_analysis) == 2  # 2 fields
        assert "avg_temperature" in field_analysis.columns
        assert "yield_prediction" in field_analysis.columns
        assert len(temporal_trends) == 2
        assert np.isfinite(
            temporal_trends[("temperature", "_linear_trend")].to_numpy()
        ).all()

    def test_crop_health_monitoring(self, sample_agricultural_workflow_data):
        """Test crop health monitoring with spatial-temporal analysis."""
        fields = sample_agricultural_workflow_data["fields"]
        sensor_data = sample_agricultural_workflow_data["sensor_data"]

        # Add H3 spatial indexing
        resolution = 9
        projected_centroids = fields.to_crs("EPSG:3857").geometry.centroid.to_crs(
            "EPSG:4326"
        )
        fields["h3_index"] = projected_centroids.apply(
            lambda point: h3.latlng_to_cell(point.y, point.x, resolution)
        )

        # Analyze crop health indicators
        health_indicators = []
        for field_id in fields["field_id"]:
            field_data = sensor_data[sensor_data["field_id"] == field_id]

            # Calculate health indicators
            recent_data = field_data.tail(30)  # Last 30 days

            health_score = (
                recent_data["ndvi"].mean() * 0.4
                + recent_data["soil_moisture"].mean() * 0.3
                + (1 - recent_data["temperature"].std() / 20) * 0.3
            )

            health_indicators.append(
                {
                    "field_id": field_id,
                    "health_score": health_score,
                    "ndvi_trend": recent_data["ndvi"].iloc[-1]
                    - recent_data["ndvi"].iloc[0],
                    "moisture_status": (
                        "optimal"
                        if recent_data["soil_moisture"].mean() > 0.3
                        else "dry"
                    ),
                    "stress_level": (
                        "low"
                        if health_score > 0.7
                        else "medium"
                        if health_score > 0.5
                        else "high"
                    ),
                }
            )

        health_analysis = pd.DataFrame(health_indicators)

        assert len(health_analysis) == 2
        assert "health_score" in health_analysis.columns
        assert "stress_level" in health_analysis.columns


class TestHealthEpidemiologyWorkflow:
    """Test health epidemiology workflow involving SPACE, TIME, and HEALTH modules."""

    def test_disease_spread_modeling(self, sample_health_epidemiology_data):
        """Test disease spread modeling with spatial-temporal analysis."""
        regions = sample_health_epidemiology_data["regions"]
        health_data = sample_health_epidemiology_data["health_data"]

        # Spatial analysis of regions
        metric_regions = regions.to_crs("EPSG:3857")
        _region_areas = metric_regions.geometry.area
        _region_centroids = metric_regions.geometry.centroid

        # Add H3 spatial indexing
        resolution = 8
        projected_centroids = regions.to_crs("EPSG:3857").geometry.centroid.to_crs(
            "EPSG:4326"
        )
        regions["h3_index"] = projected_centroids.apply(
            lambda point: h3.latlng_to_cell(point.y, point.x, resolution)
        )

        # Temporal analysis of disease spread
        disease_spread = (
            health_data.groupby(["region_id", "date"])
            .agg(
                {
                    "cases": "sum",
                    "hospitalizations": "sum",
                    "deaths": "sum",
                    "vaccinations": "sum",
                }
            )
            .reset_index()
        )

        # Calculate disease metrics
        disease_metrics = []
        for region_id in regions["region_id"]:
            region_data = disease_spread[disease_spread["region_id"] == region_id]

            # Calculate incidence rate
            population = regions[regions["region_id"] == region_id]["population"].iloc[
                0
            ]
            total_cases = region_data["cases"].sum()
            incidence_rate = total_cases / population * 1000  # per 1000 people

            # Calculate case fatality rate
            total_deaths = region_data["deaths"].sum()
            case_fatality_rate = total_deaths / total_cases if total_cases > 0 else 0

            disease_metrics.append(
                {
                    "region_id": region_id,
                    "incidence_rate": incidence_rate,
                    "case_fatality_rate": case_fatality_rate,
                    "total_cases": total_cases,
                    "total_deaths": total_deaths,
                    "vaccination_coverage": region_data["vaccinations"].sum()
                    / population,
                }
            )

        metrics_df = pd.DataFrame(disease_metrics)

        assert len(metrics_df) == 3  # 3 regions
        assert "incidence_rate" in metrics_df.columns
        assert "case_fatality_rate" in metrics_df.columns

    def test_healthcare_accessibility_analysis(self, sample_health_epidemiology_data):
        """Test healthcare accessibility analysis."""
        regions = sample_health_epidemiology_data["regions"]
        _health_data = sample_health_epidemiology_data["health_data"]

        # Calculate healthcare accessibility metrics
        accessibility_analysis = []
        for region_id in regions["region_id"]:
            region_info = regions[regions["region_id"] == region_id]
            population = region_info["population"].iloc[0]
            facilities = region_info["healthcare_facilities"].iloc[0]

            # Calculate facility density
            area_km2 = region_info.geometry.iloc[0].area * 111 * 111  # Approximate km2
            facility_density = facilities / area_km2

            # Calculate population per facility
            population_per_facility = (
                population / facilities if facilities > 0 else float("inf")
            )

            accessibility_analysis.append(
                {
                    "region_id": region_id,
                    "facility_density": facility_density,
                    "population_per_facility": population_per_facility,
                    "accessibility_score": 1
                    / (1 + population_per_facility / 10000),  # Normalized score
                    "area_km2": area_km2,
                }
            )

        accessibility_df = pd.DataFrame(accessibility_analysis)

        assert len(accessibility_df) == 3
        assert "facility_density" in accessibility_df.columns
        assert "accessibility_score" in accessibility_df.columns


class TestEconomicAnalysisWorkflow:
    """Test economic analysis workflow involving SPACE, TIME, and ECON modules."""

    def test_spatial_economic_modeling(self):
        """Test spatial economic modeling with temporal components."""
        # Create sample economic data with spatial components
        regions = [
            {
                "id": "region_A",
                "lat": 37.7749,
                "lng": -122.4194,
                "gdp": 1000000,
                "population": 500000,
            },
            {
                "id": "region_B",
                "lat": 37.7800,
                "lng": -122.4000,
                "gdp": 800000,
                "population": 400000,
            },
            {
                "id": "region_C",
                "lat": 37.7600,
                "lng": -122.4500,
                "gdp": 600000,
                "population": 300000,
            },
        ]

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(
            regions,
            geometry=[Point(r["lng"], r["lat"]) for r in regions],
            crs="EPSG:4326",
        )

        # Calculate economic metrics
        gdf["gdp_per_capita"] = gdf["gdp"] / gdf["population"]
        gdf["economic_density"] = gdf["gdp"] / 100  # Simplified area calculation

        # Spatial economic analysis
        # Calculate economic gravity model (simplified)
        economic_interactions = []
        for i, region1 in gdf.iterrows():
            for j, region2 in gdf.iterrows():
                if i != j:
                    distance = region1.geometry.distance(region2.geometry)
                    interaction = (region1["gdp"] * region2["gdp"]) / (distance**2)
                    economic_interactions.append(
                        {
                            "from_region": region1["id"],
                            "to_region": region2["id"],
                            "distance": distance,
                            "economic_interaction": interaction,
                        }
                    )

        interactions_df = pd.DataFrame(economic_interactions)

        assert len(interactions_df) == 6  # 3 regions × 2 other regions
        assert "economic_interaction" in interactions_df.columns
        assert all(interactions_df["economic_interaction"] > 0)


class TestLogisticsOptimizationWorkflow:
    """Test logistics optimization workflow involving SPACE, TIME, and LOG modules."""

    def test_route_optimization_workflow(self):
        """Test route optimization with spatial-temporal constraints."""
        # Create sample logistics network
        locations = [
            {"id": "warehouse", "lat": 37.7749, "lng": -122.4194, "type": "warehouse"},
            {"id": "customer_A", "lat": 37.7800, "lng": -122.4000, "type": "customer"},
            {"id": "customer_B", "lat": 37.7600, "lng": -122.4500, "type": "customer"},
            {"id": "customer_C", "lat": 37.8000, "lng": -122.3500, "type": "customer"},
            {"id": "supplier_A", "lat": 37.7500, "lng": -122.5000, "type": "supplier"},
        ]

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(
            locations,
            geometry=[Point(loc["lng"], loc["lat"]) for loc in locations],
            crs="EPSG:4326",
        )

        # Calculate distances between all locations
        distance_matrix = {}
        for i, loc1 in gdf.iterrows():
            for j, loc2 in gdf.iterrows():
                if i != j:
                    distance = (
                        loc1.geometry.distance(loc2.geometry) * 111
                    )  # Convert to km
                    distance_matrix[(loc1["id"], loc2["id"])] = distance

        # Simple route optimization (nearest neighbor)
        def optimize_route(start_location, locations):
            route = [start_location]
            unvisited = [loc for loc in locations if loc != start_location]

            current = start_location
            while unvisited:
                # Find nearest unvisited location
                nearest = min(
                    unvisited,
                    key=lambda loc: distance_matrix.get((current, loc), float("inf")),
                )
                route.append(nearest)
                unvisited.remove(nearest)
                current = nearest

            return route

        # Optimize route from warehouse
        optimal_route = optimize_route("warehouse", [loc["id"] for loc in locations])

        # Calculate route metrics
        total_distance = 0
        for i in range(len(optimal_route) - 1):
            total_distance += distance_matrix.get(
                (optimal_route[i], optimal_route[i + 1]), 0
            )

        route_analysis = {
            "optimal_route": optimal_route,
            "total_distance_km": total_distance,
            "number_of_stops": len(optimal_route) - 1,
            "average_distance_per_stop": (
                total_distance / (len(optimal_route) - 1)
                if len(optimal_route) > 1
                else 0
            ),
        }

        assert len(optimal_route) == 5  # All locations
        assert optimal_route[0] == "warehouse"
        assert route_analysis["total_distance_km"] > 0


class TestCrossModuleDataFlow:
    """Test data flow between different modules."""

    def test_spatial_data_to_ai_workflow(self):
        """Test data flow from SPACE module to AI module."""
        # Create spatial data
        points = gpd.GeoDataFrame(
            {
                "geometry": [
                    Point(-122.4194, 37.7749),
                    Point(-122.4000, 37.7800),
                    Point(-122.4500, 37.7600),
                ],
                "feature_1": [1.0, 2.0, 3.0],
                "feature_2": [4.0, 5.0, 6.0],
                "target": [0, 1, 0],
            },
            crs="EPSG:4326",
        )

        # Add H3 spatial features
        resolution = 10
        points["h3_index"] = points.geometry.apply(
            lambda point: h3.latlng_to_cell(point.y, point.x, resolution)
        )

        # Extract spatial features for ML
        spatial_features = []
        for _, point in points.iterrows():
            # Get neighboring H3 cells
            h3_cell = point["h3_index"]
            neighbors = h3.grid_disk(h3_cell, 1)

            spatial_features.append(
                {
                    "h3_index": h3_cell,
                    "neighbor_count": len(neighbors),
                    "feature_1": point["feature_1"],
                    "feature_2": point["feature_2"],
                    "target": point["target"],
                }
            )

        features_df = pd.DataFrame(spatial_features)

        # Simulate ML model training
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        X = features_df[["neighbor_count", "feature_1", "feature_2"]]
        y = features_df["target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate model
        accuracy = model.score(X_test, y_test)

        assert accuracy >= 0  # Should be a valid accuracy score
        assert len(features_df) == 3
        assert "neighbor_count" in features_df.columns

    def test_temporal_data_to_bayesian_workflow(self):
        """Test data flow from TIME module to BAYES module."""
        # Create temporal data
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        np.random.seed(42)

        temporal_data = pd.DataFrame(
            {
                "date": dates,
                "value": np.cumsum(np.random.normal(0, 1, 100)),
                "noise": np.random.normal(0, 0.1, 100),
            }
        )

        # Add temporal features
        temporal_data["day_of_week"] = temporal_data["date"].dt.dayofweek
        temporal_data["month"] = temporal_data["date"].dt.month
        temporal_data["trend"] = np.arange(len(temporal_data))

        # Simulate Bayesian analysis
        # Calculate posterior probabilities for trend detection

        # Simple Bayesian trend analysis
        trend_slope = np.polyfit(temporal_data["trend"], temporal_data["value"], 1)[0]

        # Calculate confidence intervals
        residuals = temporal_data["value"] - (
            trend_slope * temporal_data["trend"]
            + np.polyfit(temporal_data["trend"], temporal_data["value"], 1)[1]
        )
        std_error = np.std(residuals)

        # 95% confidence interval
        confidence_interval = 1.96 * std_error

        bayesian_analysis = {
            "trend_slope": trend_slope,
            "confidence_interval": confidence_interval,
            "trend_significant": abs(trend_slope) > confidence_interval,
            "data_points": len(temporal_data),
        }

        assert "trend_slope" in bayesian_analysis
        assert "confidence_interval" in bayesian_analysis
        assert bayesian_analysis["data_points"] == 100


@pytest.mark.performance
class TestPerformanceIntegration:
    """Test performance characteristics of cross-module workflows."""

    def test_small_spatial_temporal_reference(self):
        """Keep an analytical reference in integration, with real H3 indexing."""
        from geo_infer_space import H3StateSpace, align_h3_observations
        from geo_infer_time.core.analysis import TemporalAnalyzer

        center = h3.latlng_to_cell(37.8, -122.4, 8)
        cells = [center, *sorted(set(h3.grid_disk(center, 1)) - {center})]
        space = H3StateSpace(cells)
        timestamps = pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC")
        records = [
            {"cell": cell, "timestamp": timestamp, "value": float(3 * step + index)}
            for step, timestamp in enumerate(timestamps)
            for index, cell in enumerate(cells)
        ]
        series = align_h3_observations(
            pd.DataFrame(records), state_space=space, timestamps=timestamps
        )
        np.testing.assert_array_equal(
            series.data.to_numpy(),
            np.arange(len(cells))[None, :] + 3 * np.arange(4)[:, None],
        )
        trend = TemporalAnalyzer().detect_trend(series)
        assert trend["trend_direction"] == "increasing"
        np.testing.assert_allclose(trend["slope_per_sample"], 3.0, atol=1e-12)


if __name__ == "__main__":
    pytest.main([__file__])
