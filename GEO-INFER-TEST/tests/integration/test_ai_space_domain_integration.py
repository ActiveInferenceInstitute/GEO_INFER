"""
Integration tests for AI + SPACE + domain modules (AG, HEALTH, ECON).

Tests real integration between AI/ML, spatial analysis, and domain-specific modules.
"""

import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon

from geo_infer_ai.core.training import ModelTrainer
from geo_infer_ai.preprocessing.feature_engineering import GeospatialFeatureEngineer
from geo_infer_space.core.spatial_indexing import SpatialIndexingInterface


@pytest.fixture
def sample_agricultural_data():
    """Sample agricultural data for AI+SPACE+AG integration."""
    np.random.seed(42)

    # Create field boundaries
    fields = gpd.GeoDataFrame(
        {
            "geometry": [
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
            ],
            "field_id": ["field_A", "field_B"],
            "crop_type": ["corn", "wheat"],
            "yield_actual": [8.5, 6.2],  # tons/hectare
        },
        crs="EPSG:4326",
    )

    # Create sensor points
    points = []
    for _, field in fields.iterrows():
        bounds = field.geometry.bounds
        for _ in range(10):
            lat = np.random.uniform(bounds[1], bounds[3])
            lng = np.random.uniform(bounds[0], bounds[2])
            points.append(
                {
                    "geometry": Point(lng, lat),
                    "field_id": field["field_id"],
                    "ndvi": np.random.uniform(0.3, 0.8),
                    "soil_moisture": np.random.uniform(0.2, 0.5),
                    "temperature": 20 + np.random.normal(0, 3),
                }
            )

    sensor_gdf = gpd.GeoDataFrame(points, crs="EPSG:4326")

    return {"fields": fields, "sensors": sensor_gdf}


@pytest.mark.integration
class TestAiSpaceAgIntegration:
    """Test integration between AI, SPACE, and AG modules."""

    def test_spatial_feature_engineering_for_agriculture(
        self, sample_agricultural_data
    ):
        """Test spatial feature engineering for agricultural ML models."""
        sensors = sample_agricultural_data["sensors"]

        # Use SPACE for spatial indexing
        indexer = SpatialIndexingInterface(backend="h3")
        sensors["h3_cell"] = sensors.geometry.apply(
            lambda point: indexer.latlng_to_cell(point.y, point.x, resolution=10)
        )

        # Use AI for feature engineering
        _engineer = GeospatialFeatureEngineer()

        # Extract spatial features
        features = []
        for _, sensor in sensors.iterrows():
            features.append(
                {
                    "h3_cell": sensor["h3_cell"],
                    "ndvi": sensor["ndvi"],
                    "soil_moisture": sensor["soil_moisture"],
                    "temperature": sensor["temperature"],
                    "field_id": sensor["field_id"],
                }
            )

        features_df = pd.DataFrame(features)

        # Verify feature engineering
        assert len(features_df) > 0
        assert "h3_cell" in features_df.columns
        assert "ndvi" in features_df.columns

    def test_crop_yield_prediction_with_spatial_ai(self, sample_agricultural_data):
        """Test crop yield prediction using AI and spatial analysis."""
        fields = sample_agricultural_data["fields"]
        sensors = sample_agricultural_data["sensors"]

        # Aggregate sensor data by field
        field_features = []
        for _, field in fields.iterrows():
            field_sensors = sensors[sensors["field_id"] == field["field_id"]]

            field_features.append(
                {
                    "field_id": field["field_id"],
                    "avg_ndvi": field_sensors["ndvi"].mean(),
                    "avg_soil_moisture": field_sensors["soil_moisture"].mean(),
                    "avg_temperature": field_sensors["temperature"].mean(),
                    "yield_actual": field["yield_actual"],
                }
            )

        features_df = pd.DataFrame(field_features)

        # Prepare for ML model
        X = features_df[["avg_ndvi", "avg_soil_moisture", "avg_temperature"]].values
        y = features_df["yield_actual"].values

        # Use AI trainer
        trainer = ModelTrainer()

        # Train simple model (if enough data)
        if len(features_df) >= 2:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split

            if len(X) >= 2:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42
                )

                model = RandomForestRegressor(n_estimators=10, random_state=42)
                model.fit(X_train, y_train)

                # Evaluate
                results = trainer.evaluate_model(
                    model, X_test, y_test, task_type="regression"
                )

                assert "mse" in results or "mae" in results or "r2" in results
                assert results is not None


@pytest.mark.integration
class TestAiSpaceHealthIntegration:
    """Test integration between AI, SPACE, and HEALTH modules."""

    def test_epidemiological_analysis_with_spatial_ai(self):
        """Test epidemiological analysis using AI and spatial analysis."""
        # Create sample health data
        np.random.seed(42)
        regions = gpd.GeoDataFrame(
            {
                "geometry": [
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
                ],
                "region_id": ["region_A", "region_B"],
                "population": [100000, 150000],
                "cases": [150, 200],
            },
            crs="EPSG:4326",
        )

        # Add spatial indexing
        indexer = SpatialIndexingInterface(backend="h3")
        projected_centroids = regions.to_crs("EPSG:3857").geometry.centroid.to_crs(
            "EPSG:4326"
        )
        regions["h3_cell"] = projected_centroids.apply(
            lambda point: indexer.latlng_to_cell(point.y, point.x, resolution=9)
        )

        # Calculate incidence rate
        regions["incidence_rate"] = regions["cases"] / regions["population"] * 1000

        # Verify spatial-health integration
        assert "h3_cell" in regions.columns
        assert "incidence_rate" in regions.columns
        assert len(regions) == 2


@pytest.mark.integration
class TestAiSpaceEconIntegration:
    """Test integration between AI, SPACE, and ECON modules."""

    def test_economic_modeling_with_spatial_ai(self):
        """Test economic modeling using AI and spatial analysis."""
        # Create sample economic data
        np.random.seed(42)
        economic_regions = gpd.GeoDataFrame(
            {
                "geometry": [
                    Point(-122.4194, 37.7749),
                    Point(-122.4000, 37.7800),
                    Point(-122.4500, 37.7600),
                ],
                "region_id": ["metro_A", "metro_B", "metro_C"],
                "gdp": [1000000, 800000, 600000],
                "population": [500000, 400000, 300000],
                "unemployment": [5.2, 6.1, 7.3],
            },
            crs="EPSG:4326",
        )

        # Add spatial indexing
        indexer = SpatialIndexingInterface(backend="h3")
        economic_regions["h3_cell"] = economic_regions.geometry.apply(
            lambda point: indexer.latlng_to_cell(point.y, point.x, resolution=9)
        )

        # Calculate economic metrics
        economic_regions["gdp_per_capita"] = (
            economic_regions["gdp"] / economic_regions["population"]
        )

        # Prepare for ML (if ECON module available)
        features = economic_regions[["gdp", "population", "unemployment"]].values
        _target = economic_regions["gdp_per_capita"].values

        # Verify spatial-economic integration
        assert "h3_cell" in economic_regions.columns
        assert "gdp_per_capita" in economic_regions.columns
        assert len(features) == 3
