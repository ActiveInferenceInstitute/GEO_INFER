"""
Unit tests for data validation and quality assurance.

This module tests the DataQualityManager and GeospatialValidator classes
for comprehensive data validation functionality.
"""

import pytest
import pandas as pd
import geopandas as gpd
import numpy as np
from datetime import datetime
from shapely.geometry import Point

from geo_infer_data.core.validation import DataQualityManager
from geo_infer_data.utils.validation import GeospatialValidator, ValidationConfig
from geo_infer_data.models.schemas import (
    DatasetMetadata,
    SpatialExtent,
    TemporalExtent,
    DataLineage,
    DataQualityReport,
    QualityCheck,
    QualityStatus,
)


class TestGeospatialValidator:
    """Test cases for GeospatialValidator."""

    @pytest.fixture
    def validator(self):
        """Create test validator."""
        return GeospatialValidator()

    @pytest.fixture
    def valid_geodataframe(self):
        """Create valid GeoDataFrame for testing."""
        return gpd.GeoDataFrame(
            {
                "id": range(100),
                "temperature": np.random.normal(20, 5, 100),
                "humidity": np.random.normal(60, 10, 100),
            },
            geometry=gpd.points_from_xy(
                np.random.normal(-122.4, 0.1, 100), np.random.normal(37.7, 0.1, 100)
            ),
            crs="EPSG:4326",
        )

    @pytest.fixture
    def invalid_geodataframe(self):
        """Create invalid GeoDataFrame for testing."""
        # Create some invalid geometries
        geometries = []
        for i in range(50):
            if i < 10:  # Invalid geometries
                geometries.append(None)
            elif i < 20:  # Invalid coordinates
                geometries.append(Point(200, 100))  # Invalid longitude
            else:
                geometries.append(Point(-122.4, 37.7))

        return gpd.GeoDataFrame(
            {"id": range(50), "temperature": np.random.normal(20, 5, 50)},
            geometry=geometries,
            crs="EPSG:4326",
        )

    @pytest.fixture
    def valid_dataframe(self):
        """Create valid DataFrame for testing."""
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=1000, freq="h"),
                "temperature": np.random.normal(20, 5, 1000),
                "humidity": np.random.normal(60, 10, 1000),
                "latitude": np.random.normal(37.7, 0.1, 1000),
                "longitude": np.random.normal(-122.4, 0.1, 1000),
            }
        )

    @pytest.mark.asyncio
    async def test_validate_complete_data(self, validator, valid_geodataframe):
        """Test validation of complete, valid data."""
        result = await validator.build_quality_report(valid_geodataframe)

        assert result.overall_score >= 0.8
        assert result.status == QualityStatus.PASS
        assert len(result.issues) == 0

    @pytest.mark.asyncio
    async def test_validate_incomplete_data(self, validator):
        """Test validation of incomplete data."""
        incomplete_data = pd.DataFrame(
            {
                "col1": [1, 2, None, 4, None] * 200,  # 40% missing
                "col2": [None, 2, 3, None, 5] * 200,  # 40% missing
            }
        )

        result = await validator.build_quality_report(incomplete_data)

        assert result.overall_score < 0.8
        assert result.status in [QualityStatus.WARNING, QualityStatus.FAIL]
        assert any("missing" in issue["type"] for issue in result.issues)

    @pytest.mark.asyncio
    async def test_validate_invalid_coordinates(self, validator, invalid_geodataframe):
        """Test validation of invalid coordinates."""
        result = await validator.build_quality_report(invalid_geodataframe)

        assert result.overall_score < 0.8
        assert result.status == QualityStatus.FAIL
        assert any("invalid" in issue["type"] for issue in result.issues)

    def test_geometry_validation(
        self, validator, valid_geodataframe, invalid_geodataframe
    ):
        """Test geometry-specific validation."""
        # Test valid geometries
        valid_result = validator.validate_geometries(valid_geodataframe)
        assert valid_result.score >= 0.9
        assert valid_result.status == QualityStatus.PASS

        # Test invalid geometries
        invalid_result = validator.validate_geometries(invalid_geodataframe)
        assert invalid_result.score < 0.8
        assert invalid_result.status == QualityStatus.FAIL

    def test_coordinate_validation(self, validator, valid_dataframe):
        """Test coordinate-specific validation."""
        result = validator.validate_coordinates(valid_dataframe)

        # Should pass validation for valid coordinates
        assert result.score >= 0.8
        assert result.status == QualityStatus.PASS

        # Test with invalid coordinates
        invalid_df = valid_dataframe.copy()
        invalid_df["longitude"] = 200  # Invalid longitude

        invalid_result = validator.validate_coordinates(invalid_df)
        assert invalid_result.score < 0.8
        assert invalid_result.status == QualityStatus.FAIL

    def test_temporal_validation(self, validator, valid_dataframe):
        """Test temporal validation."""
        result = validator.validate_temporal_data(valid_dataframe)

        # Should pass validation for reasonable temporal data
        assert result.score >= 0.8
        assert result.status == QualityStatus.PASS

        # Test with future dates
        future_df = valid_dataframe.copy()
        future_df["timestamp"] = datetime.now() + pd.Timedelta(days=365)

        future_result = validator.validate_temporal_data(future_df)
        assert future_result.score < 0.9  # Should have warnings about future dates

    def test_timezone_aware_temporal_validation(self, validator):
        data = pd.DataFrame(
            {"timestamp": pd.date_range("2024-01-01", periods=2, tz="UTC")}
        )
        result = validator.validate_temporal_data(data)
        assert result.status == QualityStatus.PASS


class TestDataQualityManager:
    """Test cases for DataQualityManager."""

    @pytest.fixture
    def quality_config(self):
        """Create test quality configuration."""
        return ValidationConfig(
            validation_rules=["completeness", "accuracy", "consistency"],
            quality_threshold=0.8,
            strict_mode=False,
            real_time_monitoring=True,
        )

    @pytest.fixture
    def quality_manager(self, quality_config):
        """Create test quality manager."""
        return DataQualityManager(
            validation_rules=",".join(quality_config.validation_rules),
            quality_threshold=quality_config.quality_threshold,
            real_time_monitoring=quality_config.real_time_monitoring,
        )

    @pytest.fixture
    def test_metadata(self):
        """Create metadata for validation tests."""
        return DatasetMetadata(
            title="Test Dataset",
            description="Test dataset for quality validation",
            spatial=SpatialExtent(bbox=[-122.5, 37.7, -122.3, 37.9], crs="EPSG:4326"),
            temporal=TemporalExtent(
                start=datetime(2023, 1, 1), end=datetime(2023, 12, 31)
            ),
            lineage=DataLineage(
                source="test_source", process="test_process", created_by="test_system"
            ),
        )

    @pytest.mark.asyncio
    async def test_validate_dataset(self, quality_manager, test_metadata):
        """Test dataset validation."""
        dataset_id = "test_dataset_123"

        data = pd.DataFrame(
            {
                "temperature": np.random.normal(20, 5, 1000),
                "humidity": np.random.normal(60, 10, 1000),
                "latitude": np.random.normal(37.7, 0.1, 1000),
                "longitude": np.random.normal(-122.4, 0.1, 1000),
            }
        )
        quality_manager.register_dataset(dataset_id, data, test_metadata)

        report = await quality_manager.validate_dataset(dataset_id)

        assert report.dataset_id == dataset_id
        assert 0.0 <= report.overall_score <= 1.0
        assert report.checks
        assert len(report.checks) > 0

    def test_improvement_recommendations(self, quality_manager):
        """Test improvement recommendations."""
        # Create a report with issues
        report = DataQualityReport(
            dataset_id="test_dataset",
            overall_score=0.6,  # Below threshold
            checks={
                "completeness": QualityCheck(
                    score=0.7,
                    status=QualityStatus.WARNING,
                    issues=[
                        {"type": "missing_values", "message": "Missing data found"}
                    ],
                ),
                "accuracy": QualityCheck(
                    score=0.4,
                    status=QualityStatus.FAIL,
                    issues=[{"type": "outliers", "message": "Many outliers detected"}],
                ),
            },
            recommendations=[],
        )

        recommendations = quality_manager.get_improvement_recommendations(report)

        assert len(recommendations) > 0
        assert any("quality" in rec.lower() for rec in recommendations)
        assert any("completeness" in rec.lower() for rec in recommendations)

    def test_quality_trends(self, quality_manager):
        """Test quality trend analysis."""
        # Add some mock reports to history
        for i in range(5):
            report = DataQualityReport(
                dataset_id=f"dataset_{i}",
                overall_score=0.7 + i * 0.05,  # Improving trend
                checks={},
                recommendations=[],
            )
            quality_manager.quality_history.append(report)

        trends = quality_manager.get_quality_trends(days=30)

        assert "average_score" in trends
        assert trends["reports_count"] == 5
        assert trends["score_trend"] == "improving"


class TestValidationConfig:
    """Test cases for ValidationConfig."""

    def test_config_initialization(self):
        """Test configuration initialization."""
        config = ValidationConfig(
            validation_rules=["completeness", "accuracy"],
            quality_threshold=0.85,
            strict_mode=True,
        )

        assert config.validation_rules == ["completeness", "accuracy"]
        assert config.quality_threshold == 0.85
        assert config.strict_mode is True

    def test_config_defaults(self):
        """Test configuration defaults."""
        config = ValidationConfig()

        assert config.validation_rules is not None
        assert len(config.validation_rules) > 0
        assert config.quality_threshold == 0.8
        assert config.strict_mode is False
        assert config.real_time_monitoring is True
