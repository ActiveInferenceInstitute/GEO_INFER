"""
Unit tests for API implementations.

This module tests the REST API and service implementations for
geospatial data management.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from fastapi.testclient import TestClient
from datetime import datetime, UTC

from geo_infer_data.api.rest_api import DataAPI
from geo_infer_data.api.service import DataService
from geo_infer_data.models.schemas import (
    DatasetMetadata,
    SpatialExtent,
    TemporalExtent,
    DataLineage,
)


class TestDataAPI:
    """Test cases for DataAPI."""

    @pytest.fixture
    def api_config(self):
        """Create test API configuration."""
        return {"host": "localhost", "port": 8001, "enable_cors": True}

    @pytest.fixture
    def data_api(self, api_config):
        """Create test DataAPI."""
        return DataAPI(
            config_path=None,
            host=api_config["host"],
            port=api_config["port"],
            enable_cors=api_config["enable_cors"],
        )

    def test_api_initialization(self, data_api):
        """Test API initialization."""
        assert data_api.host == "localhost"
        assert data_api.port == 8001
        assert data_api.enable_cors is True
        assert data_api.app is not None

    def test_api_routes(self, data_api):
        """Test API routes are properly configured."""
        # Test root endpoint
        client = TestClient(data_api.app)
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "name" in data
        assert "version" in data
        assert data["status"] == "running"

    def test_health_endpoint(self, data_api):
        """Test health check endpoint."""
        client = TestClient(data_api.app)
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "message" in data
        assert "checked_at" in data

    def test_list_datasets_endpoint(self, data_api):
        """Test list datasets endpoint."""
        client = TestClient(data_api.app)
        response = client.get("/datasets")
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)

        # Check pagination parameters work
        response = client.get("/datasets?page=1&limit=10")
        assert response.status_code == 200

        response = client.get("/datasets?type=vector&bbox=-122.5,37.7,-122.3,37.9")
        assert response.status_code == 200

    def test_create_dataset_endpoint(self, data_api):
        """Test create dataset endpoint."""
        client = TestClient(data_api.app)

        dataset_data = {
            "title": "Test Dataset",
            "description": "Test dataset for API testing",
            "type": "vector",
            "format": "geojson",
            "metadata": {
                "title": "Test Dataset",
                "spatial": {
                    "bbox": [-122.5, 37.7, -122.3, 37.9],
                    "crs": {"epsg_code": "EPSG:4326"},
                },
                "temporal": {
                    "start": "2023-01-01T00:00:00Z",
                    "end": "2023-12-31T23:59:59Z",
                },
                "lineage": {"source": "test", "process": "test", "created_by": "test"},
            },
        }

        response = client.post("/datasets", json=dataset_data)
        assert response.status_code == 201

        data = response.json()
        assert data["title"] == "Test Dataset"
        assert data["type"] == "vector"

    def test_get_dataset_endpoint(self, data_api):
        """Unknown dataset requests return an explicit not-found response."""
        client = TestClient(data_api.app)
        response = client.get("/datasets/test_dataset_123")
        assert response.status_code == 404

    def test_search_endpoint(self, data_api):
        """Test search datasets endpoint."""
        client = TestClient(data_api.app)

        # Test basic search
        response = client.get("/search?q=temperature")
        assert response.status_code == 200

        data = response.json()
        assert "results" in data
        assert "total" in data

        # Test search with filters
        response = client.get(
            "/search?bbox=-122.5,37.7,-122.3,37.9&temporal=2023-01-01/2023-12-31&tags=weather"
        )
        assert response.status_code == 200


class TestDataService:
    """Test cases for DataService."""

    @pytest.fixture
    def mock_storage(self):
        """Create mock storage service."""
        storage = Mock()
        storage.store_geospatial_data = AsyncMock(return_value="service_dataset_123")
        storage.get_storage_stats = Mock(
            return_value={
                "backends": ["postgresql"],
                "total_size": 1000000,
                "optimization_strategy": "balanced",
            }
        )
        return storage

    @pytest.fixture
    def mock_quality(self):
        """Create mock quality service."""
        quality = Mock()
        quality.validator.validate_data = AsyncMock(
            return_value=Mock(overall_score=0.9, status="pass", issues=[])
        )
        quality.validate_dataset = AsyncMock(
            return_value=Mock(
                dataset_id="test", overall_score=0.85, checks={}, recommendations=[]
            )
        )
        return quality

    @pytest.fixture
    def data_service(self, mock_storage, mock_quality):
        """Create test DataService."""
        return DataService(mock_storage, mock_quality)

    @pytest.fixture
    def test_metadata(self):
        """Create test metadata."""
        return DatasetMetadata(
            title="Test Dataset",
            description="Test dataset for service API testing",
            spatial=SpatialExtent(bbox=[-122.5, 37.7, -122.3, 37.9], crs="EPSG:4326"),
            temporal=TemporalExtent(
                start=datetime(2023, 1, 1, tzinfo=UTC),
                end=datetime(2023, 12, 31, tzinfo=UTC),
            ),
            lineage=DataLineage(
                source="test_source", process="test_process", created_by="test_system"
            ),
        )

    @pytest.mark.asyncio
    async def test_list_datasets(self, data_service):
        """Test listing datasets."""
        datasets = await data_service.list_datasets(limit=10)

        assert isinstance(datasets, list)
        assert len(datasets) <= 10

    @pytest.mark.asyncio
    async def test_get_dataset(self, data_service):
        """Unknown datasets return no metadata."""
        dataset = await data_service.get_dataset("dataset_123")
        assert dataset is None

    @pytest.mark.asyncio
    async def test_create_dataset(self, data_service, test_metadata):
        """Test creating dataset."""
        data = Mock()  # Mock data

        dataset_id = await data_service.create_dataset(test_metadata, data)

        assert dataset_id == "service_dataset_123"
        data_service.storage_service.store_geospatial_data.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_dataset_data(self, data_service):
        """Data access rejects an unregistered dataset."""
        # Mock storage query
        data_service.storage_service.adaptive_query = AsyncMock(
            return_value={"data": "mock"}
        )

        with pytest.raises(KeyError, match="not registered"):
            await data_service.get_dataset_data(
                "dataset_123", spatial_bounds=[-122.5, 37.7, -122.3, 37.9]
            )
        data_service.storage_service.adaptive_query.assert_not_called()

    def test_bbox_intersection(self, data_service):
        """Test bounding box intersection."""
        # Test intersecting bboxes
        bbox1 = [-122.5, 37.7, -122.3, 37.9]
        bbox2 = [-122.4, 37.8, -122.2, 38.0]

        assert data_service._bboxes_intersect(bbox1, bbox2) is True

        # Test non-intersecting bboxes
        bbox3 = [-123.0, 38.0, -122.8, 38.2]
        bbox4 = [-122.0, 37.0, -121.8, 37.2]

        assert data_service._bboxes_intersect(bbox3, bbox4) is False

    def test_access_pattern_analysis(self, data_service):
        """Test access pattern analysis."""
        # Mock access log
        data_service.access_log = [
            {
                "dataset_id": "dataset_123",
                "timestamp": datetime(2023, 1, 1, 10, 0),
                "spatial_bounds": [-122.5, 37.7, -122.3, 37.9],
                "format": "geojson",
            },
            {
                "dataset_id": "dataset_456",
                "timestamp": datetime(2023, 1, 1, 14, 0),
                "temporal_range": (datetime(2023, 1, 1), datetime(2023, 1, 31)),
                "format": "csv",
            },
        ]

        patterns = data_service.get_access_patterns()

        assert "total_accesses" in patterns
        assert patterns["total_accesses"] == 2
        assert "spatial_queries" in patterns
        assert "formats_requested" in patterns

    def test_storage_stats(self, data_service):
        """Test storage statistics."""
        stats = data_service.get_storage_stats()

        assert "backends" in stats
        assert "total_size" in stats
        assert stats["backends"] == ["postgresql"]

    def test_performance_optimization(self, data_service):
        """Test performance optimization."""
        # Mock access patterns with high usage
        data_service.access_log = [
            {"dataset_id": "frequent", "timestamp": datetime.now()}
        ] * 150

        optimizations = data_service.optimize_performance()

        assert "cache_optimization" in optimizations
        assert "storage_optimization" in optimizations
