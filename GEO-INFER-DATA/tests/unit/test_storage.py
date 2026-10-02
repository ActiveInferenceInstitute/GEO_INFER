"""
Unit tests for adaptive data storage.

This module tests the AdaptiveDataStorage class and related
functionality for storing and retrieving geospatial data.
"""

import pytest
from unittest.mock import AsyncMock, patch
import pandas as pd
import geopandas as gpd
import numpy as np
from datetime import datetime, UTC

from geo_infer_data.core.storage import (
    AdaptiveDataStorage,
    PostgreSQLBackend,
    MinIOBackend,
    LocalFileBackend,
    StorageConfig,
    OptimizationStrategy,
)
from geo_infer_data.models.schemas import (
    DatasetMetadata,
    SpatialExtent,
    TemporalExtent,
    DataLineage,
)


class TestAdaptiveDataStorage:
    """Test cases for AdaptiveDataStorage."""

    @pytest.fixture
    def storage_config(self):
        """Create test storage configuration."""
        return StorageConfig(
            storage_backends=["postgresql", "minio", "redis"],
            optimization_strategy=OptimizationStrategy.BALANCED,
            compression_enabled=True,
            indexing_strategy="h3",
            caching_enabled=True,
        )

    @pytest.fixture
    def storage_system(self, storage_config):
        """Create test storage system."""
        return AdaptiveDataStorage(
            storage_backends=storage_config.storage_backends,
            optimization_strategy=storage_config.optimization_strategy.value,
            compression_enabled=storage_config.compression_enabled,
            indexing_strategy=storage_config.indexing_strategy.value,
            caching_enabled=storage_config.caching_enabled,
        )

    @pytest.fixture
    def mock_metadata(self):
        """Create mock dataset metadata."""
        return DatasetMetadata(
            title="Test Dataset",
            description="Test dataset for unit tests",
            spatial=SpatialExtent(bbox=[-122.5, 37.7, -122.3, 37.9], crs="EPSG:4326"),
            temporal=TemporalExtent(
                start=datetime(2023, 1, 1, tzinfo=UTC),
                end=datetime(2023, 12, 31, tzinfo=UTC),
            ),
            lineage=DataLineage(
                source="test_source", process="test_process", created_by="test_system"
            ),
        )

    @pytest.fixture
    def mock_geodataframe(self):
        """Create mock GeoDataFrame."""
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
    def mock_dataframe(self):
        """Create mock DataFrame."""
        return pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2023-01-01", periods=1000, freq="h", tz="UTC"
                ),
                "value": np.random.normal(100, 15, 1000),
            }
        )

    def test_storage_initialization(self, storage_system):
        """Test storage system initialization."""
        assert len(storage_system.backend_manager.backends) == 3
        assert "postgresql" in storage_system.backend_manager.backends
        assert "minio" in storage_system.backend_manager.backends
        assert "redis" in storage_system.backend_manager.backends
        assert storage_system.config.compression_enabled is True

    @pytest.mark.asyncio
    async def test_store_geodataframe(
        self, storage_system, mock_metadata, mock_geodataframe
    ):
        """Test storing GeoDataFrame."""
        # Mock backend store method
        storage_system.backend_manager.backends["postgresql"].store = AsyncMock(
            return_value="test_id_123"
        )

        data_id = await storage_system.store_geospatial_data(
            mock_geodataframe, mock_metadata
        )

        assert data_id == "test_id_123"
        assert data_id in storage_system.storage_stats

    @pytest.mark.asyncio
    async def test_store_dataframe(self, storage_system, mock_metadata, mock_dataframe):
        """Test storing regular DataFrame."""
        # Mock backend store method
        storage_system.backend_manager.backends["postgresql"].store = AsyncMock(
            return_value="test_id_456"
        )

        data_id = await storage_system.store_geospatial_data(
            mock_dataframe, mock_metadata
        )

        assert data_id == "test_id_456"

    @pytest.mark.asyncio
    async def test_adaptive_query(
        self, storage_system, mock_metadata, mock_geodataframe
    ):
        """Test adaptive querying over datasets stored in this process."""
        backend = storage_system.backend_manager.backends["postgresql"]
        backend.store = AsyncMock(return_value="test_id_123")

        await storage_system.store_geospatial_data(mock_geodataframe, mock_metadata)

        results = await storage_system.adaptive_query(
            spatial_bounds=[-122.5, 37.7, -122.3, 37.9],
        )

        assert isinstance(results, gpd.GeoDataFrame)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_adaptive_query_empty_store(self, storage_system):
        """An empty store returns an empty result instead of querying 'all'."""
        results = await storage_system.adaptive_query(
            spatial_bounds=[-122.5, 37.7, -122.3, 37.9],
        )
        assert results == []

    def test_backend_selection(self, storage_system, mock_geodataframe, mock_metadata):
        """Test optimal backend selection."""
        # Test with GeoDataFrame - should select PostgreSQL
        backend = storage_system._select_optimal_backend(
            mock_geodataframe, mock_metadata, None
        )
        assert backend in ["postgresql", "minio"]

        # Test with large dataset - should select MinIO
        large_data = gpd.GeoDataFrame(
            {"id": range(200000)},
            geometry=gpd.points_from_xy([0] * 200000, [0] * 200000),
        )
        backend = storage_system._select_optimal_backend(
            large_data, mock_metadata, None
        )
        assert backend == "minio"  # Large datasets go to object storage

    def test_access_pattern_analysis(self, storage_system):
        """Test access pattern analysis."""
        patterns = {
            "spatial_queries": [
                {"bbox": [-122.5, 37.7, -122.3, 37.9], "frequency": "high"}
            ],
            "temporal_queries": [
                {"start": datetime(2023, 1, 1), "end": datetime(2023, 12, 31)}
            ],
            "query_frequency": "medium",
        }

        storage_system._analyze_access_patterns("test_dataset", patterns)

        assert "test_dataset" in storage_system.access_patterns
        assert storage_system.access_patterns["test_dataset"].query_frequency == {
            "medium": 1
        }

    def test_optimization_for_patterns(self, storage_system):
        """Test storage optimization based on patterns."""
        patterns = {
            "frequent_dataset": {"frequent_queries": True, "batch_processing": False},
            "batch_dataset": {"frequent_queries": False, "batch_processing": True},
        }

        optimizations = storage_system.optimize_for_patterns(patterns)

        assert "actions" in optimizations
        assert "optimizations" in optimizations
        assert len(optimizations["actions"]) > 0

    def test_storage_stats(self, storage_system):
        """Test storage statistics."""
        stats = storage_system.get_storage_stats()

        assert "backends" in stats
        assert "total_size" in stats
        assert "optimization_strategy" in stats
        assert stats["compression_enabled"] is True
        assert stats["caching_enabled"] is True


class TestPostgreSQLBackend:
    """Test cases for PostgreSQLBackend."""

    @pytest.fixture
    def backend_config(self):
        """Create test backend configuration."""
        return {
            "host": "localhost",
            "port": 5432,
            "user": "test_user",
            "password": "test_password",
            "database": "test_db",
        }

    @pytest.fixture
    def backend(self, backend_config):
        """Create test PostgreSQL backend."""
        return PostgreSQLBackend(backend_config)

    def test_connection_string(self, backend, backend_config):
        """Test connection string generation."""
        expected = (
            f"postgresql://{backend_config['user']}:{backend_config['password']}"
            f"@{backend_config['host']}:{backend_config['port']}/{backend_config['database']}"
        )
        assert backend.connection_string == expected

    @pytest.mark.asyncio
    async def test_store_dataframe(self, backend):
        """Test storing DataFrame."""
        data = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
        metadata = DatasetMetadata(
            title="Test Data",
            spatial=SpatialExtent(bbox=[0, 0, 1, 1]),
            temporal=TemporalExtent(
                start=datetime(2023, 1, 1, tzinfo=UTC),
                end=datetime(2023, 1, 2, tzinfo=UTC),
            ),
            lineage=DataLineage(source="test", process="test", created_by="test"),
        )

        # Mock the actual storage implementation
        with patch.object(backend, "_store_dataframe") as mock_store:
            mock_store.return_value = None
            data_id = await backend.store(data, metadata)
            assert data_id.startswith("pg_")

    @pytest.mark.asyncio
    async def test_retrieve_data(self, backend):
        """Test data retrieval."""
        data_id = "test_id"
        query = {"spatial": [-122.5, 37.7, -122.3, 37.9]}

        # Mock retrieval
        with patch.object(backend, "_retrieve_dataframe") as mock_retrieve:
            mock_retrieve.return_value = pd.DataFrame({"id": [1, 2, 3]})
            result = await backend.retrieve(data_id, query)
            assert isinstance(result, pd.DataFrame)


class TestMinIOBackend:
    """Test cases for MinIOBackend."""

    @pytest.fixture
    def backend_config(self):
        """Create test backend configuration."""
        return {
            "endpoint": "localhost:9000",
            "access_key": "test_key",
            "secret_key": "test_secret",
            "bucket": "test-bucket",
        }

    @pytest.fixture
    def backend(self, backend_config):
        """Create test MinIO backend."""
        return MinIOBackend(backend_config)

    def test_backend_initialization(self, backend, backend_config):
        """Test backend initialization."""
        assert backend.endpoint == backend_config["endpoint"]
        assert backend.access_key == backend_config["access_key"]
        assert backend.secret_key == backend_config["secret_key"]
        assert backend.bucket == backend_config["bucket"]

    @pytest.mark.asyncio
    async def test_store_data(self, backend):
        """Test storing data in MinIO."""
        data = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
        metadata = DatasetMetadata(
            title="Test Data",
            spatial=SpatialExtent(bbox=[0, 0, 1, 1]),
            temporal=TemporalExtent(
                start=datetime(2023, 1, 1, tzinfo=UTC),
                end=datetime(2023, 1, 2, tzinfo=UTC),
            ),
            lineage=DataLineage(source="test", process="test", created_by="test"),
        )

        # Mock the storage implementation
        with patch.object(backend, "_store_to_minio") as mock_store:
            mock_store.return_value = "minio_test_id_123"
            data_id = await backend.store(data, metadata)
            assert data_id.startswith("minio_")


class TestLocalFileBackend:
    """Test cases for LocalFileBackend."""

    @pytest.fixture
    def backend_config(self):
        """Create test backend configuration."""
        return {"base_path": "/tmp/test_geo_data"}

    @pytest.fixture
    def backend(self, backend_config):
        """Create test local file backend."""
        return LocalFileBackend(backend_config)

    def test_backend_initialization(self, backend, backend_config):
        """Test backend initialization."""
        assert str(backend.base_path) == backend_config["base_path"]
        assert backend.base_path.exists()

    @pytest.mark.asyncio
    async def test_store_data(self, backend):
        """Test storing data locally."""
        data = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
        metadata = DatasetMetadata(
            title="Test Data",
            spatial=SpatialExtent(bbox=[0, 0, 1, 1]),
            temporal=TemporalExtent(
                start=datetime(2023, 1, 1, tzinfo=UTC),
                end=datetime(2023, 1, 2, tzinfo=UTC),
            ),
            lineage=DataLineage(source="test", process="test", created_by="test"),
        )

        data_id = await backend.store(data, metadata)

        assert data_id.startswith("local_")

        # Check if files were created
        data_file = backend._find_data_file(data_id)
        assert data_file is not None
        assert data_file.exists()

        # Check metadata file
        metadata_file = data_file.with_suffix(".json")
        assert metadata_file.exists()

    @pytest.mark.asyncio
    async def test_retrieve_data(self, backend):
        """Test retrieving data locally."""
        # First store some data
        data = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
        metadata = DatasetMetadata(
            title="Test Data",
            spatial=SpatialExtent(bbox=[0, 0, 1, 1]),
            temporal=TemporalExtent(
                start=datetime(2023, 1, 1, tzinfo=UTC),
                end=datetime(2023, 1, 2, tzinfo=UTC),
            ),
            lineage=DataLineage(source="test", process="test", created_by="test"),
        )

        data_id = await backend.store(data, metadata)

        # Now retrieve it
        retrieved_data = await backend.retrieve(data_id, {})

        assert isinstance(retrieved_data, pd.DataFrame)
        assert len(retrieved_data) == 3
        assert "id" in retrieved_data.columns

    @pytest.mark.asyncio
    async def test_delete_data(self, backend):
        """Test deleting data locally."""
        # First store some data
        data = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
        metadata = DatasetMetadata(
            title="Test Data",
            spatial=SpatialExtent(bbox=[0, 0, 1, 1]),
            temporal=TemporalExtent(
                start=datetime(2023, 1, 1, tzinfo=UTC),
                end=datetime(2023, 1, 2, tzinfo=UTC),
            ),
            lineage=DataLineage(source="test", process="test", created_by="test"),
        )

        data_id = await backend.store(data, metadata)

        # Verify file exists
        data_file = backend._find_data_file(data_id)
        assert data_file is not None

        # Delete data
        deleted = await backend.delete(data_id)
        assert deleted is True

        # Verify file is gone
        data_file = backend._find_data_file(data_id)
        assert data_file is None


class TestDevCredentialBoundary:
    """GS19-43: dev-credential defaults warn on loopback, fail closed elsewhere."""

    def test_env_unset_loopback_warns(self, caplog, monkeypatch):
        import logging

        monkeypatch.delenv("GEO_INFER_POSTGRES_PASSWORD", raising=False)
        monkeypatch.delenv("GEO_INFER_MINIO_ACCESS_KEY", raising=False)
        monkeypatch.delenv("GEO_INFER_MINIO_SECRET_KEY", raising=False)

        with caplog.at_level(logging.WARNING, logger="geo_infer_data.core.storage"):
            AdaptiveDataStorage(storage_backends=["postgresql", "minio", "local"])

        messages = [record.getMessage() for record in caplog.records]
        assert any("GEO_INFER_POSTGRES_PASSWORD" in m for m in messages)
        assert any("GEO_INFER_MINIO_ACCESS_KEY" in m for m in messages)

    def test_env_set_produces_no_dev_credential_warning(self, caplog, monkeypatch):
        import logging

        monkeypatch.setenv("GEO_INFER_POSTGRES_PASSWORD", "test-only-password")
        monkeypatch.setenv("GEO_INFER_MINIO_ACCESS_KEY", "test-only-key")
        monkeypatch.setenv("GEO_INFER_MINIO_SECRET_KEY", "test-only-secret")

        with caplog.at_level(logging.WARNING, logger="geo_infer_data.core.storage"):
            AdaptiveDataStorage(storage_backends=["postgresql", "minio", "local"])

        assert not any(
            "development credential" in record.getMessage() for record in caplog.records
        )

    def test_non_loopback_host_fails_closed(self, monkeypatch):
        monkeypatch.delenv("GEO_INFER_POSTGRES_PASSWORD", raising=False)
        with patch.object(
            AdaptiveDataStorage,
            "_is_loopback_host",
            staticmethod(lambda host: False),
        ):
            with pytest.raises(RuntimeError, match="GEO_INFER_POSTGRES_PASSWORD"):
                AdaptiveDataStorage(storage_backends=["postgresql", "local"])
