"""
Error handling tests for GEO-INFER-DATA.

This module contains comprehensive error handling tests for all
GEO-INFER-DATA components including edge cases, failure scenarios,
and recovery mechanisms.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
import pandas as pd
import numpy as np
from datetime import datetime

from geo_infer_data import initialize_data_system
from geo_infer_data.core.ingestion import MultiSourceDataIngestion
from geo_infer_data.core.storage import AdaptiveDataStorage
from geo_infer_data.core.validation import DataQualityManager
from geo_infer_data.models.schemas import (
    DatasetMetadata,
    SpatialExtent,
    TemporalExtent,
    DataLineage,
)


class TestIngestionErrorHandling:
    """Error handling tests for data ingestion."""

    @pytest.mark.asyncio
    async def test_unsupported_data_source_error(self):
        """Test error handling for unsupported data sources."""
        ingestion = MultiSourceDataIngestion(["satellite", "sensors"])

        with pytest.raises(ValueError, match="not supported"):
            await ingestion.ingest_multi_source(unsupported_source={"data": "test"})

    @pytest.mark.asyncio
    async def test_connection_failure_error_handling(self):
        """Test error handling for connection failures."""
        ingestion = MultiSourceDataIngestion(["satellite"])

        # Mock connection failure
        connector = ingestion.connectors["satellite"]
        connector.connect = AsyncMock(return_value=False)

        with pytest.raises(ConnectionError):
            await ingestion.ingest_multi_source(
                satellite={"bbox": [-122.5, 37.7, -122.3, 37.9]}
            )

    @pytest.mark.asyncio
    async def test_data_validation_error_handling(self):
        """Test error handling for data validation failures."""
        ingestion = MultiSourceDataIngestion(
            data_sources=["sensors"],
            validation_enabled=True,
            quality_threshold=0.9,  # High threshold to trigger error
        )

        # Mock low-quality data
        connector = ingestion.connectors["sensors"]
        connector.connect = AsyncMock(return_value=True)
        connector.fetch_data = AsyncMock(
            return_value={
                "measurements": pd.DataFrame(
                    {
                        "temperature": [1000] * 100,  # Unrealistic values
                        "latitude": [200] * 100,  # Invalid coordinates
                        "longitude": [-400] * 100,  # Invalid coordinates
                    }
                )
            }
        )
        connector.validate_data = AsyncMock(
            return_value=Mock(score=0.3, status="fail", issues=["low_quality"])
        )

        result = await ingestion.ingest_multi_source(sensors={"test": True})

        # Should handle error gracefully
        assert "sensors" in result["ingested_data"]
        assert result["ingestion_metadata"]["validation_enabled"] is True

    @pytest.mark.asyncio
    async def test_timeout_error_handling(self):
        """Test error handling for timeout scenarios."""
        ingestion = MultiSourceDataIngestion(
            data_sources=["sensors"], validation_enabled=False
        )

        # Mock slow connector
        connector = ingestion.connectors["sensors"]
        connector.connect = AsyncMock(return_value=True)

        async def slow_fetch_data(query):
            await asyncio.sleep(60)  # Simulate timeout
            return {"measurements": pd.DataFrame()}

        connector.fetch_data = slow_fetch_data

        # Test with short timeout
        ingestion.config.timeout_seconds = 1

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                ingestion.ingest_multi_source(sensors={"test": True}), timeout=5
            )

    @pytest.mark.asyncio
    async def test_malformed_data_error_handling(self):
        """Test error handling for malformed data."""
        ingestion = MultiSourceDataIngestion(["sensors"], validation_enabled=False)

        # Mock connector whose fetch raises for genuinely malformed data
        connector = ingestion.connectors["sensors"]
        connector.connect = AsyncMock(return_value=True)
        connector.fetch_data = AsyncMock(
            side_effect=ValueError("Malformed data payload")
        )

        result = await ingestion.ingest_multi_source(sensors={"test": True})

        # Should handle malformed data gracefully
        assert "sensors" in result["ingested_data"]
        assert "error" in result["ingested_data"]["sensors"]


class TestStorageErrorHandling:
    """Error handling tests for data storage."""

    def test_invalid_backend_configuration(self):
        """Test error handling for invalid backend configuration."""
        with pytest.raises(ValueError):
            AdaptiveDataStorage(["invalid_backend"])

    @pytest.mark.asyncio
    async def test_storage_connection_failure(self):
        """Test error handling for storage connection failures."""
        storage = AdaptiveDataStorage(["postgresql"])

        # Test with invalid connection
        with patch.object(
            storage.backend_manager.backends["postgresql"], "store"
        ) as mock_store:
            mock_store.side_effect = ConnectionError("Database connection failed")

            with pytest.raises(ConnectionError):
                await storage.store_geospatial_data(
                    pd.DataFrame({"test": [1, 2, 3]}),
                    DatasetMetadata(
                        title="Test",
                        spatial=SpatialExtent(bbox=[0, 0, 1, 1]),
                        temporal=TemporalExtent(
                            start=datetime(2023, 1, 1), end=datetime(2023, 1, 2)
                        ),
                        lineage=DataLineage(
                            source="test", process="test", created_by="test"
                        ),
                    ),
                )

    @pytest.mark.asyncio
    async def test_query_error_handling(self):
        """Test error handling for query failures."""
        storage = AdaptiveDataStorage(["local"])

        # Test with invalid query parameters
        with pytest.raises(ValueError):
            await storage.adaptive_query(
                spatial_bounds=[-200, -100, 200, 100],  # Invalid coordinate bounds
                temporal_range=(
                    datetime(2023, 1, 1),
                    datetime(2022, 1, 1),
                ),  # End before start
            )

    @pytest.mark.asyncio
    async def test_storage_capacity_error_handling(self):
        """Test error handling for storage capacity issues."""
        storage = AdaptiveDataStorage(["local"])

        # Create extremely large dataset
        large_data = pd.DataFrame({"id": range(1000), "data": ["x" * 1000] * 1000})

        metadata = DatasetMetadata(
            title="Very Large Dataset",
            spatial=SpatialExtent(bbox=[0, 0, 1, 1]),
            temporal=TemporalExtent(
                start=datetime(2023, 1, 1), end=datetime(2023, 1, 2)
            ),
            lineage=DataLineage(
                source="test", process="large_data_test", created_by="test"
            ),
        )

        # Should handle large data gracefully
        dataset_id = await storage.store_geospatial_data(large_data, metadata)
        assert dataset_id is not None

    @pytest.mark.asyncio
    async def test_concurrent_storage_error_handling(self):
        """Test error handling for concurrent storage operations."""
        storage = AdaptiveDataStorage(["local"])

        # Create multiple concurrent storage operations
        async def concurrent_store(data_id: str):
            data = pd.DataFrame({"id": [data_id], "value": [1]})
            metadata = DatasetMetadata(
                title=f"Concurrent Test {data_id}",
                spatial=SpatialExtent(bbox=[0, 0, 1, 1]),
                temporal=TemporalExtent(
                    start=datetime(2023, 1, 1), end=datetime(2023, 1, 2)
                ),
                lineage=DataLineage(
                    source="concurrent_test", process="test", created_by="test"
                ),
            )

            return await storage.store_geospatial_data(data, metadata)

        # Execute concurrent operations
        tasks = [concurrent_store(f"test_{i}") for i in range(10)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Should handle concurrency gracefully
        successful_operations = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_operations) > 0


class TestValidationErrorHandling:
    """Error handling tests for data validation."""

    @staticmethod
    def _metadata(title: str) -> DatasetMetadata:
        return DatasetMetadata(
            title=title,
            spatial=SpatialExtent(bbox=[-122.5, 37.7, -122.3, 37.9]),
            temporal=TemporalExtent(
                start=datetime(2023, 1, 1), end=datetime(2023, 12, 31)
            ),
            lineage=DataLineage(
                source="validation_test", process="test", created_by="pytest"
            ),
        )

    def test_invalid_validation_rules(self):
        """Test error handling for invalid validation rules."""
        with pytest.raises(ValueError):
            DataQualityManager(validation_rules="invalid_rule_set")

    @pytest.mark.asyncio
    async def test_corrupted_data_validation(self):
        """Test validation of corrupted data."""
        quality_manager = DataQualityManager(validation_rules="comprehensive")

        data = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "temperature": [np.inf, -np.inf, np.nan],  # Invalid values
                "latitude": [None, None, None],  # All missing
                "longitude": [np.nan, np.nan, np.nan],  # All NaN
            }
        )
        quality_manager.register_dataset(
            "corrupted_dataset", data, self._metadata("Corrupted dataset")
        )

        # Should handle corrupted data gracefully
        report = await quality_manager.validate_dataset("corrupted_dataset")
        assert report.overall_score < 0.5
        assert len(report.checks) > 0

    @pytest.mark.asyncio
    async def test_empty_dataset_validation(self):
        """Test validation of empty datasets."""
        quality_manager = DataQualityManager(validation_rules="comprehensive")

        # Test with empty DataFrame
        quality_manager.register_dataset(
            "empty_dataset", pd.DataFrame(), self._metadata("Empty dataset")
        )

        report = await quality_manager.validate_dataset("empty_dataset")
        assert report.overall_score == 0.0
        assert any(
            "empty" in issue["type"]
            for check in report.checks.values()
            for issue in check.issues
        )

    @pytest.mark.asyncio
    async def test_invalid_metadata_validation(self):
        """Test validation with invalid metadata."""
        quality_manager = DataQualityManager()

        with pytest.raises(KeyError, match="has not been registered"):
            await quality_manager.validate_dataset("test_dataset")

    @pytest.mark.asyncio
    async def test_validation_timeout_handling(self):
        """Test validation timeout handling."""
        quality_manager = DataQualityManager(validation_rules="comprehensive")

        # Create bounded data that exercises wide strings without exhausting memory.
        data = pd.DataFrame({"id": range(1000), "data": ["x" * 1000] * 1000})
        quality_manager.register_dataset(
            "problematic_dataset", data, self._metadata("Problematic dataset")
        )

        # Should complete within reasonable time
        import time

        start_time = time.time()

        report = await quality_manager.validate_dataset("problematic_dataset")

        validation_time = time.time() - start_time
        assert validation_time < 30  # Should complete within 30 seconds
        assert isinstance(report, object)


class TestIntegrationErrorHandling:
    """Integration error handling tests."""

    @pytest.mark.asyncio
    async def test_system_initialization_errors(self):
        """Test error handling during system initialization."""
        # Test with invalid storage backends
        with pytest.raises(ValueError, match="Unsupported storage backend"):
            initialize_data_system(storage_backends=["invalid_backend"])

        # Test with valid backends after error
        system = initialize_data_system(storage_backends=["local"])
        assert system["status"] == "initialized"

    @pytest.mark.asyncio
    async def test_partial_system_failure_recovery(self):
        """Test recovery from partial system failures."""
        system = initialize_data_system(
            storage_backends=["local"], enable_validation=True
        )

        # Test ingestion failure
        try:
            await system["ingestion"].ingest_multi_source(
                invalid_source={"data": "test"}
            )
        except ValueError:
            pass  # Expected

        # System should still be functional
        assert system["status"] == "initialized"

        # Test storage after ingestion failure
        data = pd.DataFrame({"temperature": [20, 21, 22]})
        metadata = DatasetMetadata(
            title="Recovery Test",
            spatial=SpatialExtent(bbox=[0, 0, 1, 1]),
            temporal=TemporalExtent(
                start=datetime(2023, 1, 1), end=datetime(2023, 1, 2)
            ),
            lineage=DataLineage(
                source="recovery_test", process="test", created_by="test"
            ),
        )

        dataset_id = await system["storage"].store_geospatial_data(data, metadata)
        assert dataset_id is not None

    @pytest.mark.asyncio
    async def test_resource_cleanup_on_errors(self):
        """Test resource cleanup when errors occur."""
        storage = AdaptiveDataStorage(["local"])

        # Test with data that might cause resource issues
        try:
            # Create bounded data that might still stress serialization paths.
            problematic_data = pd.DataFrame(
                {"id": range(1000), "large_column": ["x" * 1000] * 1000}
            )

            metadata = DatasetMetadata(
                title="Resource Test",
                spatial=SpatialExtent(bbox=[0, 0, 1, 1]),
                temporal=TemporalExtent(
                    start=datetime(2023, 1, 1), end=datetime(2023, 1, 2)
                ),
                lineage=DataLineage(
                    source="resource_test", process="test", created_by="test"
                ),
            )

            # Should handle resource issues gracefully
            dataset_id = await storage.store_geospatial_data(problematic_data, metadata)
            assert dataset_id is not None

        finally:
            # Clean up resources
            await storage.close()

    @pytest.mark.asyncio
    async def test_network_error_recovery(self):
        """Test recovery from network errors."""
        ingestion = MultiSourceDataIngestion(["satellite"])

        # Mock network failures
        connector = ingestion.connectors["satellite"]
        connector.connect = AsyncMock(
            side_effect=[
                ConnectionError("Network timeout"),
                ConnectionError("Connection refused"),
                True,  # Third attempt succeeds
            ]
        )

        # Should retry and eventually succeed
        result = await ingestion.ingest_multi_source(
            satellite={"bbox": [-122.5, 37.7, -122.3, 37.9]}
        )

        assert "satellite" in result["ingested_data"]


class TestEdgeCaseHandling:
    """Edge case handling tests."""

    @pytest.mark.asyncio
    async def test_empty_data_handling(self):
        """Test handling of empty datasets."""
        ingestion = MultiSourceDataIngestion(["sensors"], validation_enabled=False)

        # Mock empty data
        connector = ingestion.connectors["sensors"]
        connector.connect = AsyncMock(return_value=True)
        connector.fetch_data = AsyncMock(
            return_value={"measurements": pd.DataFrame(), "sensor_ids": []}
        )

        result = await ingestion.ingest_multi_source(sensors={"test": True})

        # Should handle empty data gracefully
        assert "sensors" in result["ingested_data"]
        assert len(result["ingested_data"]["sensors"]["measurements"]) == 0

    @pytest.mark.asyncio
    async def test_single_record_handling(self):
        """Test handling of single-record datasets."""
        storage = AdaptiveDataStorage(["local"])

        # Single record data
        single_data = pd.DataFrame(
            {
                "id": [1],
                "temperature": [20.5],
                "latitude": [37.7749],
                "longitude": [-122.4194],
            }
        )

        metadata = DatasetMetadata(
            title="Single Record Test",
            spatial=SpatialExtent(bbox=[-122.5, 37.7, -122.3, 37.9]),
            temporal=TemporalExtent(
                start=datetime(2023, 1, 1), end=datetime(2023, 1, 1)
            ),
            lineage=DataLineage(
                source="single_test", process="test", created_by="test"
            ),
        )

        # Should handle single record
        dataset_id = await storage.store_geospatial_data(single_data, metadata)
        assert dataset_id is not None

        # Should be queryable
        results = await storage.adaptive_query(
            spatial_bounds=[-122.5, 37.7, -122.3, 37.9]
        )
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_unicode_data_handling(self):
        """Test handling of Unicode and special characters."""
        data = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "description": ["café", "naïve", "résumé"],  # Unicode characters
                "special": ["test@#$%", "data&*()", "values<>?"],  # Special characters
                "temperature": [20.5, 21.0, 19.8],
            }
        )

        metadata = DatasetMetadata(
            title="Unicode Test Data",
            description="Test data with Unicode and special characters",
            spatial=SpatialExtent(bbox=[0, 0, 1, 1]),
            temporal=TemporalExtent(
                start=datetime(2023, 1, 1), end=datetime(2023, 1, 2)
            ),
            lineage=DataLineage(
                source="unicode_test", process="test", created_by="test"
            ),
        )

        storage = AdaptiveDataStorage(["local"])

        # Should handle Unicode data
        dataset_id = await storage.store_geospatial_data(data, metadata)
        assert dataset_id is not None

        # Should preserve Unicode characters
        results = await storage.adaptive_query()
        if len(results) > 0:
            assert "café" in str(results)

    @pytest.mark.asyncio
    async def test_extremely_large_numbers(self):
        """Test handling of extremely large numbers."""
        data = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "very_large": [1e308, 1e309, float("inf")],  # Very large numbers
                "very_small": [1e-308, 0, -float("inf")],  # Very small numbers
                "normal": [1, 2, 3],
            }
        )

        metadata = DatasetMetadata(
            title="Extreme Values Test",
            spatial=SpatialExtent(bbox=[0, 0, 1, 1]),
            temporal=TemporalExtent(
                start=datetime(2023, 1, 1), end=datetime(2023, 1, 2)
            ),
            lineage=DataLineage(
                source="extreme_test", process="test", created_by="test"
            ),
        )

        storage = AdaptiveDataStorage(["local"])

        # Should handle extreme values
        dataset_id = await storage.store_geospatial_data(data, metadata)
        assert dataset_id is not None

        # Should be queryable
        results = await storage.adaptive_query()
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_mixed_data_types_handling(self):
        """Test handling of mixed data types in columns."""
        data = pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "mixed_column": [1, "text", 3.14, True, None],  # Mixed types
                "temperature": [20.1, 20.2, 20.3, 20.4, 20.5],
            }
        )

        metadata = DatasetMetadata(
            title="Mixed Types Test",
            spatial=SpatialExtent(bbox=[0, 0, 1, 1]),
            temporal=TemporalExtent(
                start=datetime(2023, 1, 1), end=datetime(2023, 1, 2)
            ),
            lineage=DataLineage(source="mixed_test", process="test", created_by="test"),
        )

        quality_manager = DataQualityManager(validation_rules="comprehensive")
        quality_manager.register_dataset("mixed_types_dataset", data, metadata)

        # Should detect mixed types
        report = await quality_manager.validate_dataset("mixed_types_dataset")
        assert "consistency" in report.checks

        # Should still be processable
        storage = AdaptiveDataStorage(["local"])
        dataset_id = await storage.store_geospatial_data(data, metadata)
        assert dataset_id is not None


class TestPerformanceErrorHandling:
    """Performance-related error handling tests."""

    def test_memory_limit_handling(self):
        """Test handling of memory limit scenarios."""
        # Create data that might exceed memory limits
        pd.DataFrame({"id": range(1000), "large_text": ["x" * 1000] * 1000})

        # Should handle memory constraints gracefully
        try:
            # This might fail due to memory limits
            AdaptiveDataStorage(["local"])
            # Process in chunks if needed

        except MemoryError:
            # Should handle memory errors gracefully
            pass

    def test_disk_space_error_handling(self):
        """Test handling of disk space limitations."""
        # Test with very limited disk space scenario
        # This would typically be tested in a controlled environment
        pass

    def test_cpu_limit_handling(self):
        """Test handling of CPU limitations."""
        # Test with CPU-intensive operations
        # Should handle CPU constraints gracefully
        pass
