"""
End-to-end integration tests for GEO-INFER-DATA.

This module tests complete workflows from data ingestion through
storage, validation, and processing to ensure all components work
together seamlessly.
"""

import pytest
import asyncio
import copy
import pandas as pd
import geopandas as gpd
import numpy as np
from datetime import datetime

from geo_infer_data import (
    initialize_data_system,
    MultiSourceDataIngestion,
    AdaptiveDataStorage,
    DataQualityManager,
)
from geo_infer_data.core.ingestion import DataSourceConnector
from geo_infer_data.models.schemas import (
    DatasetMetadata,
    SpatialExtent,
    TemporalExtent,
    DataLineage,
)


class PayloadConnector(DataSourceConnector):
    """Deterministic in-memory connector for orchestration tests."""

    def __init__(self, payload):
        super().__init__({"source": "integration-fixture"})
        self.payload = payload

    async def connect(self) -> bool:
        return True

    async def fetch_data(self, query):
        del query
        return copy.deepcopy(self.payload)


class TestEndToEndWorkflows:
    """End-to-end workflow integration tests."""

    @pytest.fixture
    async def data_system(self):
        """Initialize complete data system for testing."""
        return initialize_data_system(
            storage_backends=["local"], enable_validation=True
        )

    @pytest.fixture
    def mock_environmental_data(self):
        """Create mock environmental monitoring data."""
        n_records = 1000

        # Create realistic environmental data
        data = {
            "sensor_id": [f"sensor_{i % 10}" for i in range(n_records)],
            "timestamp": pd.date_range("2023-01-01", periods=n_records, freq="h"),
            "temperature": np.random.normal(20, 5, n_records),
            "humidity": np.random.normal(60, 10, n_records),
            "air_quality": np.random.normal(50, 15, n_records),
            "latitude": np.random.normal(37.7749, 0.1, n_records),
            "longitude": np.random.normal(-122.4194, 0.1, n_records),
            "wind_speed": np.random.normal(5, 2, n_records),
            "precipitation": np.random.exponential(0.1, n_records),
        }

        return pd.DataFrame(data)

    @pytest.fixture
    def mock_metadata(self):
        """Create mock dataset metadata."""
        return DatasetMetadata(
            title="Environmental Monitoring Test Data",
            description="Test dataset for integration testing",
            spatial=SpatialExtent(bbox=[-122.6, 37.6, -122.2, 38.0], crs="EPSG:4326"),
            temporal=TemporalExtent(
                start=datetime(2023, 1, 1),
                end=datetime(2023, 12, 31),
                resolution="PT1H",
            ),
            lineage=DataLineage(
                source="integration_test",
                process="automated_testing",
                created_by="test_system",
            ),
            keywords=["environment", "monitoring", "test"],
            contact={"organization": "Test Organization", "email": "test@example.com"},
        )

    @pytest.mark.asyncio
    async def test_complete_ingestion_to_storage_workflow(
        self, mock_environmental_data, mock_metadata
    ):
        """Test complete workflow from ingestion to storage."""
        # Initialize system
        system = initialize_data_system(
            storage_backends=["local"], enable_validation=True
        )

        ingestion = system["ingestion"]
        storage = system["storage"]
        quality_manager = system["quality_manager"]
        ingestion.connectors["sensors"] = PayloadConnector(
            {"measurements": mock_environmental_data}
        )

        # Step 1: Ingest data
        ingestion_result = await ingestion.ingest_multi_source(
            sensors={
                "time_range": "2023-01-01/2023-01-31",
                "sensor_types": ["temperature", "humidity", "air_quality"],
            }
        )

        # Verify ingestion results
        assert "ingested_data" in ingestion_result
        assert "quality_reports" in ingestion_result
        assert ingestion_result["ingestion_metadata"]["validation_enabled"] is True

        # Step 2: Validate and clean data
        cleaned_result = await ingestion.validate_and_clean(ingestion_result)

        # Verify cleaning results
        assert "cleaned_data" in cleaned_result
        assert "validation_summary" in cleaned_result
        assert cleaned_result["cleaning_metadata"]["sources_cleaned"] > 0

        # Step 3: Generate quality report
        quality_report = ingestion.generate_quality_report(cleaned_result)

        # Verify quality report
        assert "overall_score" in quality_report
        assert 0.0 <= quality_report["overall_score"] <= 1.0
        assert "recommendations" in quality_report

        # Step 4: Store data
        dataset_id = await storage.store_geospatial_data(
            mock_environmental_data,
            mock_metadata,
            access_patterns={
                "query_frequency": "high",
                "spatial_queries": [{"bbox": [-122.5, 37.7, -122.3, 37.9]}],
            },
        )

        # Verify storage
        assert dataset_id is not None
        assert dataset_id in storage.storage_stats

        # Step 5: Query stored data
        query_results = await storage.adaptive_query(
            spatial_bounds=[-122.5, 37.7, -122.3, 37.9],
            temporal_range=(datetime(2023, 1, 1), datetime(2023, 1, 31)),
            optimization_hints={"frequent_queries": True},
        )

        # Verify query results
        assert isinstance(query_results, (pd.DataFrame, gpd.GeoDataFrame))
        assert len(query_results) > 0

        # Step 6: Validate stored dataset
        quality_manager.register_dataset(
            dataset_id, mock_environmental_data, mock_metadata
        )
        stored_quality_report = await quality_manager.validate_dataset(dataset_id)

        # Verify validation
        assert stored_quality_report.overall_score >= 0.0
        assert stored_quality_report.dataset_id == dataset_id

        print("✅ Complete workflow test passed")
        print(f"   - Records processed: {len(mock_environmental_data)}")
        print(f"   - Dataset ID: {dataset_id}")
        print(f"   - Quality score: {stored_quality_report.overall_score:.2f}")

    @pytest.mark.asyncio
    async def test_multi_source_integration(self):
        """Test integration of multiple data sources."""
        # Initialize ingestion with multiple sources
        ingestion = MultiSourceDataIngestion(
            data_sources=["satellite", "sensors", "crowdsourced"],
            validation_enabled=True,
            parallel_processing=True,
        )

        # Use configured payloads so this orchestration test has no network dependency.
        satellite_data = {
            "bbox": [-122.5, 37.7, -122.3, 37.9],
            "date_range": "2023-01-01/2023-01-31",
            "bands": ["red", "green", "blue"],
        }

        sensor_data = {
            "time_range": "2023-01-01/2023-01-31",
            "sensor_types": ["temperature", "humidity"],
        }

        crowdsourced_data = {
            "category": "environment",
            "time_range": "2023-01-01/2023-01-31",
        }

        ingestion.connectors["satellite"] = PayloadConnector(
            {
                "imagery": np.ones((2, 2, 3)),
                "metadata": {"source": "integration-fixture"},
            }
        )
        ingestion.connectors["sensors"] = PayloadConnector(
            {
                "measurements": pd.DataFrame(
                    {
                        "timestamp": pd.date_range("2023-01-01", periods=2, freq="h"),
                        "temperature": [20.0, 21.0],
                        "latitude": [37.7, 37.71],
                        "longitude": [-122.4, -122.41],
                    }
                )
            }
        )
        ingestion.connectors["crowdsourced"] = PayloadConnector(
            {
                "reports": pd.DataFrame(
                    {
                        "timestamp": pd.date_range("2023-01-01", periods=2, freq="h"),
                        "latitude": [37.7, 37.71],
                        "longitude": [-122.4, -122.41],
                        "category": ["environment", "environment"],
                    }
                )
            }
        )

        # Ingest from multiple sources
        result = await ingestion.ingest_multi_source(
            satellite=satellite_data,
            sensors=sensor_data,
            crowdsourced=crowdsourced_data,
        )

        # Verify multi-source integration
        assert result["ingestion_metadata"]["sources_processed"] == 3
        assert "satellite" in result["ingested_data"]
        assert "sensors" in result["ingested_data"]
        assert "crowdsourced" in result["ingested_data"]

        # Verify quality validation for each source
        for source_name in ["satellite", "sensors", "crowdsourced"]:
            assert source_name in result["quality_reports"]

        print("✅ Multi-source integration test passed")
        print(
            f"   - Sources processed: {result['ingestion_metadata']['sources_processed']}"
        )
        print(
            f"   - Parallel processing: {result['ingestion_metadata']['parallel_processing']}"
        )

    @pytest.mark.asyncio
    async def test_storage_backend_integration(
        self, mock_environmental_data, mock_metadata
    ):
        """Test integration across multiple storage backends."""
        # Initialize storage with multiple backends
        storage = AdaptiveDataStorage(
            storage_backends=["local", "postgresql", "minio"],
            optimization_strategy="access_pattern_based",
        )

        # Store data in primary backend
        primary_id = await storage.store_geospatial_data(
            mock_environmental_data,
            mock_metadata,
            access_patterns={"query_frequency": "high"},
        )

        # Query from primary backend
        results = await storage.adaptive_query(
            spatial_bounds=[-122.5, 37.7, -122.3, 37.9]
        )

        # Verify backend selection and optimization
        assert primary_id in storage.storage_stats
        assert isinstance(results, (pd.DataFrame, gpd.GeoDataFrame))

        # Test backend switching based on query characteristics
        _realtime_results = await storage.adaptive_query(
            optimization_hints={"real_time": True}
        )

        # Verify optimization applied
        stats = storage.get_storage_stats()
        assert len(stats["backends"]) >= 1

        print("✅ Storage backend integration test passed")
        print(f"   - Backends available: {len(stats['backends'])}")
        print(f"   - Data stored: {primary_id}")
        print(f"   - Query results: {len(results)} records")

    @pytest.mark.asyncio
    async def test_quality_validation_integration(
        self, mock_environmental_data, mock_metadata
    ):
        """Test integration of quality validation across the system."""
        # Initialize quality management
        quality_manager = DataQualityManager(
            validation_rules="comprehensive",
            quality_threshold=0.8,
            real_time_monitoring=True,
        )

        # Test individual validation components
        completeness_check = await quality_manager.validator.validate_data(
            mock_environmental_data
        )
        assert completeness_check.score >= 0.0

        # Test coordinate validation
        coord_check = quality_manager.validator.validate_coordinates(
            mock_environmental_data
        )
        assert coord_check.score >= 0.0

        # Test temporal validation
        temporal_check = quality_manager.validator.validate_temporal_data(
            mock_environmental_data
        )
        assert temporal_check.score >= 0.0

        # Test comprehensive validation
        quality_manager.register_dataset(
            "test_dataset", mock_environmental_data, mock_metadata
        )
        comprehensive_report = await quality_manager.validate_dataset("test_dataset")

        # Verify comprehensive validation
        assert comprehensive_report.overall_score >= 0.0
        assert len(comprehensive_report.checks) > 0
        assert isinstance(comprehensive_report.recommendations, list)

        # Test trend analysis
        trends = quality_manager.get_quality_trends(days=7)
        assert "average_score" in trends

        print("✅ Quality validation integration test passed")
        print(f"   - Comprehensive score: {comprehensive_report.overall_score:.2f}")
        print(f"   - Validation checks: {len(comprehensive_report.checks)}")
        print(f"   - Recommendations: {len(comprehensive_report.recommendations)}")

    @pytest.mark.asyncio
    async def test_error_recovery_integration(self):
        """Test error recovery across integrated components."""
        # Test with invalid configuration
        with pytest.raises(ValueError, match="Unsupported storage backend"):
            initialize_data_system(
                storage_backends=["invalid_backend"], enable_validation=True
            )

        # Test with valid configuration after error
        system = initialize_data_system(
            storage_backends=["local"], enable_validation=True
        )
        system["ingestion"].connectors["sensors"] = PayloadConnector(
            {
                "measurements": pd.DataFrame(
                    {
                        "timestamp": pd.date_range("2023-01-01", periods=2, freq="h"),
                        "temperature": [20.0, 21.0],
                        "latitude": [37.7, 37.71],
                        "longitude": [-122.4, -122.41],
                    }
                )
            }
        )

        assert system["status"] == "initialized"

        # Test partial failure recovery
        ingestion = system["ingestion"]

        # Test with unsupported source
        with pytest.raises(ValueError):
            await ingestion.ingest_multi_source(unsupported_source={"data": "test"})

        print("✅ Error recovery integration test passed")
        print("   - Invalid backend handled correctly")
        print("   - Unsupported source handled correctly")
        print("   - System remains stable after errors")

    @pytest.mark.asyncio
    async def test_performance_integration(
        self, mock_environmental_data, mock_metadata
    ):
        """Test performance characteristics across integrated components."""
        # Initialize system with performance monitoring
        system = initialize_data_system(
            storage_backends=["local"], enable_validation=True
        )
        system["ingestion"].connectors["sensors"] = PayloadConnector(
            {"measurements": mock_environmental_data}
        )

        # Test ingestion performance
        import time

        start_time = time.time()

        _ingestion_result = await system["ingestion"].ingest_multi_source(
            sensors={"time_range": "2023-01-01/2023-01-31"}
        )

        ingestion_time = time.time() - start_time

        # Test storage performance
        start_time = time.time()

        dataset_id = await system["storage"].store_geospatial_data(
            mock_environmental_data, mock_metadata
        )
        system["quality_manager"].register_dataset(
            dataset_id, mock_environmental_data, mock_metadata
        )

        storage_time = time.time() - start_time

        # Test query performance
        start_time = time.time()

        _query_results = await system["storage"].adaptive_query(
            spatial_bounds=[-122.5, 37.7, -122.3, 37.9]
        )

        query_time = time.time() - start_time

        # Test validation performance
        start_time = time.time()

        quality_report = await system["quality_manager"].validate_dataset(dataset_id)

        validation_time = time.time() - start_time

        # Verify performance metrics
        assert ingestion_time < 30  # Should complete within 30 seconds
        assert storage_time < 10  # Storage should be fast
        assert query_time < 5  # Queries should be fast
        assert validation_time < 15  # Validation should complete quickly

        print("✅ Performance integration test passed")
        print(f"   - Ingestion time: {ingestion_time:.2f}s")
        print(f"   - Storage time: {storage_time:.2f}s")
        print(f"   - Query time: {query_time:.2f}s")
        print(f"   - Validation time: {validation_time:.2f}s")
        print(f"   - Final quality: {quality_report.overall_score:.2f}")


class TestCrossComponentIntegration:
    """Cross-component integration tests."""

    @pytest.mark.asyncio
    async def test_data_flow_ingestion_to_storage(self):
        """Test data flow from ingestion to storage."""
        # Initialize components
        ingestion = MultiSourceDataIngestion(["sensors"], validation_enabled=True)
        storage = AdaptiveDataStorage(["local"], optimization_strategy="balanced")

        # Create test data
        test_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=100, freq="h"),
                "temperature": np.random.normal(20, 5, 100),
                "latitude": np.random.normal(37.7, 0.1, 100),
                "longitude": np.random.normal(-122.4, 0.1, 100),
            }
        )

        metadata = DatasetMetadata(
            title="Test Data Flow",
            spatial=SpatialExtent(bbox=[-122.5, 37.6, -122.3, 37.8]),
            temporal=TemporalExtent(
                start=datetime(2023, 1, 1), end=datetime(2023, 1, 5)
            ),
            lineage=DataLineage(
                source="test", process="integration_test", created_by="test"
            ),
        )

        # Test data flow
        # 1. Ingest data
        ingestion.connectors["sensors"] = PayloadConnector({"measurements": test_data})
        result = await ingestion.ingest_multi_source(sensors={"test_data": test_data})
        assert "sensors" in result["ingested_data"]

        # 2. Store data
        dataset_id = await storage.store_geospatial_data(test_data, metadata)
        assert dataset_id is not None

        # 3. Query data
        query_results = await storage.adaptive_query(
            spatial_bounds=[-122.5, 37.6, -122.3, 37.8]
        )
        assert len(query_results) > 0

        print("✅ Data flow integration test passed")

    @pytest.mark.asyncio
    async def test_quality_to_storage_integration(self):
        """Test integration between quality validation and storage."""
        storage = AdaptiveDataStorage(["local"])
        quality_manager = DataQualityManager(validation_rules="comprehensive")

        # Create test data with known quality issues
        data_with_issues = pd.DataFrame(
            {
                "id": range(100),
                "temperature": [100 if i < 10 else 20 for i in range(100)],  # Outliers
                "latitude": [
                    200 if i < 5 else 37.7 for i in range(100)
                ],  # Invalid coords
                "longitude": [
                    -300 if i < 5 else -122.4 for i in range(100)
                ],  # Invalid coords
            }
        )

        metadata = DatasetMetadata(
            title="Data with Quality Issues",
            spatial=SpatialExtent(bbox=[-122.5, 37.6, -122.3, 37.8]),
            temporal=TemporalExtent(
                start=datetime(2023, 1, 1), end=datetime(2023, 1, 2)
            ),
            lineage=DataLineage(
                source="test", process="quality_test", created_by="test"
            ),
        )

        # Test quality assessment
        quality_manager.register_dataset("test_dataset", data_with_issues, metadata)
        quality_report = await quality_manager.validate_dataset("test_dataset")
        assert quality_report.overall_score < 0.9  # Should detect issues

        # Test storage with quality metadata
        dataset_id = await storage.store_geospatial_data(
            data_with_issues, metadata, access_patterns={"query_frequency": "low"}
        )

        # Verify quality metadata is stored
        assert dataset_id in storage.storage_stats

        print("✅ Quality to storage integration test passed")
        print(f"   - Quality score: {quality_report.overall_score:.2f}")
        print(f"   - Issues detected: {len(quality_report.checks)}")

    @pytest.mark.asyncio
    async def test_etl_to_quality_integration(self):
        """Test integration between ETL pipeline and quality management."""
        from geo_infer_data.core.pipeline import IntelligentETLPipeline

        # Initialize pipeline with quality integration
        pipeline = IntelligentETLPipeline(
            workflow_config=None,
            error_recovery="intelligent_retry",
            monitoring_enabled=True,
        )

        quality_manager = DataQualityManager(validation_rules="comprehensive")

        # Create test data
        raw_data = pd.DataFrame(
            {
                "raw_temperature": np.random.normal(20, 15, 1000),  # Some outliers
                "raw_humidity": np.random.normal(60, 25, 1000),  # Some outliers
                "latitude": np.random.normal(37.7, 0.1, 1000),
                "longitude": np.random.normal(-122.4, 0.1, 1000),
            }
        )

        # Test ETL execution
        result = await pipeline.execute_workflow(
            source_data=raw_data,
            target_storage={"type": "memory"},
            transformation_rules={
                "clean_outliers": True,
                "normalize_values": True,
                "validate_coordinates": True,
            },
        )

        # Verify ETL results
        assert result["status"] == "completed"
        assert result["transformed_records"] > 0

        # Test quality assessment of ETL output
        processed_data = result.get("processed_data")
        if processed_data is not None:
            quality_manager.register_dataset(
                "processed_dataset",
                processed_data,
                DatasetMetadata(
                    title="Processed dataset",
                    lineage=DataLineage(
                        source="etl", process="integration_test", created_by="test"
                    ),
                ),
            )
            quality_report = await quality_manager.validate_dataset("processed_dataset")
            assert quality_report.overall_score >= 0.0

        print("✅ ETL to quality integration test passed")
        print(f"   - ETL status: {result['status']}")
        print(f"   - Records processed: {result['transformed_records']}")


class TestDataFlowIntegration:
    """Data flow integration tests."""

    @pytest.mark.asyncio
    async def test_geospatial_data_pipeline(self):
        """Test complete geospatial data pipeline."""
        # Initialize system
        system = initialize_data_system(
            storage_backends=["local"], enable_validation=True
        )

        # Create geospatial data
        gdf = gpd.GeoDataFrame(
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

        metadata = DatasetMetadata(
            title="Geospatial Test Data",
            spatial=SpatialExtent(bbox=[-122.5, 37.6, -122.3, 37.8]),
            temporal=TemporalExtent(
                start=datetime(2023, 1, 1), end=datetime(2023, 1, 2)
            ),
            lineage=DataLineage(
                source="test", process="geospatial_test", created_by="test"
            ),
        )

        # Test complete pipeline
        dataset_id = await system["storage"].store_geospatial_data(gdf, metadata)
        system["quality_manager"].register_dataset(dataset_id, gdf, metadata)

        # Test spatial queries
        spatial_results = await system["storage"].adaptive_query(
            spatial_bounds=[-122.5, 37.6, -122.3, 37.8]
        )

        # Test quality validation
        quality_report = await system["quality_manager"].validate_dataset(dataset_id)

        # Verify results
        assert len(spatial_results) > 0
        assert quality_report.overall_score >= 0.0

        print("✅ Geospatial data pipeline test passed")
        print(f"   - Dataset stored: {dataset_id}")
        print(f"   - Spatial query results: {len(spatial_results)}")
        print(f"   - Quality score: {quality_report.overall_score:.2f}")

    @pytest.mark.asyncio
    async def test_temporal_data_pipeline(self):
        """Test complete temporal data pipeline."""
        system = initialize_data_system(
            storage_backends=["local"], enable_validation=True
        )

        # Create time series data
        ts_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=1000, freq="h"),
                "temperature": np.random.normal(20, 5, 1000),
                "humidity": np.random.normal(60, 10, 1000),
            }
        )

        metadata = DatasetMetadata(
            title="Time Series Test Data",
            temporal=TemporalExtent(
                start=datetime(2023, 1, 1), end=datetime(2023, 1, 5), resolution="PT1H"
            ),
            lineage=DataLineage(
                source="test", process="temporal_test", created_by="test"
            ),
        )

        # Test temporal pipeline
        dataset_id = await system["storage"].store_geospatial_data(ts_data, metadata)
        system["quality_manager"].register_dataset(dataset_id, ts_data, metadata)

        # Test temporal queries
        temporal_results = await system["storage"].adaptive_query(
            temporal_range=(datetime(2023, 1, 1), datetime(2023, 1, 2))
        )

        # Test quality validation
        quality_report = await system["quality_manager"].validate_dataset(dataset_id)

        # Verify results
        assert len(temporal_results) > 0
        assert quality_report.overall_score >= 0.0

        print("✅ Temporal data pipeline test passed")
        print(f"   - Time series stored: {dataset_id}")
        print(f"   - Temporal query results: {len(temporal_results)}")
        print(f"   - Quality score: {quality_report.overall_score:.2f}")


class TestPerformanceIntegration:
    """Performance integration tests."""

    @pytest.mark.asyncio
    async def test_large_dataset_performance(self):
        """Test performance with large datasets."""
        # Create large dataset
        n_records = 50000

        large_data = pd.DataFrame(
            {
                "id": range(n_records),
                "timestamp": pd.date_range(
                    "2023-01-01", periods=n_records, freq="1min"
                ),
                "temperature": np.random.normal(20, 5, n_records),
                "humidity": np.random.normal(60, 10, n_records),
                "latitude": np.random.normal(37.7, 0.1, n_records),
                "longitude": np.random.normal(-122.4, 0.1, n_records),
            }
        )

        metadata = DatasetMetadata(
            title="Large Dataset Performance Test",
            description="Performance test with large dataset",
            spatial=SpatialExtent(bbox=[-122.5, 37.6, -122.3, 37.8]),
            temporal=TemporalExtent(
                start=datetime(2023, 1, 1), end=datetime(2023, 1, 2)
            ),
            lineage=DataLineage(
                source="performance_test", process="large_data", created_by="test"
            ),
        )

        # Test performance with large data
        import time

        start_time = time.time()

        storage = AdaptiveDataStorage(["local"])
        _dataset_id = await storage.store_geospatial_data(large_data, metadata)

        storage_time = time.time() - start_time

        # Test query performance
        start_time = time.time()
        results = await storage.adaptive_query(
            spatial_bounds=[-122.5, 37.6, -122.3, 37.8]
        )
        query_time = time.time() - start_time

        # Verify performance requirements
        assert storage_time < 60  # Should store within 60 seconds
        assert query_time < 10  # Queries should be fast
        assert len(results) > 0

        print("✅ Large dataset performance test passed")
        print(f"   - Records: {n_records}")
        print(f"   - Storage time: {storage_time:.2f}s")
        print(f"   - Query time: {query_time:.2f}s")
        print(f"   - Throughput: {n_records / storage_time:.0f} records/second")

    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test concurrent operations across components."""

        async def concurrent_ingestion(source_name: str, data: dict):
            """Simulate concurrent ingestion."""
            ingestion = MultiSourceDataIngestion([source_name])
            payloads = {
                "sensors": {"measurements": test_data},
                "satellite": {
                    "imagery": np.ones((2, 2, 3)),
                    "metadata": {"source": "integration-fixture"},
                },
            }
            ingestion.connectors[source_name] = PayloadConnector(payloads[source_name])
            result = await ingestion.ingest_multi_source(**{source_name: data})
            return result

        async def concurrent_storage(
            dataset_id: str, data: pd.DataFrame, metadata: DatasetMetadata
        ):
            """Simulate concurrent storage."""
            storage = AdaptiveDataStorage(["local"])
            stored_id = await storage.store_geospatial_data(data, metadata)
            return stored_id

        # Create test data
        test_data = pd.DataFrame(
            {
                "temperature": np.random.normal(20, 5, 1000),
                "latitude": np.random.normal(37.7, 0.1, 1000),
                "longitude": np.random.normal(-122.4, 0.1, 1000),
            }
        )

        metadata = DatasetMetadata(
            title="Concurrent Test Data",
            spatial=SpatialExtent(bbox=[-122.5, 37.6, -122.3, 37.8]),
            temporal=TemporalExtent(
                start=datetime(2023, 1, 1), end=datetime(2023, 1, 2)
            ),
            lineage=DataLineage(
                source="concurrent_test", process="parallel", created_by="test"
            ),
        )

        # Test concurrent operations
        start_time = asyncio.get_running_loop().time()

        tasks = [
            concurrent_ingestion("sensors", {"time_range": "2023-01-01/2023-01-31"}),
            concurrent_ingestion("satellite", {"bbox": [-122.5, 37.7, -122.3, 37.9]}),
            concurrent_storage("concurrent_test", test_data, metadata),
        ]

        results = await asyncio.gather(*tasks)

        total_time = asyncio.get_running_loop().time() - start_time

        # Verify concurrent execution
        assert len(results) == 3
        assert total_time < 30  # Should complete within 30 seconds

        print("✅ Concurrent operations test passed")
        print(f"   - Concurrent tasks: {len(tasks)}")
        print(f"   - Total time: {total_time:.2f}s")
        print(f"   - Parallel efficiency: {len(tasks) / total_time:.2f} tasks/second")
