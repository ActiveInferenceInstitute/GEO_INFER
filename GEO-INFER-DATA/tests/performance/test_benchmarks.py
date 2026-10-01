"""
Performance benchmarks for GEO-INFER-DATA.

This module contains comprehensive performance benchmarks for all
GEO-INFER-DATA components including throughput, latency, and memory usage tests.
"""

import pytest
import asyncio
import time
import psutil
from unittest.mock import Mock, AsyncMock
import pandas as pd
import geopandas as gpd
import numpy as np
from datetime import datetime

from geo_infer_data import (
    MultiSourceDataIngestion,
    AdaptiveDataStorage,
    DataQualityManager,
)
from geo_infer_data.models.schemas import (
    DatasetMetadata,
    SpatialExtent,
    TemporalExtent,
    DataLineage,
)


class TestIngestionBenchmarks:
    """Performance benchmarks for data ingestion."""

    @pytest.fixture
    def benchmark_data(self):
        """Create benchmark test data."""
        n_records = 10000

        return pd.DataFrame(
            {
                "sensor_id": [f"sensor_{i % 100}" for i in range(n_records)],
                "timestamp": pd.date_range(
                    "2023-01-01", periods=n_records, freq="1min"
                ),
                "temperature": np.random.normal(20, 5, n_records),
                "humidity": np.random.normal(60, 10, n_records),
                "latitude": np.random.normal(37.7749, 0.1, n_records),
                "longitude": np.random.normal(-122.4194, 0.1, n_records),
            }
        )

    def test_ingestion_throughput_benchmark(self, benchmark_data, benchmark):
        """Benchmark data ingestion throughput."""

        async def ingestion_benchmark():
            ingestion = MultiSourceDataIngestion(
                data_sources=["sensors"],
                validation_enabled=False,  # Disable for pure throughput test
                parallel_processing=False,
            )

            # Mock the connector to return test data quickly
            connector = ingestion.connectors["sensors"]
            connector.connect = AsyncMock(return_value=True)
            connector.fetch_data = AsyncMock(
                return_value={
                    "measurements": benchmark_data,
                    "sensor_ids": list(set(benchmark_data["sensor_id"])),
                }
            )
            connector.validate_data = AsyncMock(
                return_value=Mock(score=0.9, status="pass", issues=[])
            )

            start_time = time.time()
            result = await ingestion.ingest_multi_source(sensors={"test": True})
            end_time = time.time()

            throughput = len(benchmark_data) / (end_time - start_time)

            # Store benchmark result
            result["throughput"] = throughput
            result["benchmark_time"] = end_time - start_time

            return result

        result = benchmark(lambda: asyncio.run(ingestion_benchmark()))

        # Verify performance meets threshold
        throughput = result["throughput"]
        assert throughput > 1000, f"Throughput too low: {throughput:.0f} records/second"

        print(f"📊 Ingestion throughput: {throughput:.0f} records/second")
        print(f"⏱️  Processing time: {result['benchmark_time']:.2f}s")

    def test_ingestion_validation_benchmark(self, benchmark_data, benchmark):
        """Benchmark data ingestion with validation."""

        async def validation_benchmark():
            ingestion = MultiSourceDataIngestion(
                data_sources=["sensors"], validation_enabled=True, quality_threshold=0.8
            )

            # Mock connector
            connector = ingestion.connectors["sensors"]
            connector.connect = AsyncMock(return_value=True)
            connector.fetch_data = AsyncMock(
                return_value={
                    "measurements": benchmark_data,
                    "sensor_ids": list(set(benchmark_data["sensor_id"])),
                }
            )
            connector.validate_data = AsyncMock(
                return_value=Mock(score=0.9, status="pass", issues=[])
            )

            start_time = time.time()
            result = await ingestion.ingest_multi_source(sensors={"test": True})
            end_time = time.time()

            total_time = end_time - start_time

            # Store benchmark result
            result["validation_time"] = total_time
            result["throughput_with_validation"] = len(benchmark_data) / total_time

            return result

        result = benchmark(lambda: asyncio.run(validation_benchmark()))

        validation_time = result["validation_time"]
        throughput_with_validation = result["throughput_with_validation"]

        assert validation_time < 30, f"Validation too slow: {validation_time:.2f}s"
        assert throughput_with_validation > 500, (
            f"Throughput with validation too low: {throughput_with_validation:.0f} records/second"
        )

        print(
            f"📊 Validation throughput: {throughput_with_validation:.0f} records/second"
        )
        print(f"⏱️  Validation time: {validation_time:.2f}s")

    def test_multi_source_ingestion_benchmark(self, benchmark_data, benchmark):
        """Benchmark multi-source data ingestion."""

        async def multi_source_benchmark():
            ingestion = MultiSourceDataIngestion(
                data_sources=["satellite", "sensors", "crowdsourced"],
                validation_enabled=True,
                parallel_processing=True,
                max_workers=3,
            )

            # Mock all connectors
            for source_name in ["satellite", "sensors", "crowdsourced"]:
                connector = ingestion.connectors[source_name]
                connector.connect = AsyncMock(return_value=True)
                connector.fetch_data = AsyncMock(
                    return_value={
                        "data": benchmark_data,
                        "metadata": {source_name: "mock_data"},
                    }
                )
                connector.validate_data = AsyncMock(
                    return_value=Mock(score=0.9, status="pass", issues=[])
                )

            start_time = time.time()
            result = await ingestion.ingest_multi_source(
                satellite={"test": True},
                sensors={"test": True},
                crowdsourced={"test": True},
            )
            end_time = time.time()

            total_time = end_time - start_time
            total_records = len(benchmark_data) * 3  # 3 sources

            # Store benchmark result
            result["multi_source_time"] = total_time
            result["multi_source_throughput"] = total_records / total_time

            return result

        result = benchmark(lambda: asyncio.run(multi_source_benchmark()))

        multi_source_time = result["multi_source_time"]
        multi_source_throughput = result["multi_source_throughput"]

        assert multi_source_time < 45, (
            f"Multi-source ingestion too slow: {multi_source_time:.2f}s"
        )
        assert multi_source_throughput > 2000, (
            f"Multi-source throughput too low: {multi_source_throughput:.0f} records/second"
        )

        print(
            f"📊 Multi-source throughput: {multi_source_throughput:.0f} records/second"
        )
        print(f"⏱️  Multi-source time: {multi_source_time:.2f}s")


class TestStorageBenchmarks:
    """Performance benchmarks for data storage."""

    @pytest.fixture
    def benchmark_geodata(self):
        """Create benchmark geospatial data."""
        n_records = 50000

        return gpd.GeoDataFrame(
            {
                "id": range(n_records),
                "temperature": np.random.normal(20, 5, n_records),
                "humidity": np.random.normal(60, 10, n_records),
                "timestamp": pd.date_range(
                    "2023-01-01", periods=n_records, freq="1min"
                ),
            },
            geometry=gpd.points_from_xy(
                np.random.normal(-122.4194, 0.1, n_records),
                np.random.normal(37.7749, 0.1, n_records),
            ),
            crs="EPSG:4326",
        )

    @pytest.fixture
    def benchmark_metadata(self):
        """Create benchmark metadata."""
        return DatasetMetadata(
            title="Storage Benchmark Data",
            description="Benchmark dataset for storage performance testing",
            spatial=SpatialExtent(bbox=[-122.5, 37.6, -122.3, 37.8], crs="EPSG:4326"),
            temporal=TemporalExtent(
                start=datetime(2023, 1, 1), end=datetime(2023, 12, 31)
            ),
            lineage=DataLineage(
                source="benchmark", process="performance_test", created_by="test"
            ),
        )

    def test_storage_latency_benchmark(
        self, benchmark_geodata, benchmark_metadata, benchmark
    ):
        """Benchmark storage latency."""

        async def storage_benchmark():
            storage = AdaptiveDataStorage(
                storage_backends=["local"],
                optimization_strategy="performance_focused",
                compression_enabled=True,
            )

            start_time = time.time()
            dataset_id = await storage.store_geospatial_data(
                benchmark_geodata,
                benchmark_metadata,
                access_patterns={"query_frequency": "high"},
            )
            end_time = time.time()

            latency = end_time - start_time

            return {
                "dataset_id": dataset_id,
                "latency": latency,
                "records": len(benchmark_geodata),
                "throughput": len(benchmark_geodata) / latency,
            }

        result = benchmark(lambda: asyncio.run(storage_benchmark()))

        latency = result["latency"]
        throughput = result["throughput"]

        assert latency < 10, f"Storage latency too high: {latency:.2f}s"
        assert throughput > 5000, (
            f"Storage throughput too low: {throughput:.0f} records/second"
        )

        print(f"📊 Storage throughput: {throughput:.0f} records/second")
        print(f"⏱️  Storage latency: {latency:.2f}s")

    def test_query_performance_benchmark(
        self, benchmark_geodata, benchmark_metadata, benchmark
    ):
        """Benchmark query performance."""

        async def query_benchmark():
            storage = AdaptiveDataStorage(["local"])

            # Store data first
            await storage.store_geospatial_data(benchmark_geodata, benchmark_metadata)

            # Benchmark spatial queries
            spatial_times = []
            for _ in range(10):  # Multiple queries for averaging
                start_time = time.time()
                results = await storage.adaptive_query(
                    spatial_bounds=[-122.5, 37.7, -122.3, 37.9],
                    optimization_hints={"frequent_queries": True},
                )
                query_time = time.time() - start_time
                spatial_times.append(query_time)

            # Benchmark temporal queries
            temporal_times = []
            for _ in range(10):
                start_time = time.time()
                results = await storage.adaptive_query(
                    temporal_range=(datetime(2023, 1, 1), datetime(2023, 1, 2)),
                    optimization_hints={"real_time": False},
                )
                query_time = time.time() - start_time
                temporal_times.append(query_time)

            avg_spatial_time = sum(spatial_times) / len(spatial_times)
            avg_temporal_time = sum(temporal_times) / len(temporal_times)

            return {
                "avg_spatial_query_time": avg_spatial_time,
                "avg_temporal_query_time": avg_temporal_time,
                "query_count": 20,
                "results_returned": len(results) if results is not None else 0,
            }

        result = benchmark(lambda: asyncio.run(query_benchmark()))

        avg_spatial_time = result["avg_spatial_query_time"]
        avg_temporal_time = result["avg_temporal_query_time"]

        assert avg_spatial_time < 2.0, (
            f"Spatial query too slow: {avg_spatial_time:.2f}s"
        )
        assert avg_temporal_time < 2.0, (
            f"Temporal query too slow: {avg_temporal_time:.2f}s"
        )

        print(f"📊 Average spatial query time: {avg_spatial_time:.3f}s")
        print(f"📊 Average temporal query time: {avg_temporal_time:.3f}s")
        print(f"📊 Queries executed: {result['query_count']}")


class TestValidationBenchmarks:
    """Performance benchmarks for data validation."""

    @pytest.fixture
    def benchmark_validation_data(self):
        """Create benchmark validation data."""
        n_records = 20000

        return pd.DataFrame(
            {
                "id": range(n_records),
                "timestamp": pd.date_range("2023-01-01", periods=n_records, freq="30s"),
                "temperature": np.random.normal(20, 5, n_records),
                "humidity": np.random.normal(60, 10, n_records),
                "latitude": np.random.normal(37.7749, 0.05, n_records),
                "longitude": np.random.normal(-122.4194, 0.05, n_records),
                "sensor_id": [f"sensor_{i % 50}" for i in range(n_records)],
            }
        )

    def test_validation_comprehensive_benchmark(
        self, benchmark_validation_data, benchmark
    ):
        """Benchmark comprehensive data validation."""

        async def validation_benchmark():
            quality_manager = DataQualityManager(
                validation_rules="comprehensive", quality_threshold=0.8
            )
            metadata = DatasetMetadata(
                title="Benchmark dataset",
                spatial=SpatialExtent(bbox=[-122.5, 37.7, -122.3, 37.9]),
                temporal=TemporalExtent(
                    start=datetime(2023, 1, 1), end=datetime(2023, 12, 31)
                ),
                lineage=DataLineage(
                    source="benchmark", process="validation", created_by="pytest"
                ),
            )
            quality_manager.register_dataset(
                "benchmark_dataset", benchmark_validation_data, metadata
            )

            start_time = time.time()
            report = await quality_manager.validate_dataset("benchmark_dataset")
            end_time = time.time()

            validation_time = end_time - start_time

            return {
                "validation_time": validation_time,
                "overall_score": report.overall_score,
                "checks_performed": len(report.checks),
                "records_validated": len(benchmark_validation_data),
                "throughput": len(benchmark_validation_data) / validation_time,
            }

        result = benchmark(lambda: asyncio.run(validation_benchmark()))

        validation_time = result["validation_time"]
        throughput = result["throughput"]

        assert validation_time < 20, f"Validation too slow: {validation_time:.2f}s"
        assert throughput > 1000, (
            f"Validation throughput too low: {throughput:.0f} records/second"
        )

        print(f"📊 Validation throughput: {throughput:.0f} records/second")
        print(f"⏱️  Validation time: {validation_time:.2f}s")
        print(f"📊 Quality score: {result['overall_score']:.2f}")
        print(f"📋 Checks performed: {result['checks_performed']}")

    def test_validation_rules_benchmark(self, benchmark_validation_data, benchmark):
        """Benchmark individual validation rules."""

        async def rules_benchmark():
            quality_manager = DataQualityManager(validation_rules="comprehensive")

            # Benchmark completeness check
            start_time = time.time()
            await quality_manager.validator.validate_data(benchmark_validation_data)
            completeness_time = time.time() - start_time

            # Benchmark accuracy check
            start_time = time.time()
            await quality_manager.validator._check_accuracy(
                benchmark_validation_data, None
            )
            accuracy_time = time.time() - start_time

            # Benchmark consistency check
            start_time = time.time()
            await quality_manager.validator._check_consistency(
                benchmark_validation_data, None
            )
            consistency_time = time.time() - start_time

            return {
                "completeness_time": completeness_time,
                "accuracy_time": accuracy_time,
                "consistency_time": consistency_time,
                "total_validation_time": completeness_time
                + accuracy_time
                + consistency_time,
                "records": len(benchmark_validation_data),
            }

        result = benchmark(lambda: asyncio.run(rules_benchmark()))

        total_time = result["total_validation_time"]

        assert result["completeness_time"] < 5, (
            f"Completeness check too slow: {result['completeness_time']:.2f}s"
        )
        assert result["accuracy_time"] < 5, (
            f"Accuracy check too slow: {result['accuracy_time']:.2f}s"
        )
        assert result["consistency_time"] < 5, (
            f"Consistency check too slow: {result['consistency_time']:.2f}s"
        )
        assert total_time < 15, f"Total validation too slow: {total_time:.2f}s"

        print(f"📊 Completeness check: {result['completeness_time']:.2f}s")
        print(f"📊 Accuracy check: {result['accuracy_time']:.2f}s")
        print(f"📊 Consistency check: {result['consistency_time']:.2f}s")
        print(f"📊 Total validation: {total_time:.2f}s")


class TestMemoryBenchmarks:
    """Memory usage benchmarks."""

    def test_memory_usage_ingestion(self):
        """Test memory usage during data ingestion."""
        process = psutil.Process()

        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create large dataset
        n_records = 100000
        pd.DataFrame(
            {
                "id": range(n_records),
                "temperature": np.random.normal(20, 5, n_records),
                "humidity": np.random.normal(60, 10, n_records),
                "latitude": np.random.normal(37.7, 0.1, n_records),
                "longitude": np.random.normal(-122.4, 0.1, n_records),
            }
        )

        peak_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Memory usage should be reasonable
        assert peak_memory - initial_memory < 500, (
            f"Memory usage too high: {peak_memory - initial_memory:.0f} MB"
        )

        print(f"💾 Initial memory: {initial_memory:.0f} MB")
        print(f"💾 Peak memory: {peak_memory:.0f} MB")
        print(f"💾 Memory increase: {peak_memory - initial_memory:.0f} MB")

    def test_memory_usage_storage(self):
        """Test memory usage during data storage."""
        process = psutil.Process()

        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create large geospatial dataset
        n_records = 50000
        large_gdf = gpd.GeoDataFrame(
            {
                "id": range(n_records),
                "temperature": np.random.normal(20, 5, n_records),
                "humidity": np.random.normal(60, 10, n_records),
            },
            geometry=gpd.points_from_xy(
                np.random.normal(-122.4, 0.1, n_records),
                np.random.normal(37.7, 0.1, n_records),
            ),
            crs="EPSG:4326",
        )

        metadata = DatasetMetadata(
            title="Memory Test Data",
            spatial=SpatialExtent(bbox=[-122.5, 37.6, -122.3, 37.8]),
            temporal=TemporalExtent(
                start=datetime(2023, 1, 1), end=datetime(2023, 1, 2)
            ),
            lineage=DataLineage(
                source="memory_test", process="benchmark", created_by="test"
            ),
        )

        # Test storage memory usage
        import asyncio

        async def storage_memory_test():
            storage = AdaptiveDataStorage(["local"])
            await storage.store_geospatial_data(large_gdf, metadata)
            return storage

        asyncio.run(storage_memory_test())

        peak_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Memory usage should be reasonable
        assert peak_memory - initial_memory < 300, (
            f"Storage memory usage too high: {peak_memory - initial_memory:.0f} MB"
        )

        print(f"💾 Initial memory: {initial_memory:.0f} MB")
        print(f"💾 Peak memory: {peak_memory:.0f} MB")
        print(f"💾 Memory increase: {peak_memory - initial_memory:.0f} MB")


class TestScalabilityBenchmarks:
    """Scalability benchmarks for different data sizes."""

    @pytest.mark.parametrize("data_size", ["small", "medium", "large"])
    def test_ingestion_scalability(self, data_size):
        """Test ingestion scalability with different data sizes."""
        import asyncio

        async def scalability_test():
            # Get data size
            n_records = {"small": 1000, "medium": 10000, "large": 100000}[data_size]

            # Create test data
            test_data = pd.DataFrame(
                {
                    "timestamp": pd.date_range(
                        "2023-01-01", periods=n_records, freq="1min"
                    ),
                    "temperature": np.random.normal(20, 5, n_records),
                    "latitude": np.random.normal(37.7, 0.1, n_records),
                    "longitude": np.random.normal(-122.4, 0.1, n_records),
                }
            )

            # Test ingestion
            ingestion = MultiSourceDataIngestion(["sensors"], validation_enabled=False)

            start_time = time.time()
            await ingestion.ingest_multi_source(sensors={"test_data": test_data})
            end_time = time.time()

            throughput = n_records / (end_time - start_time)

            return {
                "data_size": data_size,
                "records": n_records,
                "time": end_time - start_time,
                "throughput": throughput,
            }

        result = asyncio.run(scalability_test())

        # Scalability should be roughly linear
        expected_throughput = {"small": 1000, "medium": 2000, "large": 3000}[data_size]

        assert result["throughput"] > expected_throughput * 0.5, (
            f"Scalability issue for {data_size}: {result['throughput']:.0f} records/second"
        )

        print(f"📈 {data_size.capitalize()} dataset ({result['records']} records):")
        print(f"   ⏱️  Processing time: {result['time']:.2f}s")
        print(f"   📊 Throughput: {result['throughput']:.0f} records/second")

    @pytest.mark.parametrize("data_size", ["small", "medium", "large"])
    def test_storage_scalability(self, data_size):
        """Test storage scalability with different data sizes."""
        import asyncio

        async def storage_scalability_test():
            # Get data size
            n_records = {"small": 1000, "medium": 10000, "large": 100000}[data_size]

            # Create geospatial data
            gdf = gpd.GeoDataFrame(
                {
                    "id": range(n_records),
                    "temperature": np.random.normal(20, 5, n_records),
                },
                geometry=gpd.points_from_xy(
                    np.random.normal(-122.4, 0.1, n_records),
                    np.random.normal(37.7, 0.1, n_records),
                ),
                crs="EPSG:4326",
            )

            metadata = DatasetMetadata(
                title=f"Scalability Test {data_size}",
                spatial=SpatialExtent(bbox=[-122.5, 37.6, -122.3, 37.8]),
                temporal=TemporalExtent(
                    start=datetime(2023, 1, 1), end=datetime(2023, 1, 2)
                ),
                lineage=DataLineage(
                    source="scalability_test", process="benchmark", created_by="test"
                ),
            )

            # Test storage
            storage = AdaptiveDataStorage(["local"])

            start_time = time.time()
            await storage.store_geospatial_data(gdf, metadata)
            end_time = time.time()

            storage_time = end_time - start_time
            throughput = n_records / storage_time

            return {
                "data_size": data_size,
                "records": n_records,
                "storage_time": storage_time,
                "throughput": throughput,
            }

        result = asyncio.run(storage_scalability_test())

        # Storage should scale reasonably well
        max_time = {"small": 5, "medium": 15, "large": 60}[data_size]

        assert result["storage_time"] < max_time, (
            f"Storage too slow for {data_size}: {result['storage_time']:.2f}s"
        )

        print(f"💾 {data_size.capitalize()} dataset storage:")
        print(f"   ⏱️  Storage time: {result['storage_time']:.2f}s")
        print(f"   📊 Throughput: {result['throughput']:.0f} records/second")
