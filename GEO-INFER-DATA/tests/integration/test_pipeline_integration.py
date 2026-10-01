"""
Integration tests for the data pipeline workflow.

Tests the full flow: load -> transform -> validate -> output.
"""

import asyncio

import numpy as np
import pandas as pd
import pytest

from geo_infer_data.core.pipeline import (
    IntelligentETLPipeline,
    TransformationEngine,
)
from geo_infer_data.models.schemas import Transformation
from geo_infer_data.utils.compression import DataCompressor
from geo_infer_data.utils.format_detection import FormatDetector


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Multi-step pipeline integration
# ---------------------------------------------------------------------------


class TestPipelineIntegration:
    def test_filter_then_aggregate(self):
        """Pipeline: filter rows by range -> aggregate by group."""
        engine = TransformationEngine()
        df = pd.DataFrame(
            {
                "region": ["A", "A", "B", "B", "B", "A"],
                "temperature": [15, 25, 35, 45, 10, 20],
                "humidity": [50, 60, 70, 80, 40, 55],
            }
        )

        # Step 1: filter temperature >= 15 and <= 35
        filter_tx = Transformation(
            type="filter",
            parameters={"conditions": {"temperature": {"min": 15, "max": 35}}},
        )
        filtered = _run(engine.execute_transformation(filter_tx, df, {}))
        assert len(filtered) == 4  # 15, 25, 35, 20

        # Step 2: aggregate by region
        agg_tx = Transformation(
            type="aggregate",
            parameters={
                "group_by": ["region"],
                "aggregations": {"temperature": "mean"},
            },
        )
        aggregated = _run(engine.execute_transformation(agg_tx, filtered, {}))
        assert len(aggregated) == 2
        row_a = aggregated[aggregated["region"] == "A"]
        assert row_a["temperature"].iloc[0] == 20.0  # mean of 15, 25, 20

    def test_transform_then_filter(self):
        """Pipeline: scale values -> filter by scaled threshold."""
        engine = TransformationEngine()
        df = pd.DataFrame({"score": [1.0, 2.0, 3.0, 4.0, 5.0]})

        # Step 1: scale by 10
        scale_tx = Transformation(
            type="transform",
            parameters={"transformations": {"score": {"type": "scale", "factor": 10}}},
        )
        scaled = _run(engine.execute_transformation(scale_tx, df, {}))

        # Step 2: filter score >= 30
        filter_tx = Transformation(
            type="filter",
            parameters={"conditions": {"score": {"min": 30}}},
        )
        result = _run(engine.execute_transformation(filter_tx, scaled, {}))
        assert len(result) == 3
        assert result["score"].min() >= 30.0

    def test_normalize_transform(self):
        """Pipeline: normalize -> verify range 0-1."""
        engine = TransformationEngine()
        df = pd.DataFrame({"value": [10.0, 20.0, 30.0, 40.0, 50.0]})

        tx = Transformation(
            type="transform",
            parameters={"transformations": {"value": {"type": "normalize"}}},
        )
        result = _run(engine.execute_transformation(tx, df, {}))
        assert result["value"].min() == pytest.approx(0.0)
        assert result["value"].max() == pytest.approx(1.0)

    def test_full_etl_workflow_passthrough(self):
        """IntelligentETLPipeline: end-to-end with no transformations."""
        pipeline = IntelligentETLPipeline(monitoring_enabled=False)
        source_data = pd.DataFrame(
            {
                "lat": np.random.uniform(37, 38, 50),
                "lon": np.random.uniform(-123, -122, 50),
                "value": np.random.rand(50),
            }
        )
        result = _run(
            pipeline.execute_workflow(source_data=source_data, target_storage=None)
        )
        assert result["status"] == "completed"
        assert result["extracted_records"] == 50

    def test_etl_records_execution_history(self):
        """IntelligentETLPipeline: execution history grows."""
        pipeline = IntelligentETLPipeline(monitoring_enabled=False)
        for _ in range(3):
            _run(
                pipeline.execute_workflow(
                    source_data=pd.DataFrame({"x": [1]}), target_storage=None
                )
            )
        assert len(pipeline.execution_history) == 3


# ---------------------------------------------------------------------------
# Compression + format detection integration
# ---------------------------------------------------------------------------


class TestCompressionFormatIntegration:
    def test_detect_compress_roundtrip(self):
        """Detect format of data, compress, decompress, verify integrity."""
        detector = FormatDetector()
        compressor = DataCompressor(algorithm="gzip")

        data = {"type": "FeatureCollection", "features": []}
        detector.detect_format(data)
        compressed = compressor.compress_data(data)
        assert isinstance(compressed, bytes)

        decompressed = compressor.decompress_data(compressed, verified=True)
        assert decompressed == data

    def test_compress_dataframe_and_stats(self):
        """Compress a DataFrame and verify stats are populated."""
        compressor = DataCompressor(algorithm="gzip")
        df = pd.DataFrame({"a": range(500), "b": np.random.rand(500)})

        compressed = compressor.compress_data(df)
        assert isinstance(compressed, bytes)

        stats = compressor.get_compression_stats()
        assert stats["compression_count"] == 1
        assert stats["total_original_bytes"] > 0
        assert stats["compression_ratio"] >= 1.0


# ---------------------------------------------------------------------------
# Bottleneck identification integration
# ---------------------------------------------------------------------------


class TestBottleneckIntegration:
    def test_pipeline_bottleneck_detection_short_execution(self):
        pipeline = IntelligentETLPipeline()
        bottlenecks = pipeline.identify_bottlenecks({"execution_time_seconds": 5})
        assert bottlenecks == []

    def test_pipeline_bottleneck_detection_long_execution(self):
        pipeline = IntelligentETLPipeline()
        bottlenecks = pipeline.identify_bottlenecks({"execution_time_seconds": 7200})
        assert len(bottlenecks) > 0
