"""
Tests for Pydantic data models in geo_infer_data.models.schemas.

Validates schema construction, field validation, serialization,
and constraint enforcement for all core data models.
"""

from datetime import datetime, UTC
import pytest
from pydantic import ValidationError

from geo_infer_data.models.schemas import (
    CoordinateReferenceSystem,
    DataDestination,
    DataFormat,
    DataLineage,
    DataQualityReport,
    DataSource,
    DataType,
    Dataset,
    DatasetMetadata,
    ETLPipeline,
    ExecutionState,
    ExecutionStatus,
    HealthStatus,
    Pagination,
    QualityCheck,
    QualityStatus,
    SpatialExtent,
    StorageBackend,
    TemporalExtent,
    Transformation,
)


# ---------------------------------------------------------------------------
# SpatialExtent
# ---------------------------------------------------------------------------


class TestSpatialExtent:
    def test_valid_4_element_bbox(self):
        se = SpatialExtent(bbox=[-122.5, 37.7, -122.3, 37.9])
        assert len(se.bbox) == 4

    def test_valid_6_element_bbox(self):
        se = SpatialExtent(bbox=[-122.5, 37.7, 0.0, -122.3, 37.9, 100.0])
        assert len(se.bbox) == 6

    def test_invalid_bbox_lon_raises(self):
        with pytest.raises(ValidationError):
            SpatialExtent(bbox=[-200.0, 37.7, -122.3, 37.9])

    def test_invalid_bbox_order_raises(self):
        with pytest.raises(ValidationError):
            SpatialExtent(bbox=[-120.0, 37.9, -122.3, 37.7])  # min_lon > max_lon

    def test_crs_string_converted(self):
        se = SpatialExtent(bbox=[-10, -10, 10, 10], crs="EPSG:4326")
        assert isinstance(se.crs, CoordinateReferenceSystem)


# ---------------------------------------------------------------------------
# TemporalExtent
# ---------------------------------------------------------------------------


class TestTemporalExtent:
    def test_valid_temporal_extent(self):
        te = TemporalExtent(
            start=datetime(2023, 1, 1, tzinfo=UTC),
            end=datetime(2023, 12, 31, tzinfo=UTC),
        )
        assert te.start < te.end

    def test_invalid_order_raises(self):
        with pytest.raises(ValidationError):
            TemporalExtent(
                start=datetime(2023, 12, 31, tzinfo=UTC),
                end=datetime(2023, 1, 1, tzinfo=UTC),
            )


# ---------------------------------------------------------------------------
# QualityCheck
# ---------------------------------------------------------------------------


class TestQualityCheck:
    def test_valid_quality_check(self):
        qc = QualityCheck(score=0.85, status=QualityStatus.PASS)
        assert qc.score == 0.85

    def test_score_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            QualityCheck(score=1.5, status=QualityStatus.PASS)

    def test_issues_default_empty(self):
        qc = QualityCheck(score=0.5, status=QualityStatus.WARNING)
        assert qc.issues == []


# ---------------------------------------------------------------------------
# DatasetMetadata
# ---------------------------------------------------------------------------


class TestDatasetMetadata:
    def _make_metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            title="Test Dataset",
            spatial=SpatialExtent(bbox=[-10, -10, 10, 10]),
            lineage=DataLineage(
                source="test",
                process="ingest",
                created_by="unit_test",
            ),
        )

    def test_creation(self):
        md = self._make_metadata()
        assert md.title == "Test Dataset"
        assert md.version == "1.0.0"

    def test_optional_fields_default(self):
        md = self._make_metadata()
        assert md.description is None
        assert md.keywords == []
        assert md.quality == {}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class TestDataset:
    def test_dataset_creation(self):
        md = DatasetMetadata(
            title="DS",
            spatial=SpatialExtent(bbox=[-1, -1, 1, 1]),
            lineage=DataLineage(source="s", process="p", created_by="c"),
        )
        ds = Dataset(
            title="DS",
            type=DataType.VECTOR,
            format=DataFormat.GEOJSON,
            metadata=md,
        )
        assert ds.is_active is True
        assert ds.id  # UUID auto-generated

    def test_dataset_has_uuid(self):
        md = DatasetMetadata(
            title="DS2",
            spatial=SpatialExtent(bbox=[-1, -1, 1, 1]),
            lineage=DataLineage(source="s", process="p", created_by="c"),
        )
        ds = Dataset(
            title="DS2",
            type=DataType.RASTER,
            format=DataFormat.GEOTIFF,
            metadata=md,
        )
        assert len(ds.id) > 0


# ---------------------------------------------------------------------------
# DataQualityReport
# ---------------------------------------------------------------------------


class TestDataQualityReport:
    def test_report_creation(self):
        report = DataQualityReport(
            dataset_id="ds-123",
            overall_score=0.88,
            checks={
                "completeness": QualityCheck(score=0.95, status=QualityStatus.PASS),
                "accuracy": QualityCheck(score=0.8, status=QualityStatus.WARNING),
            },
        )
        assert report.overall_score == 0.88
        assert len(report.checks) == 2


# ---------------------------------------------------------------------------
# ETLPipeline
# ---------------------------------------------------------------------------


class TestETLPipeline:
    def test_pipeline_creation(self):
        pipeline = ETLPipeline(
            name="test_pipeline",
            source=DataSource(type="file", configuration={"path": "/data"}),
            destination=DataDestination(
                type="database", configuration={"table": "out"}
            ),
        )
        assert pipeline.name == "test_pipeline"
        assert pipeline.status == "inactive"

    def test_transformations_sorted(self):
        t1 = Transformation(type="filter", order=2)
        t2 = Transformation(type="transform", order=1)
        pipeline = ETLPipeline(
            name="p",
            source=DataSource(type="file", configuration={}),
            destination=DataDestination(type="file", configuration={}),
            transformations=[t1, t2],
        )
        assert pipeline.transformations[0].order <= pipeline.transformations[1].order


# ---------------------------------------------------------------------------
# ExecutionStatus
# ---------------------------------------------------------------------------


class TestExecutionStatus:
    def test_execution_status_defaults(self):
        es = ExecutionStatus(
            pipeline_id="p1",
            status=ExecutionState.PENDING,
        )
        assert es.progress == 0.0
        assert es.logs == []


# ---------------------------------------------------------------------------
# Pagination & HealthStatus
# ---------------------------------------------------------------------------


class TestPaginationAndHealth:
    def test_pagination(self):
        p = Pagination(page=1, limit=20, total=100)
        assert p.has_next is False

    def test_health_status(self):
        h = HealthStatus(status="healthy")
        assert h.status == "healthy"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TestEnums:
    def test_data_type_values(self):
        assert DataType.VECTOR == "vector"
        assert DataType.RASTER == "raster"

    def test_data_format_values(self):
        assert DataFormat.GEOJSON == "geojson"
        assert DataFormat.PARQUET == "parquet"

    def test_storage_backend_values(self):
        assert StorageBackend.POSTGRESQL == "postgresql"
        assert StorageBackend.LOCAL == "local"
