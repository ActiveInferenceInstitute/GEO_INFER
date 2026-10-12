"""
REST API implementation for GEO-INFER-DATA.

This module provides a comprehensive REST API for data access, management,
and processing operations.
"""

import logging
from typing import Any, cast
from datetime import datetime, UTC
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Path as PathParam
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from ..models.schemas import (
    Dataset,
    DatasetMetadata,
    DatasetSummary,
    HealthStatus,
)
from ..core.ingestion import MultiSourceDataIngestion
from geo_infer_data import __version__
from ..core.storage import AdaptiveDataStorage
from ..core.validation import DataQualityManager
from ..core.pipeline import IntelligentETLPipeline
from .service import DataService
from .errors import register_error_handlers


logger = logging.getLogger(__name__)


class DataAPI:
    """
    REST API server for geospatial data management.

    This class provides a comprehensive REST API for accessing, managing,
    and processing geospatial data with automatic optimization and validation.

    Args:
        config_path: Path to configuration file
        host: Server host
        port: Server port
        enable_cors: Whether to enable CORS

    Examples:
        >>> api = DataAPI(config_path='config/local.yaml', port=8001)
        >>> api.start()
    """

    def __init__(
        self,
        config_path: Path | None = None,
        host: str = "0.0.0.0",
        port: int = 8001,
        enable_cors: bool = True,
    ):
        self.config_path = config_path
        self.host = host
        self.port = port
        self.enable_cors = enable_cors

        # Initialize core services
        self.ingestion_service = MultiSourceDataIngestion(
            data_sources=["satellite", "sensors", "crowdsourced"],
            validation_enabled=True,
        )

        self.storage_service = AdaptiveDataStorage(
            storage_backends=["local"], optimization_strategy="access_pattern_based"
        )

        self.quality_service = DataQualityManager(
            validation_rules="comprehensive", real_time_monitoring=True
        )

        self.pipeline_service = IntelligentETLPipeline(
            workflow_config=None, error_recovery="intelligent_retry"
        )
        self.data_service = DataService(self.storage_service, self.quality_service)

        # Initialize FastAPI app
        self.app = FastAPI(
            title="GEO-INFER-DATA API",
            description="Comprehensive Data Management and Storage for Geospatial Systems",
            version=__version__,
        )

        if self.enable_cors:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

        # Non-domain exceptions escaping handlers become generic 500s
        # (LOG-EXC-01 pattern); domain errors map to 4xx in the handlers.
        register_error_handlers(self.app)

        self._setup_routes()

        logger.info(f"Initialized DataAPI on {host}:{port}")

    def _setup_routes(self) -> None:
        """Setup API routes."""

        @self.app.get("/")
        async def root() -> dict[str, Any]:
            """Root endpoint."""
            return {
                "name": "GEO-INFER-DATA API",
                "version": __version__,
                "status": "running",
                "timestamp": datetime.now(UTC).isoformat(),
            }

        @self.app.get("/health")
        async def health_check() -> HealthStatus:
            """Health check endpoint."""
            return HealthStatus(
                status="healthy",
                message="Data API is running",
                checked_at=datetime.now(UTC),
            )

        @self.app.get("/datasets", response_model=list[DatasetSummary])
        async def list_datasets(
            page: int = Query(1, ge=1),
            limit: int = Query(50, ge=1, le=1000),
            data_type: str | None = None,
            bbox: str | None = Query(None),
        ) -> list[DatasetSummary]:
            """List available datasets."""
            filters: dict[str, Any] = {}
            if data_type:
                filters["type"] = data_type
            if bbox:
                filters["bbox"] = [float(value) for value in bbox.split(",")]
            records = await self.data_service.list_datasets(
                filters=filters, limit=limit, offset=(page - 1) * limit
            )
            return [
                DatasetSummary(
                    id=record.id,
                    title=record.title,
                    description=record.description,
                    type=record.type,
                    format=record.format,
                    bbox=(
                        record.metadata.spatial.bbox if record.metadata.spatial else []
                    ),
                    temporal_extent=record.metadata.temporal,
                    created_at=record.created_at,
                    updated_at=record.updated_at,
                    tags=record.tags,
                )
                for record in records
            ]

        @self.app.post("/datasets", response_model=Dataset, status_code=201)
        async def create_dataset(dataset: Dataset) -> Dataset:
            """Register dataset metadata that has already been stored."""
            logger.info(f"Registering dataset: {dataset.title}")
            self.data_service.datasets[dataset.id] = dataset
            return dataset

        @self.app.get("/datasets/{dataset_id}", response_model=Dataset)
        async def get_dataset(dataset_id: str = PathParam(...)) -> Dataset:
            """Get dataset details."""
            dataset = self.data_service.datasets.get(dataset_id)
            if dataset is None:
                raise HTTPException(status_code=404, detail="Dataset not found")
            return dataset

        @self.app.get("/datasets/{dataset_id}/data")
        async def get_dataset_data(
            dataset_id: str = PathParam(...),
            format: str = Query("geojson", enum=["geojson", "csv"]),
            bbox: list[float] | None = Query(None),
        ) -> dict[str, Any]:
            """Get dataset data."""
            if dataset_id not in self.data_service.datasets:
                raise HTTPException(status_code=404, detail="Dataset not found")
            try:
                data = await self.data_service.get_dataset_data(
                    dataset_id, spatial_bounds=bbox, format=format
                )
            except (KeyError, ValueError) as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            return {"dataset_id": dataset_id, "format": format, "data": data}

        @self.app.post("/datasets/{dataset_id}/data")
        async def upload_dataset_data(
            dataset_id: str = PathParam(...),
            file: UploadFile = File(...),
            overwrite: bool = False,
        ) -> None:
            """Upload data to dataset."""
            raise HTTPException(
                status_code=501,
                detail=(
                    "Upload requires a storage-backed ingestion workflow; "
                    "use POST /data/ingest/multi-source"
                ),
            )

        @self.app.get("/datasets/{dataset_id}/metadata", response_model=DatasetMetadata)
        async def get_dataset_metadata(
            dataset_id: str = PathParam(...),
        ) -> DatasetMetadata:
            """Get dataset metadata."""
            dataset = self.data_service.datasets.get(dataset_id)
            if dataset is None:
                raise HTTPException(status_code=404, detail="Dataset not found")
            return dataset.metadata

        @self.app.post("/data/ingest/multi-source")
        async def ingest_multi_source(request: dict[str, Any]) -> Any:
            """Ingest data from multiple sources."""
            try:
                result = await self.ingestion_service.ingest_multi_source(**request)
                return result
            except (ValueError, ConnectionError) as e:
                raise HTTPException(status_code=400, detail=str(e)) from e

        @self.app.post("/data/etl/execute")
        async def execute_etl(request: dict[str, Any]) -> Any:
            """Execute ETL pipeline."""
            try:
                result = await self.pipeline_service.execute_workflow(
                    source_data=request.get("source_data"),
                    target_storage=request.get("target_storage"),
                    transformation_rules=request.get("transformations"),
                )
                return result
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e

        @self.app.post("/quality/validate/{dataset_id}")
        async def validate_dataset_quality(
            dataset_id: str = PathParam(...),
            checks: list[str] = Query(["completeness", "accuracy", "consistency"]),
        ) -> Any:
            """Validate dataset quality."""
            try:
                report = await self.quality_service.validate_dataset(dataset_id)
                return report
            except KeyError as e:
                raise HTTPException(
                    status_code=404,
                    detail="Dataset not registered for validation",
                ) from e
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e

        @self.app.get("/search")
        async def search_datasets(
            q: str | None = None,
            bbox: str | None = Query(None),
            temporal: str | None = None,
            data_type: str | None = None,
            tags: list[str] = Query([]),
        ) -> dict[str, Any]:
            """Search datasets."""
            records = await self.data_service.list_datasets(limit=1000)
            if q:
                query = q.casefold()
                records = [
                    record
                    for record in records
                    if query in record.title.casefold()
                    or query in (record.description or "").casefold()
                    or any(query in tag.casefold() for tag in record.tags)
                ]
            if data_type:
                records = [record for record in records if record.type == data_type]
            if tags:
                records = [
                    record for record in records if set(tags).issubset(record.tags)
                ]
            return {
                "query": q,
                "results": [record.model_dump(mode="json") for record in records],
                "total": len(records),
            }

        @self.app.get("/storage/backends")
        async def list_storage_backends() -> list[Any]:
            """List storage backends."""
            stats = self.storage_service.get_storage_stats()
            return cast(list[Any], stats.get("backends", []))

        @self.app.get("/metrics")
        async def get_metrics() -> dict[str, Any]:
            """Get API metrics."""
            stats = self.storage_service.get_storage_stats()
            return {"storage": stats, "datasets": len(self.data_service.datasets)}

    def start(self, reload: bool = False) -> None:
        """Start the API server."""
        logger.info(f"Starting DataAPI server on {self.host}:{self.port}")

        uvicorn.run(
            self.app, host=self.host, port=self.port, reload=reload, log_level="info"
        )

    def stop(self) -> None:
        """Stop the API server."""
        logger.info("Stopping DataAPI server")
        # Implementation for graceful shutdown
