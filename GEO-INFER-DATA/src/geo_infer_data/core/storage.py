"""
Adaptive data storage system for GEO-INFER-DATA.

This module provides comprehensive data storage capabilities with adaptive
optimization based on access patterns, performance requirements, and
cost considerations.

Security note (serialization trust boundary): the MinIO, Redis, local-file,
and generic-table backends in this module serialise payloads with
:mod:`pickle`, which can execute arbitrary code during unpickling. Every one
of those payloads is therefore written as an authenticated GISP1 envelope (see
:mod:`geo_infer_data.utils.secure_serialization`) and every read verifies the
envelope's HMAC-SHA256 under a per-backend context key *before* the bytes
reach a deserialiser — including before decompression, so a hostile blob
cannot even be inflated. Unsigned, truncated, cross-backend, or tampered
payloads raise
:class:`geo_infer_data.utils.secure_serialization.PayloadSecurityError`
and are never deserialised.

The ``data_id`` glob hardening in ``_find_data_file`` remains as defence in
depth against path-widening; the envelope verification is what makes a
retrieved payload safe to load.
"""

import logging
import os
from typing import Any, cast
from datetime import datetime, UTC
from dataclasses import dataclass, field
from enum import StrEnum
import asyncio
import json
import pickle
import uuid
from io import BytesIO

import geopandas as gpd
import pandas as pd
from pathlib import Path

from ..models.schemas import DatasetMetadata
from ..utils.identifiers import validate_sql_identifier
from ..utils.dependencies import require_dependency
from ..utils.timestamps import (
    normalize_observation_timestamps,
    normalize_temporal_range,
)
from ..utils.indexing import SpatialIndexer, TemporalIndexer
from ..utils.compression import DataCompressor
from ..utils.caching import CacheManager
from ..utils.secure_serialization import (
    TEXT_PREFIX,
    CONTEXT_STORAGE_GENERIC,
    CONTEXT_STORAGE_LOCAL_FILE,
    CONTEXT_STORAGE_MINIO,
    CONTEXT_STORAGE_REDIS,
    dumps_signed,
    loads_signed,
    sign_payload,
    sign_payload_text,
    verify_payload,
    verify_payload_text,
)


logger = logging.getLogger(__name__)


async def _maybe_await(value: Any) -> Any:
    """Await coroutine-like backend helpers while accepting synchronous results."""
    while asyncio.isfuture(value) or asyncio.iscoroutine(value):
        value = await value
    return value


class OptimizationStrategy(StrEnum):
    """Storage optimization strategies."""

    ACCESS_PATTERN_BASED = "access_pattern_based"
    PERFORMANCE_FOCUSED = "performance_focused"
    COST_OPTIMIZED = "cost_optimized"
    BALANCED = "balanced"


class IndexingStrategy(StrEnum):
    """Spatial indexing strategies."""

    H3 = "h3"
    R_TREE = "r_tree"
    GEOSPATIAL = "geospatial"
    SPATIAL_HASH = "spatial_hash"


@dataclass
class StorageConfig:
    """Storage system configuration."""

    storage_backends: list[str]
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.BALANCED
    compression_enabled: bool = True
    indexing_strategy: IndexingStrategy = IndexingStrategy.H3
    caching_enabled: bool = True
    replication_factor: int = 1
    max_file_size: int = 1024 * 1024 * 1024  # 1GB
    retention_policy: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.optimization_strategy, OptimizationStrategy):
            self.optimization_strategy = OptimizationStrategy(  # type: ignore[unreachable]
                self.optimization_strategy
            )
        if not isinstance(self.indexing_strategy, IndexingStrategy):
            self.indexing_strategy = IndexingStrategy(  # type: ignore[unreachable]
                self.indexing_strategy
            )


@dataclass
class AccessPattern:
    """Data access pattern analysis."""

    query_frequency: dict[str, int] = field(default_factory=dict)
    spatial_bounds: list[list[float]] = field(default_factory=list)
    temporal_ranges: list[tuple[datetime, datetime]] = field(default_factory=list)
    data_types: list[str] = field(default_factory=list)
    peak_hours: list[int] = field(default_factory=list)
    batch_size_distribution: dict[str, int] = field(default_factory=dict)


class StorageBackendManager:
    """Manager for different storage backends."""

    def __init__(self, backend_configs: dict[str, dict[str, Any]]):
        self.backend_configs = backend_configs
        self.backends: dict[str, Any] = {}
        self._initialize_backends()

    def _initialize_backends(self) -> None:
        """Initialize storage backends."""
        for backend_name, config in self.backend_configs.items():
            backend_type = config.get("type", backend_name)

            if backend_type == "postgresql":
                self.backends[backend_name] = PostgreSQLBackend(config)
            elif backend_type == "minio":
                self.backends[backend_name] = MinIOBackend(config)
            elif backend_type == "redis":
                self.backends[backend_name] = RedisBackend(config)
            elif backend_type == "local":
                self.backends[backend_name] = LocalFileBackend(config)
            else:
                logger.warning(f"Unknown backend type: {backend_type}")

    async def store_data(
        self, data: Any, metadata: DatasetMetadata, backend: str = "default"
    ) -> str:
        """Store data in specified backend."""
        if backend not in self.backends:
            raise ValueError(f"Backend {backend} not available")

        backend_instance = self.backends[backend]
        return cast(str, await backend_instance.store(data, metadata))

    async def retrieve_data(
        self, data_id: str, query: dict[str, Any], backend: str = "default"
    ) -> Any:
        """Retrieve data from specified backend."""
        if backend not in self.backends:
            raise ValueError(f"Backend {backend} not available")

        backend_instance = self.backends[backend]
        return await backend_instance.retrieve(data_id, query)

    async def delete_data(self, data_id: str, backend: str = "default") -> bool:
        """Delete data from specified backend."""
        if backend not in self.backends:
            raise ValueError(f"Backend {backend} not available")

        backend_instance = self.backends[backend]
        return cast(bool, await backend_instance.delete(data_id))


class PostgreSQLBackend:
    """PostgreSQL/PostGIS storage backend."""

    def __init__(self, config: dict[str, Any]):
        require_dependency("psycopg2", "postgres")
        self.config = config
        self.connection_string = self._build_connection_string()
        self.spatial_indexer = SpatialIndexer()
        self.signing_key = config.get("signing_key")

    def _build_connection_string(self) -> str:
        """Build PostgreSQL connection string."""
        return (
            f"postgresql://{self.config['user']}:{self.config['password']}"
            f"@{self.config['host']}:{self.config['port']}/{self.config['database']}"
        )

    async def store(self, data: Any, metadata: DatasetMetadata) -> str:
        """Store data in PostgreSQL."""
        # Implementation for PostgreSQL storage
        data_id = f"pg_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"

        if isinstance(data, (gpd.GeoDataFrame, pd.DataFrame)):
            # Store tabular data
            await self._store_dataframe(data, data_id, metadata)
        else:
            # Store other data types
            await self._store_generic(data, data_id, metadata)

        return data_id

    async def _store_dataframe(
        self, df: pd.DataFrame, data_id: str, metadata: DatasetMetadata
    ) -> None:
        """Store pandas/geopandas DataFrame."""
        # Create table and store data
        table_name = validate_sql_identifier(f"dataset_{data_id.replace('-', '_')}")

        # Convert to SQL and execute
        if isinstance(df, gpd.GeoDataFrame):
            # Handle geospatial data
            df.to_postgis(table_name, self.connection_string, if_exists="replace")
        else:
            df.to_sql(table_name, self.connection_string, if_exists="replace")

        # Create spatial index if geospatial data
        if isinstance(df, gpd.GeoDataFrame) and df.crs:
            self.spatial_indexer.create_spatial_index(
                table_name, self.connection_string
            )

    async def _retrieve_dataframe(
        self, data_id: str, query: dict[str, Any]
    ) -> pd.DataFrame:
        """Retrieve a stored table from PostgreSQL/PostGIS."""
        from sqlalchemy import create_engine, inspect as sqlalchemy_inspect, text

        table_name = validate_sql_identifier(f"dataset_{data_id.replace('-', '_')}")
        engine = create_engine(self.connection_string)
        try:
            if not sqlalchemy_inspect(engine).has_table(table_name):
                raise FileNotFoundError(f"PostgreSQL dataset {data_id!r} was not found")
            if query.get("spatial"):
                min_lon, min_lat, max_lon, max_lat = query["spatial"]
                sql = text(
                    f"SELECT * FROM {table_name} "
                    "WHERE ST_Intersects(geometry, ST_MakeEnvelope(:min_lon, :min_lat, "
                    ":max_lon, :max_lat, 4326))"
                )
                return gpd.read_postgis(
                    sql,
                    engine,
                    params={
                        "min_lon": min_lon,
                        "min_lat": min_lat,
                        "max_lon": max_lon,
                        "max_lat": max_lat,
                    },
                    geom_col="geometry",
                )
            return pd.read_sql_table(table_name, engine)
        finally:
            engine.dispose()

    async def _store_generic(
        self, data: Any, data_id: str, metadata: DatasetMetadata
    ) -> None:
        """Store generic data by serialising to JSON in a metadata table.

        For non-DataFrame data (dicts, lists, scalars) this method
        serialises the value as JSON and stores it alongside its metadata
        in a dedicated key-value table.
        """
        table_name = validate_sql_identifier("generic_data_store")

        # Ensure the table exists
        create_stmt = (
            f"CREATE TABLE IF NOT EXISTS {table_name} ("
            "data_id TEXT PRIMARY KEY, "
            "payload TEXT NOT NULL, "
            "title TEXT, "
            "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )

        try:
            serialized = json.dumps(data, default=str)
        except (TypeError, ValueError) as ser_err:
            # JSON cannot represent this value. Fall back to pickle, but only
            # inside an authenticated text envelope so the reader can tell a
            # signed pickle from plain JSON and can reject a tampered row.
            serialized = sign_payload_text(
                pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL),
                context=CONTEXT_STORAGE_GENERIC,
                key=self.signing_key,
            )
            logger.warning(
                "Fell back to signed pickle serialisation for %s: %s",
                data_id,
                ser_err,
            )

        insert_stmt = (
            f"INSERT INTO {table_name} (data_id, payload, title) "
            f"VALUES (:data_id, :payload, :title)"
        )
        params = {
            "data_id": data_id,
            "payload": serialized,
            "title": metadata.title,
        }

        try:
            from sqlalchemy import create_engine, text as sa_text

            engine = create_engine(self.connection_string)
            with engine.begin() as conn:
                conn.execute(sa_text(create_stmt))
                conn.execute(sa_text(insert_stmt), params)
            engine.dispose()
            logger.info("Stored generic data %s in %s", data_id, table_name)
        except Exception as e:
            logger.error("Failed to store generic data %s: %s", data_id, e)
            raise

    def decode_generic_payload(self, payload: str) -> Any:
        """Decode a ``generic_data_store`` payload written by ``_store_generic``.

        Plain JSON rows decode directly. Rows that fell back to pickle are
        stored as authenticated text envelopes and are only unpickled after
        HMAC verification, so a row rewritten in the database is rejected
        instead of executed.

        Args:
            payload: Raw ``payload`` column value.

        Returns:
            The decoded value.

        Raises:
            PayloadSecurityError: If the envelope fails HMAC verification.
            ValueError: If the payload is neither JSON nor a GISP1 envelope.
        """
        if not isinstance(payload, str):
            raise ValueError(
                f"Generic payload must be str, got {type(payload).__name__}"
            )
        if payload.startswith(TEXT_PREFIX):
            verified = verify_payload_text(
                payload,
                context=CONTEXT_STORAGE_GENERIC,
                key=self.signing_key,
            )
            return pickle.loads(verified)
        try:
            return json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Generic payload is neither JSON nor a GISP1 envelope: {exc}"
            ) from exc

    async def retrieve(self, data_id: str, query: dict[str, Any]) -> Any:
        """Retrieve data from PostgreSQL."""
        return await _maybe_await(self._retrieve_dataframe(data_id, query))

    async def delete(self, data_id: str) -> bool:
        """Delete data from PostgreSQL."""
        from sqlalchemy import create_engine, inspect as sqlalchemy_inspect, text

        table_name = validate_sql_identifier(f"dataset_{data_id.replace('-', '_')}")
        engine = create_engine(self.connection_string)
        try:
            if not sqlalchemy_inspect(engine).has_table(table_name):
                return False
            with engine.begin() as conn:
                conn.execute(text(f"DROP TABLE {table_name}"))
            return True
        finally:
            engine.dispose()


class MinIOBackend:
    """MinIO/S3 object storage backend."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.endpoint = config.get("endpoint")
        self.access_key = config.get("access_key")
        self.secret_key = config.get("secret_key")
        self.bucket = config.get("bucket", "geo-infer-data")
        self.compressor = DataCompressor()
        self.signing_key = config.get("signing_key")

    async def store(self, data: Any, metadata: DatasetMetadata) -> str:
        """Store data in MinIO."""
        data_id = f"minio_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S_%f')}_{uuid.uuid4().hex[:8]}"

        stored_id = await _maybe_await(self._store_to_minio(data, data_id, metadata))
        return stored_id if isinstance(stored_id, str) else data_id

    async def _store_to_minio(
        self, data: Any, data_id: str, metadata: DatasetMetadata
    ) -> str:
        """Store an object in MinIO/S3-compatible storage.

        Data is serialized and written to the configured object store.
        """
        if not all((self.endpoint, self.access_key, self.secret_key, self.bucket)):
            raise ValueError(
                "MinIO requires endpoint, access_key, secret_key, and bucket"
            )
        assert self.endpoint is not None
        assert self.access_key is not None
        assert self.secret_key is not None
        Minio = require_dependency("minio", "minio").Minio

        serialized_data = (
            self.compressor.compress_data(data)
            if self.compressor.is_enabled()
            else pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        )
        # Sign the bytes that actually land in the object store, whether they
        # are compressed or a raw pickle stream.
        serialized_data = sign_payload(
            serialized_data,
            context=CONTEXT_STORAGE_MINIO,
            key=self.signing_key,
        )
        client = Minio(
            self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=bool(self.config.get("secure", False)),
        )
        if not client.bucket_exists(self.bucket):
            client.make_bucket(self.bucket)
        client.put_object(
            self.bucket,
            f"{data_id}.bin",
            BytesIO(serialized_data),
            length=len(serialized_data),
            content_type="application/octet-stream",
        )

        return data_id

    async def retrieve(self, data_id: str, query: dict[str, Any]) -> Any:
        """Retrieve data from MinIO."""
        if not all((self.endpoint, self.access_key, self.secret_key, self.bucket)):
            raise ValueError(
                "MinIO requires endpoint, access_key, secret_key, and bucket"
            )
        assert self.endpoint is not None
        assert self.access_key is not None
        assert self.secret_key is not None
        Minio = require_dependency("minio", "minio").Minio

        client = Minio(
            self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=bool(self.config.get("secure", False)),
        )
        response = client.get_object(self.bucket, f"{data_id}.bin")
        try:
            envelope = response.read()
        finally:
            response.close()
            response.release_conn()
        # Trust boundary: verify before decompressing or unpickling, so a
        # hostile object is never inflated or deserialised.
        payload = verify_payload(
            envelope,
            context=CONTEXT_STORAGE_MINIO,
            key=self.signing_key,
        )
        return self.compressor.decompress_data(payload, verified=True)

    async def delete(self, data_id: str) -> bool:
        """Delete data from MinIO."""
        if not all((self.endpoint, self.access_key, self.secret_key, self.bucket)):
            raise ValueError(
                "MinIO requires endpoint, access_key, secret_key, and bucket"
            )
        assert self.endpoint is not None
        assert self.access_key is not None
        assert self.secret_key is not None
        Minio = require_dependency("minio", "minio").Minio

        client = Minio(
            self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=bool(self.config.get("secure", False)),
        )
        try:
            client.stat_object(self.bucket, f"{data_id}.bin")
        except Exception:
            return False
        client.remove_object(self.bucket, f"{data_id}.bin")
        return True


class RedisBackend:
    """Redis caching and storage backend."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 6379)
        self.db = config.get("db", 0)
        self.signing_key = config.get("signing_key")
        Redis = require_dependency("redis", "redis").Redis

        self.client = Redis(host=self.host, port=self.port, db=self.db)

    async def store(self, data: Any, metadata: DatasetMetadata) -> str:
        """Store data in Redis."""
        data_id = f"redis_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S_%f')}_{uuid.uuid4().hex[:8]}"
        payload = dumps_signed(
            {"data": data, "metadata": metadata.model_dump()},
            context=CONTEXT_STORAGE_REDIS,
            key=self.signing_key,
            serializer="pickle",
        )
        await asyncio.to_thread(self.client.set, data_id, payload)
        return data_id

    async def retrieve(self, data_id: str, query: dict[str, Any]) -> Any:
        """Retrieve data from Redis."""
        payload = await asyncio.to_thread(self.client.get, data_id)
        if payload is None:
            raise FileNotFoundError(f"Redis dataset {data_id!r} was not found")
        # Trust boundary: a Redis value is untrusted input until its envelope
        # verifies under the redis-backend context key.
        return loads_signed(
            cast(bytes, payload),
            context=CONTEXT_STORAGE_REDIS,
            key=self.signing_key,
            serializer="pickle",
        )["data"]

    async def delete(self, data_id: str) -> bool:
        """Delete data from Redis."""
        return bool(await asyncio.to_thread(self.client.delete, data_id))


class LocalFileBackend:
    """Local file system storage backend."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.base_path = Path(config.get("base_path", "/tmp/geo_infer_data"))
        self.compressor = DataCompressor()
        self.signing_key = config.get("signing_key")

        # Ensure base path exists
        self.base_path.mkdir(parents=True, exist_ok=True)

    async def store(self, data: Any, metadata: DatasetMetadata) -> str:
        """Store data in local file system."""
        data_id = f"local_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S_%f')}_{uuid.uuid4().hex[:8]}"

        # Determine file format and path
        file_path = self._get_file_path(data_id, metadata)

        # Serialize data
        if hasattr(data, "to_parquet"):
            try:
                data.to_parquet(file_path)
            except Exception:
                file_path = file_path.with_suffix(".pkl")
                self._write_signed_pickle(file_path, data)
        elif hasattr(data, "to_json"):
            data.to_json(file_path)
        else:
            file_path = file_path.with_suffix(".pkl")
            self._write_signed_pickle(file_path, data)

        # Store metadata
        metadata_path = file_path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(metadata.model_dump(), f, default=str)

        return data_id

    def _get_file_path(self, data_id: str, metadata: DatasetMetadata) -> Path:
        """Get file path for data storage."""
        # Organize by data type and date
        data_type = (
            "geospatial"
            if getattr(metadata, "spatial", None) is not None
            else "tabular"
        )
        date_str = datetime.now(UTC).strftime("%Y/%m/%d")

        file_path = self.base_path / data_type / date_str / f"{data_id}.parquet"
        file_path.parent.mkdir(parents=True, exist_ok=True)

        return file_path

    def _write_signed_pickle(self, file_path: Path, data: Any) -> None:
        """Write ``data`` as an authenticated pickle envelope.

        Args:
            file_path: Destination path (``.pkl``).
            data: Object to serialise.
        """
        file_path.write_bytes(
            dumps_signed(
                data,
                context=CONTEXT_STORAGE_LOCAL_FILE,
                key=self.signing_key,
                serializer="pickle",
            )
        )

    async def retrieve(self, data_id: str, query: dict[str, Any]) -> Any:
        """Retrieve data from local file system."""
        # Find data file
        data_file = self._find_data_file(data_id)
        if not data_file:
            raise FileNotFoundError(f"Data file not found for {data_id}")

        # Load data
        if data_file.suffix == ".parquet":
            try:
                return gpd.read_parquet(data_file)
            except ValueError:
                return pd.read_parquet(data_file)
        elif data_file.suffix == ".json":
            return gpd.read_file(data_file)
        else:
            # Trust boundary: pickled files on disk are verified before load.
            return loads_signed(
                data_file.read_bytes(),
                context=CONTEXT_STORAGE_LOCAL_FILE,
                key=self.signing_key,
                serializer="pickle",
            )

    def _find_data_file(self, data_id: str) -> Path | None:
        """Find data file by ID."""
        # Reject glob metacharacters and path separators so data_id can never
        # widen the search (e.g. data_id='*' or '../').
        if not data_id or any(ch in data_id for ch in "*?[]/\\"):
            raise ValueError(f"Invalid data_id for lookup: {data_id!r}")
        matches = sorted(self.base_path.rglob(f"{data_id}.*"))
        for file_path in matches:
            if file_path.suffix != ".json":
                return file_path
        if matches:
            return matches[0]
        return None

    async def delete(self, data_id: str) -> bool:
        """Delete data from local file system."""
        data_file = self._find_data_file(data_id)
        if data_file:
            data_file.unlink()
            # Also remove metadata file
            metadata_file = data_file.with_suffix(".json")
            if metadata_file.exists():
                metadata_file.unlink()
            return True
        return False


class AdaptiveDataStorage:
    """
    Adaptive data storage with automatic optimization based on access patterns.

    This class provides comprehensive data storage capabilities with intelligent
    optimization strategies that adapt to usage patterns, performance requirements,
    and cost considerations. It automatically selects the optimal storage backend
    based on data characteristics, access patterns, and performance requirements.

    The system supports multiple storage backends:
    - PostgreSQL/PostGIS: For relational geospatial data with complex queries
    - MinIO/S3: For large object storage and archival data
    - Redis: For high-speed caching and session data
    - Local Files: For development and small-scale deployments
    - Elasticsearch: For full-text search and analytics

    Features:
    - Automatic backend selection based on data characteristics
    - Spatial indexing (H3, Quadtree, R-tree) for efficient queries
    - Data compression and deduplication
    - Access pattern analysis and optimization
    - Performance monitoring and statistics
    - Cache management and optimization

    Attributes:
        config: Storage configuration with backend settings
        backend_manager: Manages different storage backends
        spatial_indexer: Handles spatial indexing operations
        temporal_indexer: Handles temporal indexing operations
        cache_manager: Manages caching for frequently accessed data
        compressor: Handles data compression and decompression
        access_patterns: Dictionary of analyzed access patterns by dataset
        storage_stats: Statistics and metrics for storage operations

    Methods:
        store_geospatial_data(): Store data with automatic optimization
        adaptive_query(): Query data with adaptive optimization
        optimize_for_patterns(): Optimize storage based on access patterns
        get_storage_stats(): Get comprehensive storage statistics
        _select_optimal_backend(): Select best backend for data storage
        _analyze_access_patterns(): Analyze and store access patterns
        _update_storage_stats(): Update storage operation statistics

    Args:
        storage_backends: List of storage backends to enable and use
        optimization_strategy: Storage optimization strategy ('access_pattern_based',
            'performance_focused', 'cost_optimized', 'balanced')
        compression_enabled: Whether to enable automatic data compression
        indexing_strategy: Spatial indexing strategy ('h3', 'r_tree',
            'geospatial', 'spatial_hash')
        caching_enabled: Whether to enable caching for frequently accessed data

    Raises:
        ConfigurationError: If storage configuration is invalid
        ConnectionError: If unable to connect to storage backends

    Examples:
        >>> # Initialize with multiple backends
        >>> storage = AdaptiveDataStorage(
        ...     storage_backends=['postgresql', 'minio', 'redis'],
        ...     optimization_strategy='access_pattern_based',
        ...     compression_enabled=True,
        ...     indexing_strategy='h3',
        ...     caching_enabled=True
        ... )
        >>>
        >>> # Store geospatial data with automatic optimization
        >>> data_id = await storage.store_geospatial_data(
        ...     spatial_data=geodataframe,
        ...     metadata=dataset_metadata,
        ...     access_patterns={
        ...         'spatial_queries': [{'bbox': [-122.5, 37.7, -122.3, 37.9]}],
        ...         'query_frequency': 'high',
        ...         'peak_hours': [9, 10, 11, 14, 15, 16]
        ...     }
        ... )
        >>>
        >>> # Query with adaptive optimization
        >>> results = await storage.adaptive_query(
        ...     spatial_bounds=[-122.5, 37.7, -122.3, 37.9],
        ...     temporal_range=(datetime(2023, 6, 1, tzinfo=UTC), datetime(2023, 6, 30, tzinfo=UTC)),
        ...     optimization_hints={'frequent_queries': True, 'real_time': False}
        ... )
        >>>
        >>> # Optimize storage based on usage patterns
        >>> optimizations = storage.optimize_for_patterns(access_patterns, "30d")
        >>> print(f"Applied {len(optimizations['actions'])} optimizations")
        >>>
        >>> # Get storage statistics
        >>> stats = storage.get_storage_stats()
        >>> print(f"Total datasets: {stats['datasets']}")
        >>> print(f"Total size: {stats['total_size'] / (1024*1024):.1f} MB")
    """

    def __init__(
        self,
        storage_backends: list[str],
        optimization_strategy: str = "balanced",
        compression_enabled: bool = True,
        indexing_strategy: str = "h3",
        caching_enabled: bool = True,
    ):
        supported_backends = {"postgresql", "minio", "redis", "local"}
        unknown_backends = sorted(set(storage_backends) - supported_backends)
        if unknown_backends:
            raise ValueError(
                f"Unsupported storage backend(s): {', '.join(unknown_backends)}"
            )

        self.config = StorageConfig(
            storage_backends=storage_backends,
            optimization_strategy=OptimizationStrategy(optimization_strategy),
            compression_enabled=compression_enabled,
            indexing_strategy=IndexingStrategy(indexing_strategy),
            caching_enabled=caching_enabled,
        )

        self.backend_manager = StorageBackendManager(self._get_backend_configs())
        self.spatial_indexer = SpatialIndexer()
        self.temporal_indexer = TemporalIndexer()
        self.cache_manager = CacheManager() if caching_enabled else None
        self.compressor = DataCompressor() if compression_enabled else None

        self.access_patterns: dict[str, AccessPattern] = {}
        self.storage_stats: dict[str, Any] = {}
        self._stored_data: dict[str, Any] = {}

        logger.info(
            f"Initialized AdaptiveDataStorage with {len(storage_backends)} backends"
        )

    def _get_backend_configs(self) -> dict[str, dict[str, Any]]:
        """Get backend configurations.

        Defaults describe local development services. Credentials are sourced
        from the environment so real deployments never rely on the built-in
        development defaults:

        - PostgreSQL: ``GEO_INFER_POSTGRES_USER`` (default ``geo_infer``),
          ``GEO_INFER_POSTGRES_PASSWORD`` (default ``password``).
        - MinIO/S3: ``GEO_INFER_MINIO_ACCESS_KEY`` and
          ``GEO_INFER_MINIO_SECRET_KEY`` (default ``minioadmin`` for a local
          MinIO dev server).
        """
        configs: dict[str, dict[str, Any]] = {
            "postgresql": {
                "type": "postgresql",
                "host": "localhost",
                "port": 5432,
                "user": os.environ.get("GEO_INFER_POSTGRES_USER", "geo_infer"),
                "password": os.environ.get("GEO_INFER_POSTGRES_PASSWORD", "password"),
                "database": "geo_infer_data",
            },
            "minio": {
                "type": "minio",
                "endpoint": "localhost:9000",
                "access_key": os.environ.get(
                    "GEO_INFER_MINIO_ACCESS_KEY", "minioadmin"
                ),
                "secret_key": os.environ.get(
                    "GEO_INFER_MINIO_SECRET_KEY", "minioadmin"
                ),
                "bucket": "geo-infer-data",
            },
            "redis": {"type": "redis", "host": "localhost", "port": 6379, "db": 0},
            "local": {"type": "local", "base_path": "/tmp/geo_infer_data"},
        }

        selected = {
            backend: configs.get(backend, {})
            for backend in self.config.storage_backends
        }
        self._audit_dev_credentials(selected)
        return selected

    _DEV_CREDENTIAL_ENV_DEFAULTS = (
        ("postgresql", "GEO_INFER_POSTGRES_PASSWORD", "password"),
        ("minio", "GEO_INFER_MINIO_ACCESS_KEY", "minioadmin"),
        ("minio", "GEO_INFER_MINIO_SECRET_KEY", "minioadmin"),
    )

    @staticmethod
    def _is_loopback_host(host: str) -> bool:
        """True when host is a loopback alias or address/host:port literal."""
        normalized = host.rsplit(":", 1)[0] if host.count(":") == 1 else host
        if normalized in ("localhost", "127.0.0.1", "::1"):
            return True
        return normalized.endswith(".localhost")

    def _audit_dev_credentials(self, configs: dict[str, dict[str, Any]]) -> None:
        """Warn or fail closed when built-in development credentials are in use.

        A missing environment variable means the built-in development default
        is in effect. Against a loopback host that is acceptable for local
        development but must be loud; against any non-loopback host it is
        refused outright (GS19-43).
        """
        for backend, env_var, dev_default in self._DEV_CREDENTIAL_ENV_DEFAULTS:
            config = configs.get(backend)
            if not config or os.environ.get(env_var):
                continue
            host = str(config.get("host") or config.get("endpoint") or "")
            if not self._is_loopback_host(host):
                raise RuntimeError(
                    f"{backend}: {env_var} is unset and host {host!r} is not "
                    "loopback; refusing to start with the built-in development "
                    f"credential default ({dev_default!r}). Set {env_var} "
                    "explicitly to proceed."
                )
            logger.warning(
                "%s: %s is unset; using the built-in development credential "
                "default against loopback host %r. Set %s before non-local use.",
                backend,
                env_var,
                host,
                env_var,
            )

    async def store_geospatial_data(
        self,
        spatial_data: Any,
        metadata: DatasetMetadata,
        access_patterns: dict[str, Any] | None = None,
    ) -> str:
        """
        Store geospatial data with automatic optimization.

        This method intelligently stores geospatial data by analyzing data characteristics,
        access patterns, and performance requirements to select the optimal storage backend
        and apply appropriate optimizations including compression, indexing, and caching.

        The method performs the following operations:
        1. Analyzes data characteristics (size, type, structure)
        2. Selects optimal storage backend based on analysis
        3. Applies data compression if enabled and beneficial
        4. Creates spatial and temporal indexes for efficient querying
        5. Stores metadata and access pattern information
        6. Updates storage statistics and performance metrics

        Supported data types:
        - GeoPandas GeoDataFrame: Vector geospatial data
        - Pandas DataFrame: Tabular data (will be converted if possible)
        - NumPy arrays: Raster and array data
        - Other formats: Generic data storage

        Args:
            spatial_data: Geospatial data to store. Supported formats include:
                - GeoPandas GeoDataFrame for vector data
                - Pandas DataFrame for tabular data
                - NumPy arrays for raster data
                - Other serializable objects
            metadata: Comprehensive dataset metadata including spatial extent,
                temporal extent, data lineage, quality information, and contact details
            access_patterns: Expected access patterns for optimization. Should include:
                - spatial_queries: List of expected spatial query bounds
                - temporal_queries: List of expected temporal query ranges
                - query_frequency: Expected query frequency ('low', 'medium', 'high')
                - peak_hours: List of peak usage hours (0-23)
                - batch_processing: Whether data will be used for batch processing

        Returns:
            Unique data identifier (string) for the stored dataset. This ID can be used
            for subsequent queries, updates, and management operations.

        Raises:
            StorageError: If storage operation fails due to backend issues
            ValidationError: If data or metadata validation fails
            CompressionError: If data compression fails
            IndexError: If index creation fails
            ValueError: If data format is not supported

        Examples:
            >>> # Store environmental monitoring data
            >>> data_id = await storage.store_geospatial_data(
            ...     spatial_data=sensor_geodataframe,
            ...     metadata=DatasetMetadata(
            ...         title="Environmental Sensors 2023",
            ...         description="Real-time environmental monitoring data",
            ...         spatial=SpatialExtent(bbox=[-122.6, 37.6, -122.2, 38.0]),
            ...         temporal=TemporalExtent(
            ...             start=datetime(2023, 1, 1, tzinfo=UTC),
            ...             end=datetime(2023, 12, 31, tzinfo=UTC)
            ...         )
            ...     ),
            ...     access_patterns={
            ...         'spatial_queries': [
            ...             {'bbox': [-122.5, 37.7, -122.3, 37.9], 'frequency': 'high'}
            ...         ],
            ...         'temporal_queries': [
            ...             {'start': datetime(2023, 6, 1, tzinfo=UTC), 'end': datetime(2023, 8, 31, tzinfo=UTC)}
            ...         ],
            ...         'query_frequency': 'high',
            ...         'peak_hours': [9, 10, 11, 14, 15, 16]
            ...     }
            ... )
            >>>
            >>> # Store large satellite imagery
            >>> imagery_id = await storage.store_geospatial_data(
            ...     spatial_data=satellite_array,
            ...     metadata=imagery_metadata,
            ...     access_patterns={'batch_processing': True, 'query_frequency': 'low'}
            ... )
        """
        logger.info(f"Storing geospatial data: {metadata.title}")

        # Analyze access patterns
        if access_patterns:
            self._analyze_access_patterns(metadata.title, access_patterns)

        # Select optimal backend
        optimal_backend = self._select_optimal_backend(
            spatial_data, metadata, access_patterns
        )

        # Store data
        data_id = await self.backend_manager.store_data(
            spatial_data, metadata, optimal_backend
        )
        self._stored_data[data_id] = spatial_data

        # Update storage statistics
        self._update_storage_stats(data_id, "store", metadata)

        # Cache metadata if caching enabled
        if self.cache_manager:
            await self.cache_manager.set(f"metadata_{data_id}", metadata.model_dump())

        logger.info(f"Successfully stored data with ID: {data_id}")
        return data_id

    async def retrieve_geospatial_data(
        self,
        data_id: str,
        spatial_bounds: list[float] | None = None,
        temporal_range: tuple[datetime, datetime] | None = None,
    ) -> Any:
        """Retrieve one stored dataset by its identifier.

        Data written during this process is returned directly. Other
        identifiers are delegated to the configured backend; missing data is
        reported rather than replaced with an unrelated query result.
        """
        temporal_range = normalize_temporal_range(temporal_range)
        if data_id in self._stored_data:
            return self._filter_stored_data(
                self._stored_data[data_id], spatial_bounds, temporal_range
            )

        query: dict[str, Any] = {}
        if spatial_bounds:
            query["spatial"] = spatial_bounds
        if temporal_range:
            query["temporal"] = temporal_range
        backend = self._select_backend_for_query(query, None)
        try:
            return await self.backend_manager.retrieve_data(data_id, query, backend)
        except ValueError as exc:
            raise KeyError(f"Dataset {data_id!r} is not stored") from exc

    @staticmethod
    def _filter_stored_data(
        data: Any,
        spatial_bounds: list[float] | None = None,
        temporal_range: tuple[datetime, datetime] | None = None,
    ) -> Any:
        """Apply spatial and temporal filters to one stored dataset.

        Non-tabular data is returned unchanged; (Geo)DataFrames are filtered
        by WGS84 bounds (GeoDataFrame only) and by their ``timestamp`` column.
        """
        temporal_range = normalize_temporal_range(temporal_range)
        if not isinstance(data, (pd.DataFrame, gpd.GeoDataFrame)):
            return data
        if spatial_bounds and isinstance(data, gpd.GeoDataFrame):
            min_lon, min_lat, max_lon, max_lat = spatial_bounds
            data = data.cx[min_lon:max_lon, min_lat:max_lat]  # type: ignore[misc]
        if temporal_range and "timestamp" in data.columns:
            start, end = temporal_range
            timestamps = normalize_observation_timestamps(data["timestamp"])
            data = data.loc[(timestamps >= start) & (timestamps <= end)]
        return data

    async def adaptive_query(
        self,
        spatial_bounds: list[float] | None = None,
        temporal_range: tuple[datetime, datetime] | None = None,
        optimization_hints: dict[str, Any] | None = None,
    ) -> Any:
        """
        Execute adaptive query with automatic optimization.

        Query every dataset stored in this process, applying the spatial and
        temporal filters to each one and merging the per-dataset results.

        Result merging:
        - A single matching dataset is returned as-is (GeoDataFrame, DataFrame,
          or any other stored object).
        - Multiple matching datasets that are all (Geo)DataFrames are
          concatenated into one frame.
        - Mixed or non-tabular results are returned as a list, one entry per
          stored dataset.
        - An empty store returns an empty list.

        Backend-resident datasets written by other processes are not scanned
        here; retrieve them by identifier with ``retrieve_geospatial_data``.

        Args:
            spatial_bounds: Spatial query bounds in the format
                [min_lon, min_lat, max_lon, max_lat]. If provided, results are
                filtered to only return data within these bounds.
            temporal_range: Temporal query range as (start_datetime, end_datetime).
                If provided, results are filtered by their ``timestamp`` column.
            optimization_hints: Reserved for query-execution tuning. Currently
                accepted but not interpreted.

        Returns:
            Filtered results as described above, or the cached result of a
            previous identical query when caching is enabled.

        Raises:
            ValueError: If query parameters are invalid.

        Examples:
            >>> # Spatial query for environmental data
            >>> results = await storage.adaptive_query(
            ...     spatial_bounds=[-122.5, 37.7, -122.3, 37.9],
            ...     temporal_range=(datetime(2023, 6, 1, tzinfo=UTC), datetime(2023, 6, 30, tzinfo=UTC)),
            ...     optimization_hints={
            ...         'frequent_queries': True,
            ...         'format': 'geojson',
            ...         'max_results': 1000
            ...     }
            ... )
            >>>
            >>> # Real-time sensor query
            >>> realtime_data = await storage.adaptive_query(
            ...     temporal_range=(datetime.now(UTC) - timedelta(hours=1), datetime.now(UTC)),
            ...     optimization_hints={
            ...         'real_time': True,
            ...         'batch_processing': False
            ...     }
            ... )
            >>>
            >>> # Batch processing query
            >>> batch_results = await storage.adaptive_query(
            ...     spatial_bounds=[-123.0, 37.0, -122.0, 38.0],
            ...     optimization_hints={
            ...         'batch_processing': True,
            ...         'format': 'parquet'
            ...     }
            ... )
        """
        logger.debug("Executing adaptive query")

        if spatial_bounds:
            if len(spatial_bounds) != 4:
                raise ValueError(
                    "spatial_bounds must be [min_lon, min_lat, max_lon, max_lat]"
                )
            min_lon, min_lat, max_lon, max_lat = spatial_bounds
            if not (
                -180 <= min_lon < max_lon <= 180 and -90 <= min_lat < max_lat <= 90
            ):
                raise ValueError("spatial_bounds contain invalid WGS84 coordinates")
        temporal_range = normalize_temporal_range(temporal_range)

        # Check cache first
        if self.cache_manager:
            cache_key = self._generate_cache_key(spatial_bounds, temporal_range)
            cached_result = await self.cache_manager.get(cache_key)
            if cached_result is not None:
                logger.debug("Returning cached query result")
                return cached_result

        # Scan every dataset stored in this process and filter each one.
        # Cross-backend search by shape is not supported by the backends
        # (they retrieve by concrete data_id); use retrieve_geospatial_data
        # for backend-resident datasets.
        filtered: list[Any] = [
            self._filter_stored_data(data, spatial_bounds, temporal_range)
            for data in self._stored_data.values()
        ]

        if not filtered:
            results: Any = []
        elif len(filtered) == 1:
            results = filtered[0]
        elif all(
            isinstance(result, (pd.DataFrame, gpd.GeoDataFrame)) for result in filtered
        ):
            results = pd.concat(filtered, ignore_index=True)
        else:
            results = filtered

        # Cache result if caching enabled
        if self.cache_manager:
            await self.cache_manager.set(cache_key, results, ttl=3600)  # 1 hour TTL

        return results

    def _analyze_access_patterns(
        self, dataset_id: str, patterns: dict[str, Any]
    ) -> None:
        """Analyze access patterns for optimization."""
        access_pattern = AccessPattern()

        # Analyze spatial patterns
        if "spatial_queries" in patterns:
            for query in patterns["spatial_queries"]:
                access_pattern.spatial_bounds.append(query.get("bbox", []))

        # Analyze temporal patterns
        if "temporal_queries" in patterns:
            for query in patterns["temporal_queries"]:
                start = query.get("start")
                end = query.get("end")
                if start and end:
                    access_pattern.temporal_ranges.append((start, end))

        # Analyze query frequency
        if "query_frequency" in patterns:
            frequency = patterns["query_frequency"]
            if isinstance(frequency, str):
                access_pattern.query_frequency[frequency] = (
                    access_pattern.query_frequency.get(frequency, 0) + 1
                )
            elif isinstance(frequency, dict):
                access_pattern.query_frequency.update(frequency)

        self.access_patterns[dataset_id] = access_pattern

    def _select_optimal_backend(
        self,
        data: Any,
        metadata: DatasetMetadata,
        access_patterns: dict[str, Any] | None,
    ) -> str:
        """Select optimal storage backend."""
        if "local" in self.backend_manager.backends:
            return "local"

        # Default backend selection logic
        if isinstance(data, (gpd.GeoDataFrame, pd.DataFrame)):
            if len(data) > 100000:  # Large dataset
                preferred = "minio"
            else:
                preferred = "postgresql"
        else:
            preferred = "local"

        if preferred in self.backend_manager.backends:
            return preferred
        if "local" in self.backend_manager.backends:
            return "local"
        return next(iter(self.backend_manager.backends))

    def _select_backend_for_query(
        self, query: dict[str, Any], hints: dict[str, Any] | None
    ) -> str:
        """Select optimal backend for query execution."""
        # Query optimization logic
        if hints and hints.get("real_time"):
            preferred = "redis"
        elif "spatial" in query:
            preferred = "postgresql"
        else:
            preferred = "postgresql"

        if preferred in self.backend_manager.backends:
            return preferred
        if "local" in self.backend_manager.backends:
            return "local"
        return next(iter(self.backend_manager.backends))

    def _generate_cache_key(
        self,
        spatial_bounds: list[float] | None,
        temporal_range: tuple[datetime, datetime] | None,
    ) -> str:
        """Generate cache key for query."""
        spatial_str = (
            f"spatial_{'_'.join(map(str, spatial_bounds))}"
            if spatial_bounds
            else "no_spatial"
        )
        temporal_str = (
            f"temporal_{temporal_range[0].isoformat()}_{temporal_range[1].isoformat()}"
            if temporal_range
            else "no_temporal"
        )
        return f"query_{spatial_str}_{temporal_str}"

    def _update_storage_stats(
        self, data_id: str, operation: str, metadata: DatasetMetadata
    ) -> None:
        """Update storage statistics."""
        if data_id not in self.storage_stats:
            self.storage_stats[data_id] = {
                "created_at": datetime.now(UTC),
                "operations": [],
                "size": metadata.file_size or 0,
            }

        self.storage_stats[data_id]["operations"].append(
            {
                "operation": operation,
                "timestamp": datetime.now(UTC),
                "size": metadata.file_size or 0,
            }
        )

    def optimize_for_patterns(
        self, patterns: dict[str, Any], time_window: str = "30d"
    ) -> dict[str, Any]:
        """
        Optimize storage based on access patterns.

        Args:
            patterns: Access patterns to analyze
            time_window: Time window for pattern analysis

        Returns:
            Optimization actions taken
        """
        logger.info("Optimizing storage for access patterns")

        actions = []
        optimizations: dict[str, Any] = {}

        for dataset_id, pattern in patterns.items():
            # Analyze pattern and determine optimizations
            if pattern.get("frequent_queries"):
                # Move frequently accessed data to faster storage
                actions.append(f"Move {dataset_id} to cache")
                optimizations[dataset_id] = {"storage": "cache"}

            if pattern.get("batch_processing"):
                # Optimize for batch processing
                actions.append(f"Optimize {dataset_id} for batch processing")
                optimizations[dataset_id] = {"chunking": True}

        return {
            "actions": actions,
            "optimizations": optimizations,
            "timestamp": datetime.now(UTC),
            "time_window": time_window,
        }

    async def optimize_storage_for_patterns(
        self, patterns: dict[str, Any]
    ) -> dict[str, Any]:
        """Apply storage optimizations based on patterns."""
        return self.optimize_for_patterns(patterns)

    def get_storage_stats(self) -> dict[str, Any]:
        """Get storage statistics."""
        return {
            "backends": list(self.backend_manager.backends.keys()),
            "datasets": len(self.storage_stats),
            "total_size": sum(
                stats.get("size", 0) for stats in self.storage_stats.values()
            ),
            "optimization_strategy": self.config.optimization_strategy.value,
            "compression_enabled": self.config.compression_enabled,
            "caching_enabled": self.config.caching_enabled,
        }

    async def close(self) -> None:
        """Release local storage resources held by the storage facade."""
        self._stored_data.clear()
