"""
Database connectors for GEO-INFER-DATA.

This module provides comprehensive database connectivity for various
database systems including PostgreSQL, MySQL, MongoDB, and specialized
geospatial databases.
"""

import logging
from typing import Any
from datetime import datetime, UTC

import geopandas as gpd
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

from ..models.schemas import DatasetMetadata
from ..utils.identifiers import validate_sql_identifier
from ..utils.dependencies import require_dependency

logger = logging.getLogger(__name__)


class DatabaseConnector:
    """
    Universal database connector for geospatial data.

    This class provides connectivity to various database systems including
    PostgreSQL/PostGIS, MySQL, MongoDB, and other geospatial databases
    with automatic schema detection and query optimization.

    Args:
        connection_type: Type of database ('postgresql', 'mysql', 'mongodb', 'sqlite')
        connection_string: Database connection string
        pool_size: Connection pool size
        max_overflow: Maximum overflow connections
        enable_geospatial: Whether to enable geospatial operations

    Examples:
        >>> # PostgreSQL connection
        >>> connector = DatabaseConnector(
        ...     connection_type='postgresql',
        ...     connection_string='postgresql://user:pass@localhost/geodata'
        ... )
        >>>
        >>> # Query geospatial data
        >>> gdf = await connector.query_geospatial(
        ...     table_name='weather_stations',
        ...     spatial_filter={'bbox': [-122.5, 37.7, -122.3, 37.9]}
        ... )
        >>>
        >>> # Insert data with metadata
        >>> await connector.insert_geospatial_data(data, metadata)
    """

    def __init__(
        self,
        connection_type: str,
        connection_string: str,
        pool_size: int = 10,
        max_overflow: int = 20,
        enable_geospatial: bool = True,
    ):
        self.connection_type = connection_type
        self.connection_string = connection_string
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.enable_geospatial = enable_geospatial

        self.engine: Any = None
        self.async_engine: Any = None
        self.SessionLocal: Any = None
        self.AsyncSessionLocal: Any = None

        self._initialize_connection()

        logger.info(f"Initialized {connection_type} database connector")

    def _initialize_connection(self) -> None:
        """Initialize database connection."""
        if self.connection_type == "postgresql":
            require_dependency("psycopg2", "postgres")
            require_dependency("asyncpg", "postgres")
        try:
            # Create synchronous engine
            self.engine = create_engine(
                self.connection_string,
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                echo=False,  # Set to True for SQL debugging
            )

            # Create session factory
            self.SessionLocal = sessionmaker(bind=self.engine)

            # Create async engine for async operations
            if self.connection_type in ["postgresql", "mysql"]:
                async_url = self.connection_string.replace(
                    "postgresql://", "postgresql+asyncpg://"
                )
                async_url = async_url.replace("mysql://", "mysql+asyncmy://")

                self.async_engine = create_async_engine(async_url)
                self.AsyncSessionLocal = sessionmaker(
                    bind=self.async_engine, class_=AsyncSession, expire_on_commit=False
                )

            logger.info(f"Database connection established for {self.connection_type}")

        except Exception as e:
            logger.error(f"Failed to initialize database connection: {e}")
            raise

    async def test_connection(self) -> bool:
        """
        Test database connection.

        Returns:
            True if connection is successful
        """
        try:
            if self.async_engine:
                async with self.AsyncSessionLocal() as session:
                    await session.execute(text("SELECT 1"))
            else:
                with self.SessionLocal() as session:
                    session.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False

    async def query_geospatial(
        self,
        table_name: str,
        columns: list[str] | None = None,
        spatial_filter: dict[str, Any] | None = None,
        temporal_filter: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> gpd.GeoDataFrame:
        """
        Query geospatial data with spatial and temporal filters.

        Args:
            table_name: Name of the table to query
            columns: Columns to select (default: all)
            spatial_filter: Spatial filtering options
            temporal_filter: Temporal filtering options
            limit: Maximum number of records

        Returns:
            GeoDataFrame with query results
        """
        logger.info(f"Querying geospatial data from {table_name}")

        query, params = self._build_select_query(
            table_name=table_name,
            columns=columns,
            spatial_filter=spatial_filter,
            temporal_filter=temporal_filter,
            limit=limit,
            enable_geospatial=self.enable_geospatial,
        )

        try:
            # Execute query
            if self.async_engine:
                async with self.AsyncSessionLocal() as session:
                    result = await session.execute(text(query), params)
                    df = pd.DataFrame(
                        result.fetchall(),
                        columns=result.keys() if hasattr(result, "keys") else None,
                    )
            else:
                with self.SessionLocal() as session:
                    result = session.execute(text(query), params)
                    df = pd.DataFrame(
                        result.fetchall(),
                        columns=result.keys() if hasattr(result, "keys") else None,
                    )

            # Convert to GeoDataFrame if geometry column exists
            if "geom" in df.columns:
                df["geom"] = df["geom"].apply(
                    lambda x: x.wkt if hasattr(x, "wkt") else str(x)
                )
                gdf = gpd.GeoDataFrame(
                    df, geometry=gpd.GeoSeries.from_wkt(df["geom"]), crs="EPSG:4326"
                )
                gdf.drop("geom", axis=1, inplace=True)
                return gdf
            else:
                return gpd.GeoDataFrame(df)

        except Exception as e:
            logger.error(f"Geospatial query failed: {e}")
            raise

    @staticmethod
    def _build_select_query(
        table_name: str,
        columns: list[str] | None,
        spatial_filter: dict[str, Any] | None,
        temporal_filter: dict[str, Any] | None,
        limit: int | None,
        enable_geospatial: bool,
    ) -> tuple[str, dict[str, Any]]:
        """Build a SELECT statement with every interpolated identifier validated.

        All identifiers (table, columns, time column) pass through
        :func:`validate_sql_identifier` and the row limit is coerced to
        ``int``; filter *values* travel as bound parameters.

        Returns:
            Tuple of ``(query, params)`` ready for SQLAlchemy ``text()``.

        Raises:
            ValueError: If any identifier or limit is unsafe.
        """
        safe_table = validate_sql_identifier(table_name)
        select_clause = (
            ", ".join(f'"{validate_sql_identifier(c)}"' for c in columns)
            if columns
            else "*"
        )
        query = f"SELECT {select_clause} FROM {safe_table}"

        conditions: list[str] = []
        params: dict[str, Any] = {}

        if spatial_filter and enable_geospatial and "bbox" in spatial_filter:
            bbox = spatial_filter["bbox"]
            if len(bbox) >= 4:
                min_lon, min_lat, max_lon, max_lat = (float(v) for v in bbox[:4])
                conditions.append(
                    "ST_Intersects(geom, ST_MakeEnvelope(:min_lon, :min_lat, "
                    ":max_lon, :max_lat, 4326))"
                )
                params.update(
                    {
                        "min_lon": min_lon,
                        "min_lat": min_lat,
                        "max_lon": max_lon,
                        "max_lat": max_lat,
                    }
                )

        if temporal_filter:
            time_column = validate_sql_identifier(
                temporal_filter.get("column", "timestamp")
            )
            if "start" in temporal_filter:
                conditions.append(f"{time_column} >= :start_time")
                params["start_time"] = temporal_filter["start"]
            if "end" in temporal_filter:
                conditions.append(f"{time_column} <= :end_time")
                params["end_time"] = temporal_filter["end"]

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        if limit:
            query += f" LIMIT {int(limit)}"

        return query, params

    async def insert_geospatial_data(
        self,
        data: pd.DataFrame | gpd.GeoDataFrame,
        metadata: DatasetMetadata,
        table_name: str | None = None,
        if_exists: str = "replace",
    ) -> bool:
        """
        Insert geospatial data into database.

        Args:
            data: Data to insert
            metadata: Dataset metadata
            table_name: Target table name
            if_exists: Behavior if table exists ('replace', 'append', 'fail')

        Returns:
            True if successful
        """
        logger.info(f"Inserting geospatial data: {metadata.title}")

        if table_name is None:
            table_name = metadata.title.lower().replace(" ", "_").replace("-", "_")
        # Validate only after the title -> table_name transform so a hostile
        # title can never smuggle characters into DDL.
        table_name = validate_sql_identifier(table_name)

        try:
            # Convert to GeoDataFrame if needed
            if isinstance(data, pd.DataFrame) and "geometry" not in data.columns:
                # Try to create geometry from lat/lon columns
                if "latitude" in data.columns and "longitude" in data.columns:
                    gdf = gpd.GeoDataFrame(
                        data,
                        geometry=gpd.points_from_xy(data.longitude, data.latitude),
                        crs="EPSG:4326",
                    )
                else:
                    gdf = gpd.GeoDataFrame(data)
            else:
                gdf = (
                    data
                    if isinstance(data, gpd.GeoDataFrame)
                    else gpd.GeoDataFrame(data)
                )

            # Create table schema based on data
            await self._create_table_schema(gdf, table_name, metadata)

            # Insert data
            if self.async_engine:
                async with self.AsyncSessionLocal() as session:
                    # Convert to PostGIS format
                    if hasattr(gdf, "to_postgis"):
                        gdf.to_postgis(
                            table_name, self.async_engine, if_exists=if_exists
                        )
                    else:
                        # Manual insertion
                        await self._insert_data_manually(session, gdf, table_name)
                    await session.commit()
            else:
                if hasattr(gdf, "to_sql"):
                    gdf.to_sql(
                        table_name, self.engine, if_exists=if_exists, index=False
                    )
                else:
                    with self.SessionLocal() as session:
                        await self._insert_data_manually(session, gdf, table_name)
                        session.commit()

            # Create spatial indexes
            await self._create_spatial_indexes(table_name, gdf)

            logger.info(f"Successfully inserted data into {table_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to insert geospatial data: {e}")
            raise

    async def _create_table_schema(
        self, gdf: gpd.GeoDataFrame, table_name: str, metadata: DatasetMetadata
    ) -> None:
        """Create database table schema from GeoDataFrame structure.

        Generates and executes a CREATE TABLE statement that mirrors the
        column names and dtypes of the provided GeoDataFrame.
        """
        dtype_map = {
            "int64": "BIGINT",
            "int32": "INTEGER",
            "float64": "DOUBLE PRECISION",
            "float32": "REAL",
            "bool": "BOOLEAN",
            "datetime64[ns]": "TIMESTAMP",
            "object": "TEXT",
            "geometry": "GEOMETRY",
        }

        columns_sql = []
        for col in gdf.columns:
            col_dtype = str(gdf[col].dtype)
            sql_type = dtype_map.get(col_dtype, "TEXT")
            # Normalise then validate the column name; non-identifier columns
            # are rejected rather than escaped.
            safe_col = validate_sql_identifier(col.replace(" ", "_").replace("-", "_"))
            columns_sql.append(f'"{safe_col}" {sql_type}')

        safe_table = validate_sql_identifier(table_name)
        create_stmt = (
            f"CREATE TABLE IF NOT EXISTS {safe_table} ({', '.join(columns_sql)})"
        )

        try:
            if self.async_engine:
                async with self.AsyncSessionLocal() as session_obj:
                    await session_obj.execute(text(create_stmt))
                    await session_obj.commit()
            else:
                with self.SessionLocal() as session_obj:
                    session_obj.execute(text(create_stmt))
                    session_obj.commit()
            logger.info(
                "Created table schema for %s with %d columns",
                table_name,
                len(columns_sql),
            )
        except Exception as e:
            logger.error("Failed to create table schema for %s: %s", table_name, e)
            raise

    async def _insert_data_manually(
        self, session: Any, gdf: gpd.GeoDataFrame, table_name: str
    ) -> None:
        """Manually insert data row by row.

        Used when to_sql or to_postgis is not available.  Builds
        parameterised INSERT statements from the GeoDataFrame rows.
        """
        if len(gdf) == 0:
            logger.info("No rows to insert into %s", table_name)
            return

        safe_table = validate_sql_identifier(table_name)
        columns = [
            validate_sql_identifier(c.replace(" ", "_").replace("-", "_"))
            for c in gdf.columns
        ]
        col_list = ", ".join(f'"{c}"' for c in columns)
        baselines = ", ".join(f":{c}" for c in columns)
        insert_stmt = text(
            f"INSERT INTO {safe_table} ({col_list}) VALUES ({baselines})"
        )

        rows_inserted = 0
        for _, row in gdf.iterrows():
            params = {}
            for orig_col, safe_col in zip(gdf.columns, columns):
                val = row[orig_col]
                # Convert numpy types to Python builtins for DB driver compat
                if hasattr(val, "item"):
                    val = val.item()
                elif hasattr(val, "wkt"):
                    val = val.wkt
                params[safe_col] = val

            session.execute(insert_stmt, params)
            rows_inserted += 1

        logger.info("Manually inserted %d rows into %s", rows_inserted, table_name)

    async def _create_spatial_indexes(
        self, table_name: str, gdf: gpd.GeoDataFrame
    ) -> None:
        """Create spatial indexes for geometry columns."""
        if not self.enable_geospatial:
            return

        try:
            safe_table = validate_sql_identifier(table_name)
            index_query = (
                f"CREATE INDEX IF NOT EXISTS idx_{safe_table}_geom "
                f"ON {safe_table} USING GIST (geom)"
            )

            if self.async_engine:
                async with self.AsyncSessionLocal() as session:
                    await session.execute(text(index_query))
                    await session.commit()
            else:
                with self.SessionLocal() as session:
                    session.execute(text(index_query))
                    session.commit()

            logger.info(f"Created spatial index for {table_name}")

        except Exception as e:
            logger.warning(f"Failed to create spatial index: {e}")

    async def get_table_schema(self, table_name: str) -> dict[str, Any]:
        """
        Get table schema information.

        Args:
            table_name: Table name

        Returns:
            Schema information
        """
        try:
            if self.async_engine:
                async with self.AsyncSessionLocal() as session:
                    # Get column information
                    result = await session.execute(
                        text(
                            """
                        SELECT column_name, data_type, is_nullable, column_default
                        FROM information_schema.columns
                        WHERE table_name = :table_name
                        ORDER BY ordinal_position
                    """
                        ),
                        {"table_name": table_name},
                    )

                    columns = result.fetchall()
                    schema = {
                        "table_name": table_name,
                        "columns": [
                            {
                                "name": col[0],
                                "type": col[1],
                                "nullable": col[2] == "YES",
                                "default": col[3],
                            }
                            for col in columns
                        ],
                    }

            else:
                with self.SessionLocal() as session:
                    result = session.execute(
                        text(
                            """
                        SELECT column_name, data_type, is_nullable, column_default
                        FROM information_schema.columns
                        WHERE table_name = :table_name
                        ORDER BY ordinal_position
                    """
                        ),
                        {"table_name": table_name},
                    )

                    columns = result.fetchall()
                    schema = {
                        "table_name": table_name,
                        "columns": [
                            {
                                "name": col[0],
                                "type": col[1],
                                "nullable": col[2] == "YES",
                                "default": col[3],
                            }
                            for col in columns
                        ],
                    }

            return schema

        except Exception as e:
            logger.error(f"Failed to get table schema: {e}")
            return {"table_name": table_name, "columns": [], "error": str(e)}

    async def list_tables(self, schema: str = "public") -> list[str]:
        """
        List available tables in database.

        Args:
            schema: Database schema name

        Returns:
            List of table names
        """
        try:
            if self.async_engine:
                async with self.AsyncSessionLocal() as session:
                    result = await session.execute(
                        text(
                            """
                        SELECT table_name
                        FROM information_schema.tables
                        WHERE table_schema = :schema
                        AND table_type = 'BASE TABLE'
                        ORDER BY table_name
                    """
                        ),
                        {"schema": schema},
                    )

                    tables = [row[0] for row in result.fetchall()]

            else:
                with self.SessionLocal() as session:
                    result = session.execute(
                        text(
                            """
                        SELECT table_name
                        FROM information_schema.tables
                        WHERE table_schema = :schema
                        AND table_type = 'BASE TABLE'
                        ORDER BY table_name
                    """
                        ),
                        {"schema": schema},
                    )

                    tables = [row[0] for row in result.fetchall()]

            return tables

        except Exception as e:
            logger.error(f"Failed to list tables: {e}")
            return []

    async def close(self) -> None:
        """Close database connections."""
        if self.async_engine:
            await self.async_engine.dispose()
        if self.engine:
            self.engine.dispose()
        logger.info("Database connections closed")


class PostgreSQLConnector(DatabaseConnector):
    """
    PostgreSQL/PostGIS specialized connector.

    This class provides enhanced functionality for PostgreSQL databases
    with PostGIS spatial extensions.
    """

    def __init__(self, connection_string: str, **kwargs: Any):
        super().__init__("postgresql", connection_string, **kwargs)

    async def create_postgis_extension(self) -> None:
        """Create PostGIS extension if not exists."""
        try:
            async with self.AsyncSessionLocal() as session:
                await session.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
                await session.execute(
                    text("CREATE EXTENSION IF NOT EXISTS postgis_topology")
                )
                await session.commit()
            logger.info("PostGIS extensions created")
        except Exception as e:
            logger.warning(f"Failed to create PostGIS extensions: {e}")


class MySQLConnector(DatabaseConnector):
    """
    MySQL database connector.

    This class provides MySQL connectivity with spatial support.
    """

    def __init__(self, connection_string: str, **kwargs: Any):
        super().__init__("mysql", connection_string, **kwargs)


class MongoDBConnector:
    """
    MongoDB connector for document-based geospatial data.

    This class provides MongoDB connectivity with geospatial querying
    capabilities using GeoJSON and 2dsphere indexes.
    """

    def __init__(self, connection_string: str, database_name: str):
        self.connection_string = connection_string
        self.database_name = database_name
        self.client: Any = None
        self.database: Any = None

        self._initialize_connection()

        logger.info(f"Initialized MongoDB connector for database: {database_name}")

    def _initialize_connection(self) -> None:
        """Initialize MongoDB connection."""
        try:
            from pymongo import MongoClient  # type: ignore[import-not-found]

            self.client = MongoClient(self.connection_string)
            self.database = self.client[self.database_name]
            logger.info("MongoDB connection established")
        except Exception as e:
            logger.error(f"Failed to initialize MongoDB connection: {e}")
            raise

    async def insert_geospatial_document(
        self, collection_name: str, document: dict[str, Any]
    ) -> str:
        """
        Insert geospatial document into MongoDB collection.

        Args:
            collection_name: MongoDB collection name
            document: Document to insert

        Returns:
            Inserted document ID
        """
        try:
            collection = self.database[collection_name]

            # Add timestamp if not present
            if "created_at" not in document:
                document["created_at"] = datetime.now(UTC)

            result = collection.insert_one(document)
            return str(result.inserted_id)

        except Exception as e:
            logger.error(f"Failed to insert document: {e}")
            raise

    async def query_geospatial_collection(
        self,
        collection_name: str,
        spatial_filter: dict[str, Any] | None = None,
        query_filter: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Query geospatial collection with spatial filters.

        Args:
            collection_name: Collection name
            spatial_filter: Spatial filtering criteria
            query_filter: Additional query filters
            limit: Maximum number of documents

        Returns:
            List of matching documents
        """
        try:
            collection = self.database[collection_name]

            # Build query
            mongo_query = {}

            if query_filter:
                mongo_query.update(query_filter)

            if spatial_filter:
                if "near" in spatial_filter:
                    near = spatial_filter["near"]
                    mongo_query["location"] = {
                        "$near": {
                            "$geometry": near["geometry"],
                            "$maxDistance": near.get("max_distance", 1000),
                        }
                    }
                elif "within" in spatial_filter:
                    within = spatial_filter["within"]
                    mongo_query["location"] = {
                        "$geoWithin": {"$geometry": within["geometry"]}
                    }

            # Execute query
            cursor = collection.find(mongo_query)
            if limit:
                cursor = cursor.limit(limit)

            documents = list(cursor)
            return documents

        except Exception as e:
            logger.error(f"Failed to query geospatial collection: {e}")
            raise

    async def create_geospatial_index(
        self, collection_name: str, field_name: str = "location"
    ) -> None:
        """
        Create geospatial index on collection.

        Args:
            collection_name: Collection name
            field_name: Field to index
        """
        try:
            collection = self.database[collection_name]
            collection.create_index([(field_name, "2dsphere")])
            logger.info(f"Created 2dsphere index on {collection_name}.{field_name}")
        except Exception as e:
            logger.error(f"Failed to create geospatial index: {e}")
            raise

    async def close(self) -> None:
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
        logger.info("MongoDB connection closed")
