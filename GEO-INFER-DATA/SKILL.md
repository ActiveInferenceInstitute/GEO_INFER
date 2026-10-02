---
name: geo-infer-data
description: Data connectors, ETL pipelines, and data management for geospatial datasets. Use when loading spatial data from databases, APIs, files (GeoJSON, Shapefile, GeoParquet), or building data transformation pipelines.
prerequisites:
  required: []
  recommended:
    - geo-infer-math
difficulty: beginner
estimated_time: 30min
examples_dir: ../GEO-INFER-EXAMPLES/examples/
---

# GEO-INFER-DATA

## Instructions

### Core Capabilities

- **Connectors**: PostgreSQL/PostGIS, SQLite/SpatiaLite, MongoDB, REST/GraphQL APIs, MQTT/WebSocket streams, file I/O (GeoJSON, Shapefile, GeoPackage, GeoTIFF, CSV, Parquet), AWS S3 (boto3)
- **ETL pipelines**: `IntelligentETLPipeline` with dependency resolution and intelligent retry
- **Storage**: `AdaptiveDataStorage` with local-file, PostgreSQL, MinIO/S3, and Redis backends
- **Caching**: In-memory TTL caching and signed persistence (GISP1 HMAC envelopes)
- **Validation**: Geometry validity, WGS84 bounds, CRS, temporal, and completeness checks

### Key Imports

```python
from geo_infer_data import (
    MultiSourceDataIngestion,
    IntelligentETLPipeline,
    AdaptiveDataStorage,
    DataQualityManager,
)
from geo_infer_data.connectors.file import FileConnector
from geo_infer_data.connectors.database import DatabaseConnector
from geo_infer_data.connectors.cloud import S3Connector
from geo_infer_data.utils.validation import GeospatialValidator
from geo_infer_data.utils.indexing import SpatialIndexer
from geo_infer_data.models.schemas import DatasetMetadata, DataLineage
```

## Examples

Read and validate a GeoJSON file:

```python
import asyncio

from geo_infer_data.connectors.file import FileConnector
from geo_infer_data.utils.validation import GeospatialValidator


async def main() -> None:
    connector = FileConnector()
    gdf = await connector.read_geospatial("data/buildings.geojson")
    print(f"Loaded {len(gdf)} features")

    check = await GeospatialValidator().validate_data(gdf)
    print(f"Quality: {check.score:.2f} ({check.status})")


asyncio.run(main())
```

Store data and query it back with spatial/temporal filters:

```python
import asyncio

from geo_infer_data import AdaptiveDataStorage
from geo_infer_data.models.schemas import DatasetMetadata, DataLineage


async def main() -> None:
    storage = AdaptiveDataStorage(storage_backends=["local"])
    metadata = DatasetMetadata(
        title="sensor readings",
        lineage=DataLineage(source="demo", process="ingest", created_by="script"),
    )

    data_id = await storage.store_geospatial_data(gdf, metadata)
    results = await storage.adaptive_query(
        spatial_bounds=[-122.5, 37.7, -122.3, 37.9],
        temporal_range=(start, end),
    )
    print(f"Stored {data_id}; query returned {len(results)} rows")


asyncio.run(main())
```

Run the ETL pipeline:

```python
from geo_infer_data import IntelligentETLPipeline

pipeline = IntelligentETLPipeline(workflow_config=None, error_recovery="intelligent_retry")
result = await pipeline.execute_workflow(
    source_data=raw_data,
    target_storage=storage,
    transformation_rules={"reproject_to": "EPSG:4326", "validate_bounds": True},
)
```

## Guidelines

- SQL uses parameterized queries (`:param` placeholders) — never string interpolation
- All coordinate data validated against WGS84 bounds
- H3 indexing uses the v4 API (`latlng_to_cell`, `cell_to_latlng`, `geo_to_cells`)
- Test: `uv run python GEO-INFER-TEST/run_unified_tests.py --module DATA`

### Integrations

- **SPACE** → Spatial indexing of loaded datasets
- **GIT** → Version control for spatial data
- **API** → Data source for spatial query endpoints
- **IOT** → Sensor data ingestion pipelines
- **EXAMPLES** → Example ETL workflows

Dataset/model and cache timestamps require explicit timezone offsets and normalize
to UTC through `geo_infer_time.core.timestamps.normalize_timestamp`. Select backend
extras explicitly: `postgres` (psycopg2 + asyncpg), `s3` (boto3), `minio`, `redis`,
and `raster` (rasterio). Base/local operations do not import those clients; an
unavailable selected backend raises an error naming its installation extra.
HTTP connector calls and downloads run off the event loop; requests are spaced
using a monotonic clock, and failed pagination does not return partial success.

### DuckDB Spatial verification

Provision the signed Spatial extension matching the installed DuckDB runtime with
`uv run python GEO-INFER-DATA/provision_duckdb_spatial.py`. Provisioning failures
are fatal. `read_cloud_native_vector(path, require_duckdb=True)` requires the
actual DuckDB path and propagates backend errors. Normal reads load an installed
extension without downloading one; reader layer/keyword options use GeoPandas
when they are unsupported by DuckDB.
