# External Systems Integration

This page presents application-level patterns for external databases, files,
providers, and services. Its network/file examples require caller-provisioned
inputs, dependencies, endpoints, credentials, and authorization; they are not
self-contained offline examples or evidence that those services were qualified.
Example hosts, database names, paths, and credentials are illustrative.
Use the owning package's source and tests to establish which adapter is actually
implemented. Third-party client usage is identified separately below.

## Overview

GEO-INFER modules consume external data through several patterns: direct database connections, REST API clients, streaming pipelines, and cloud storage adapters. GEO-INFER-DATA serves as the primary data ingestion layer, but individual modules can also connect to external systems directly when needed.

## PostGIS / PostgreSQL

PostGIS is an optional application database choice, not a repository-wide
default service. The following examples use GeoPandas and SQLAlchemy directly
and require a provisioned PostgreSQL/PostGIS instance and client dependencies.
SPACE configuration metadata alone does not establish a database connector or a
completed connection. DATA's PostgreSQL integration has its own dependency and
connection contract.

```python
import geopandas as gpd
from sqlalchemy import create_engine

# Connection string with PostGIS-enabled PostgreSQL
engine = create_engine(
    "postgresql+psycopg2://geo_user:password@localhost:5432/geo_infer_db"
)

# Read spatial data with a bounding box filter
query = """
    SELECT id, name, category, geom
    FROM land_parcels
    WHERE ST_Intersects(
        geom,
        ST_MakeEnvelope(-122.5, 47.5, -122.0, 47.8, 4326)
    )
"""
gdf = gpd.read_postgis(query, engine, geom_col="geom", crs="EPSG:4326")

# Write results back to PostGIS
gdf.to_postgis("analysis_results", engine, if_exists="replace", index=False)
```

The next third-party example configures SQLAlchemy pool parameters against an
application-provisioned pooler. This repository does not provision or qualify
that service. Choose connection limits for the actual server and workload.

```python
engine = create_engine(
    "postgresql+psycopg2://geo_user:password@pgbouncer:6432/geo_infer_db",
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=3600,
)
```

## GDAL/OGR Data Sources

GDAL/OGR provides access to over 200 raster and vector formats. GEO-INFER uses GDAL through `rasterio` (rasters) and `fiona`/`geopandas` (vectors).

```python
import rasterio
from rasterio.windows import from_bounds

# Read a windowed subset of a large GeoTIFF
with rasterio.open("/data/elevation_dem.tif") as src:
    window = from_bounds(
        left=-122.5, bottom=47.5, right=-122.0, top=47.8,
        transform=src.transform
    )
    elevation = src.read(1, window=window)
    profile = src.profile
    profile.update(
        width=window.width,
        height=window.height,
        transform=rasterio.windows.transform(window, src.transform),
    )
```

For vector data from file geodatabases, shapefiles, or GeoPackage:

```python
import geopandas as gpd

# Read from a GeoPackage
gdf = gpd.read_file("watersheds.gpkg", layer="huc12_boundaries")

# Read from a file geodatabase with a bounding box filter
gdf = gpd.read_file(
    "national_data.gdb",
    layer="counties",
    bbox=(-122.5, 47.5, -122.0, 47.8),
)
```

## REST API Consumption

These third-party request examples illustrate provider integration patterns.
Provider APIs, credentials, quotas, dataset identifiers, and availability must
be checked for the application; these snippets do not establish a maintained
GEO-INFER client for every named provider.

### NOAA Weather and Climate Data

```python
import httpx
from datetime import datetime, timedelta

NOAA_BASE_URL = "https://www.ncdc.noaa.gov/cdo-web/api/v2"
NOAA_TOKEN = "your_noaa_api_token"

async def fetch_noaa_observations(
    station_id: str,
    dataset_id: str = "GHCND",
    days_back: int = 30,
) -> dict:
    """Fetch daily climate observations from NOAA CDO."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{NOAA_BASE_URL}/data",
            headers={"token": NOAA_TOKEN},
            params={
                "datasetid": dataset_id,
                "stationid": station_id,
                "startdate": start_date.strftime("%Y-%m-%d"),
                "enddate": end_date.strftime("%Y-%m-%d"),
                "limit": 1000,
                "units": "metric",
            },
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()
```

### USGS Earthquake and Water Data

```python
async def fetch_usgs_earthquakes(
    min_magnitude: float = 4.0,
    days_back: int = 7,
) -> dict:
    """Fetch recent earthquake data from USGS."""
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days_back)

    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://earthquake.usgs.gov/fdsnws/event/1/query",
            params={
                "format": "geojson",
                "starttime": start_time.isoformat(),
                "endtime": end_time.isoformat(),
                "minmagnitude": min_magnitude,
            },
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()
```

### Copernicus Climate Data Store

```python
import cdsapi

def download_copernicus_era5(
    variable: str,
    year: str,
    month: str,
    output_path: str,
) -> str:
    """Download ERA5 reanalysis data from Copernicus CDS."""
    client = cdsapi.Client()
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": variable,
            "year": year,
            "month": month,
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": ["00:00", "06:00", "12:00", "18:00"],
            "format": "netcdf",
        },
        output_path,
    )
    return output_path
```

## Apache Kafka for Streaming Data

DATA's `geo_infer_data.connectors.stream.KafkaConnector` currently raises
`NotImplementedError` for connect/send/receive because no Kafka client backend
is implemented or declared. IOT does not ship the Kafka consumer shown below.
The following is a separate third-party application pattern requiring a Kafka
client and broker; it is not an executable use of the DATA connector or evidence
of a running ingestion service.

```python
from confluent_kafka import Consumer, Producer
import json

def create_geo_consumer(
    broker: str,
    group_id: str,
    topic: str,
) -> Consumer:
    """Create a Kafka consumer for geospatial event streams."""
    consumer = Consumer({
        "bootstrap.servers": broker,
        "group.id": group_id,
        "auto.offset.reset": "latest",
        "enable.auto.commit": True,
    })
    consumer.subscribe([topic])
    return consumer

def consume_spatial_events(consumer: Consumer, batch_size: int = 100) -> list[dict]:
    """Read a batch of spatial events from Kafka."""
    events = []
    while len(events) < batch_size:
        msg = consumer.poll(timeout=1.0)
        if msg is None:
            break
        if msg.error():
            continue
        event = json.loads(msg.value().decode("utf-8"))
        events.append(event)
    return events

def produce_analysis_result(
    producer: Producer,
    topic: str,
    result: dict,
) -> None:
    """Write an analysis result back to Kafka."""
    producer.produce(
        topic,
        value=json.dumps(result).encode("utf-8"),
    )
    producer.flush()
```

## Cloud Storage

Cloud support is adapter-specific. DATA has explicit optional integrations;
that does not establish a unified implemented backend for every cloud provider.
The examples below use third-party storage clients or GDAL virtual filesystems
and require the selected dependencies, drivers, credentials, and service access.

### Amazon S3

```python
import rasterio
import boto3

# Read a raster directly from S3 using GDAL's /vsis3/ driver
with rasterio.open("s3://geo-infer-data/rasters/landcover_2025.tif") as src:
    data = src.read(1)
    bounds = src.bounds

# Upload processed results
s3 = boto3.client("s3")
s3.upload_file(
    "local_output.tif",
    "geo-infer-data",
    "results/processed_output.tif",
)
```

### Google Cloud Storage

```python
from google.cloud import storage

gcs_client = storage.Client()
bucket = gcs_client.bucket("geo-infer-data")

# Download a GeoJSON dataset
blob = bucket.blob("vectors/boundaries.geojson")
blob.download_to_filename("/tmp/boundaries.geojson")

# Read raster via GDAL virtual filesystem
with rasterio.open("/vsigs/geo-infer-data/rasters/ndvi.tif") as src:
    ndvi = src.read(1)
```

### Azure Blob Storage

```python
from azure.storage.blob import BlobServiceClient

blob_service = BlobServiceClient.from_connection_string(
    "DefaultEndpointsProtocol=https;AccountName=geoinfer;..."
)
container = blob_service.get_container_client("spatial-data")

# Download a file
blob = container.get_blob_client("vectors/parcels.gpkg")
with open("/tmp/parcels.gpkg", "wb") as f:
    stream = blob.download_blob()
    f.write(stream.readall())

# Read via GDAL virtual filesystem
with rasterio.open("/vsiaz/spatial-data/rasters/dem.tif") as src:
    elevation = src.read(1)
```

## OGC Client Patterns

The examples below use OWSLib against caller-provisioned OGC services. They do
not show a GEO-INFER WMS/WFS/WCS server or establish formal OGC conformance.
Inspect each service's capabilities, supported version, CRS axis order, layer
identifiers, and response format before executing a request.

### WMS (Web Map Service)

```python
from owslib.wms import WebMapService

wms = WebMapService("https://maps.example.com/geoserver/wms", version="1.3.0")

# List available layers
for layer_name, layer in wms.contents.items():
    print(f"{layer_name}: {layer.title} ({layer.boundingBoxWGS84})")

# Fetch a map image
response = wms.getmap(
    layers=["landcover"],
    srs="EPSG:4326",
    bbox=(-122.5, 47.5, -122.0, 47.8),
    size=(1024, 768),
    format="image/png",
)
with open("map_tile.png", "wb") as f:
    f.write(response.read())
```

### WFS (Web Feature Service)

```python
from owslib.wfs import WebFeatureService
import geopandas as gpd
from io import BytesIO

wfs = WebFeatureService("https://maps.example.com/geoserver/wfs", version="2.0.0")

response = wfs.getfeature(
    typename=["geo_infer:watersheds"],
    bbox=(-122.5, 47.5, -122.0, 47.8),
    outputFormat="application/json",
)
gdf = gpd.read_file(BytesIO(response.read()))
```

### WCS (Web Coverage Service)

```python
from owslib.wcs import WebCoverageService

wcs = WebCoverageService("https://maps.example.com/geoserver/wcs", version="2.0.1")

response = wcs.getCoverage(
    identifier=["elevation_dem"],
    bbox=(-122.5, 47.5, -122.0, 47.8),
    crs="EPSG:4326",
    format="image/tiff",
    width=1024,
    height=768,
)
with open("elevation_subset.tif", "wb") as f:
    f.write(response.read())
```

## GeoLibre & optional integrations

GEO-INFER includes a GeoLibre project-file writer and selected optional engine
adapters. Exporting a file and importing an optional dependency do not establish
viewer execution, native engine availability, or a performance gain.

### GeoLibre project emission

The writer emits schema-versioned `.geolibre.json` project data with format
version `0.1.0`. It does not launch or vendor a GeoLibre viewer. Retain the
selected viewer version and actual load/render evidence when qualifying a
consumer of that format.

```python
from geo_infer_space import polygon_to_cells
from geo_infer_space.core import (
    SpatialIndexingInterface, build_h3_grid_project, write_project,
)

cells = polygon_to_cells(SAMPLE_POLYGON, resolution=7)
indexer = SpatialIndexingInterface()
features = [{"type": "Feature",
             "properties": {"h3": c},
             "geometry": {"type": "Polygon",
                          "coordinates": [[[lng, lat] for lat, lng in indexer.get_cell_boundary(c)]]}}
            for c in cells]
project = build_h3_grid_project("H3 grid", {"type": "FeatureCollection", "features": features})
write_project(project, "./h3_grid.geolibre.json")
```

The integration emits project data and does not copy viewer JavaScript into
this repository. The repository license and any separately selected component
license remain in their respective source distributions.

Runnable example:
`GEO-INFER-EXAMPLES/examples/getting_started/geolibre_export/scripts/run_example.py`.

### Optional engines and explicit availability

- **DuckDB-Spatial** (`geo_infer_data.utils.duckdb_spatial`): fast
  GeoParquet/FlatGeobuf/Shapefile reads when DuckDB+Spatial is installed;
  otherwise `read_cloud_native_vector` transparently uses GeoPandas/Fiona.
  `HAS_DUCKDB` indicates the Python dependency, not installed native extensions.
  Use `require_duckdb=True` when fallback is unacceptable and provision the
  required native extensions explicitly; see the [migration contract](../releases/0.4.0_migration.md#optional-integrations-and-persisted-algorithms).
- **WhiteboxTools** (`geo_infer_space.core.whitebox_bridge`): terrain/hydrology
  helpers such as `flow_accumulation` for the WATER/FOREST/EMERGENCY modules.
  Gate: `HAS_WHITEBOX`; install `whitebox-workflows` to enable.
- **LLM policy helper** (`geo_infer_agent.core.llm_proxy`): validates model
  allowlists, size caps, and per-client rate limits. The helper does not itself
  start a proxy server, provide credentials, call a model provider, or qualify
  an OpenAI-compatible endpoint.

## Authentication Patterns

### API Key Authentication

Most external data providers (NOAA, Copernicus, Planet) use API keys passed as headers or query parameters.

```python
import os

# Store API keys in environment variables, never in code
NOAA_TOKEN = os.environ["NOAA_API_TOKEN"]
PLANET_API_KEY = os.environ["PLANET_API_KEY"]
COPERNICUS_UID = os.environ["COPERNICUS_UID"]
COPERNICUS_KEY = os.environ["COPERNICUS_API_KEY"]
```

### OAuth2 for Service-to-Service

For services requiring OAuth2 (Sentinel Hub, some STAC catalogs):

```python
import httpx

async def get_oauth2_token(
    token_url: str,
    client_id: str,
    client_secret: str,
) -> str:
    """Obtain an OAuth2 bearer token using client credentials grant."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
            },
        )
        response.raise_for_status()
        return response.json()["access_token"]
```

## Troubleshooting

### Connection Timeouts

External API calls can fail due to network issues or rate limits. Use retry logic with exponential backoff.

```python
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
)
async def fetch_with_retry(url: str, params: dict) -> dict:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()
```

### PostGIS Connection Failures

Common causes and fixes:

| Symptom | Cause | Fix |
|---------|-------|-----|
| `connection refused` | PostgreSQL not running or wrong port | Verify `pg_isready -h host -p port` |
| `password authentication failed` | Wrong credentials | Check `PGPASSWORD` or `.pgpass` file |
| `database does not exist` | Missing database | Run `CREATE DATABASE geo_infer_db` |
| `type "geometry" does not exist` | PostGIS extension missing | Run `CREATE EXTENSION postgis` |
| `too many clients` | Connection pool exhausted | Use PgBouncer or increase `max_connections` |

### GDAL Driver Issues

If GDAL cannot open a format, verify the driver is compiled in:

```python
from osgeo import gdal

# List all available raster drivers
for i in range(gdal.GetDriverCount()):
    driver = gdal.GetDriver(i)
    print(f"{driver.ShortName}: {driver.LongName}")
```

### Cloud Storage Access Denied

Ensure credentials are configured correctly:

```bash
# AWS: verify credentials
aws sts get-caller-identity

# GCS: verify credentials
gcloud auth application-default print-access-token

# Azure: verify credentials
az account show
```

For GDAL virtual filesystem access, set the relevant environment variables:

```bash
# AWS
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_DEFAULT_REGION="us-west-2"

# GCS
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"

# Azure
export AZURE_STORAGE_CONNECTION_STRING="..."
```
