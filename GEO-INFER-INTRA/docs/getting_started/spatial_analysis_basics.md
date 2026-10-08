# Spatial Analysis Basics

> **Tutorial**: Core spatial concepts for working with GEO-INFER
>
> This guide covers the spatial data types, operations, and tools you need to
> perform geospatial analysis within the GEO-INFER framework.

## What Spatial Analysis Means in GEO-INFER

Spatial analysis in GEO-INFER is the process of examining geographic data --
locations, boundaries, distances, and spatial relationships -- to extract
patterns and inform decisions. The framework treats spatial data as a
first-class input to Active Inference models: sensor readings, land cover
classifications, and administrative boundaries all feed into generative models
that minimize free energy over geographic state spaces.

The primary module for spatial work is **GEO-INFER-SPACE**. It provides a
backend-agnostic API for spatial indexing (H3 hexagonal grids), geometric
operations (buffering, centroids, area calculations), and spatial analytics
(hotspot detection, clustering). Other modules -- PLACE, TIME, ACT -- consume
GEO-INFER-SPACE outputs as inputs to higher-level analyses.

## Spatial Data Types

### Vector Data

Vector data represents geographic features as discrete geometric objects.

| Type | Description | Example |
|------|-------------|---------|
| **Point** | A single (x, y) coordinate | Weather station location |
| **LineString** | An ordered sequence of points | Road segment, river reach |
| **Polygon** | A closed ring of points defining an area | County boundary, lake outline |
| **MultiPolygon** | A collection of polygons | A state with islands |

In Python, vector data is typically handled with `geopandas.GeoDataFrame`
objects backed by `shapely` geometries:

```python
import geopandas as gpd
from shapely.geometry import Point, Polygon

# Create a GeoDataFrame of sensor locations
sensors = gpd.GeoDataFrame({
    "sensor_id": ["s01", "s02", "s03"],
    "temperature": [22.1, 19.8, 24.5],
    "geometry": [
        Point(-122.4194, 37.7749),  # San Francisco
        Point(-122.2727, 37.8716),  # Berkeley
        Point(-122.0322, 37.3230),  # San Jose
    ]
}, crs="EPSG:4326")

print(sensors)
```

### Raster Data

Raster data represents continuous surfaces as grids of cells (pixels). Each
cell holds a numeric value -- elevation, reflectance, temperature. Satellite
imagery and digital elevation models (DEMs) are raster data.

GEO-INFER modules consume rasters primarily through `numpy` arrays or
`xarray.DataArray` objects. GEO-INFER-SPACE does not process rasters directly;
raster-to-vector conversion (zonal statistics, contouring) happens upstream in
GEO-INFER-DATA or domain modules like GEO-INFER-CLIMATE.

### Point Clouds

Point clouds are unstructured sets of 3D coordinates, often from LiDAR. They
are relevant in GEO-INFER-FOREST (canopy height models) and GEO-INFER-SPACE
(terrain analysis). Point clouds are typically stored as LAS/LAZ files and
processed with `laspy` or `pdal` before entering the GEO-INFER pipeline.

### H3 Hexagonal Grids

H3 is Uber's hierarchical hexagonal spatial indexing system. GEO-INFER-SPACE
uses H3 v4 as its default spatial index. Hexagons tile a sphere with uniform
neighbor distances, which avoids the distortion problems of square grids at
high latitudes.

Key H3 concepts:

- **Resolution**: 0 (coarsest, ~4,250,000 km^2 per cell) to 15 (finest, ~0.9 m^2 per cell). Resolution 8 (~0.74 km^2) is common for city-scale analysis.
- **Cell ID**: A 64-bit integer encoded as a hex string (e.g., `"8928308280fffff"`).
- **Hierarchy**: Every cell at resolution `r` has 7 children at resolution `r+1`.

```python
from geo_infer_space import latlng_to_cell, cell_to_latlng
from geo_infer_space.utils.h3_utils import grid_disk

# Convert a lat/lng to an H3 cell at resolution 8
cell = latlng_to_cell(37.7749, -122.4194, 8)
print(f"H3 cell: {cell}")

# Convert back to lat/lng (cell centroid)
lat, lng = cell_to_latlng(cell)
print(f"Centroid: ({lat:.4f}, {lng:.4f})")

# Get the cell and its immediate neighbors (k=1 ring)
neighbors = grid_disk(cell, k=1)
print(f"Cell + neighbors: {len(neighbors)} cells")
```

**H3 v4 API note**: GEO-INFER uses the H3 v4 Python bindings. The function
names are `latlng_to_cell` and `cell_to_latlng`. All GEO-INFER documentation and code follows the
v4 convention.

## Core Spatial Operations

### Buffering

A buffer creates a new polygon at a specified distance from a geometry. Use
buffers to define impact zones, proximity areas, or safety perimeters.

```python
from geo_infer_space import GeometricOperationsInterface

geo_ops = GeometricOperationsInterface()

# Buffer a point geometry by 500 meters
point_geojson = {"type": "Point", "coordinates": [-122.4194, 37.7749]}
buffer_500m = geo_ops.buffer_geometry(point_geojson, distance=500.0)
```

With `geopandas`, buffering works directly on GeoDataFrame columns. Note that
buffering in geographic CRS (EPSG:4326) uses degrees, not meters. Project to a
metric CRS first for meter-based buffers:

```python
# Project to UTM Zone 10N (meters) before buffering
sensors_utm = sensors.to_crs("EPSG:32610")
sensors_utm["buffer_500m"] = sensors_utm.geometry.buffer(500)
```

### Spatial Joins

A spatial join links two datasets based on their geographic relationship. The
most common predicates are `intersects`, `within`, and `contains`.

```python
import geopandas as gpd

# Load county boundaries and sensor locations
counties = gpd.read_file("counties.geojson")
sensors = gpd.read_file("sensors.geojson")

# Find which county each sensor falls in
sensors_with_county = gpd.sjoin(
    sensors, counties, how="left", predicate="within"
)
print(sensors_with_county[["sensor_id", "county_name"]].head())
```

### Nearest Neighbor

Finding the closest feature to each point is a common operation for site
selection, nearest-facility analysis, and interpolation.

```python
from shapely.ops import nearest_points

# Find the nearest county centroid to each sensor
county_centroids = counties.geometry.centroid
for idx, sensor in sensors.iterrows():
    nearest = county_centroids.distance(sensor.geometry).idxmin()
    dist = county_centroids.iloc[nearest].distance(sensor.geometry)
    print(f"Sensor {sensor['sensor_id']}: nearest county index={nearest}, "
          f"distance={dist:.4f} degrees")
```

### Containment and Intersection

Test whether geometries overlap, contain, or touch each other:

```python
# Which sensors fall within a specific polygon?
study_area = Polygon([
    (-122.5, 37.7), (-122.3, 37.7),
    (-122.3, 37.85), (-122.5, 37.85)
])

mask = sensors.within(study_area)
sensors_in_area = sensors[mask]
print(f"{len(sensors_in_area)} sensors inside the study area")
```

### Spatial Clustering with GEO-INFER-SPACE

The `SpatialAnalyticsInterface` provides clustering and hotspot analysis:

```python
import numpy as np
from geo_infer_space import SpatialAnalyticsInterface

analytics = SpatialAnalyticsInterface()

# Cluster spatial data points
coordinates = np.array([
    [37.7749, -122.4194],
    [37.8716, -122.2727],
    [37.3230, -122.0322],
    [37.5585, -122.2711],
    [37.4419, -122.1430],
])

clusters = analytics.analyze_clusters(
    data=coordinates,
    method="kmeans",
    n_clusters=2
)
print(f"Cluster labels: {clusters}")
```

## Coordinate Reference Systems (CRS)

Every spatial dataset has a CRS that defines how coordinates map to locations
on Earth. Mismatched CRS is the most common source of spatial analysis bugs.

| CRS | EPSG Code | Units | Use Case |
|-----|-----------|-------|----------|
| WGS 84 | 4326 | Degrees | GPS, web maps, data interchange |
| Web Mercator | 3857 | Meters | Web tile maps (Leaflet, Mapbox) |
| UTM Zone 10N | 32610 | Meters | US West Coast local analysis |
| NAD83 / Conus Albers | 5070 | Meters | US-wide equal-area analysis |

Rules of thumb:

1. **Store** data in EPSG:4326 (WGS 84). It is the universal interchange format.
2. **Analyze** in a projected CRS appropriate for your study area when you need metric distances or areas.
3. **Display** in EPSG:3857 if feeding a web map, or EPSG:4326 for GeoJSON export.

```python
# Check CRS
print(f"Current CRS: {sensors.crs}")

# Reproject to UTM for metric analysis
sensors_metric = sensors.to_crs("EPSG:32610")

# Calculate pairwise distances in meters
from shapely.ops import nearest_points
dist_meters = sensors_metric.geometry.iloc[0].distance(
    sensors_metric.geometry.iloc[1]
)
print(f"Distance between sensors s01 and s02: {dist_meters:.0f} meters")
```

## GEO-INFER-SPACE Module Architecture

GEO-INFER-SPACE uses a dispatcher/interface pattern that routes operations to
the appropriate backend:

```
User Code
    |
    v
SpatialIndexingInterface / GeometricOperationsInterface / SpatialAnalyticsInterface
    |
    v
BackendDispatcher
    |
    +---> UnifiedH3Backend (default)
    +---> SRAI Backend (optional)
    +---> Custom Backend (extensible)
```

The three main interfaces are:

- **`SpatialIndexingInterface`**: Cell indexing, neighbor lookups, polygon-to-cell conversion. Wraps H3 v4 operations.
- **`GeometricOperationsInterface`**: Buffer, area, perimeter, centroid, intersection, union. Delegates to shapely or H3 depending on geometry type.
- **`SpatialAnalyticsInterface`**: Hotspot analysis, clustering, spatial context analysis. Uses numpy/scipy internally.

All three are importable directly from `geo_infer_space`:

```python
from geo_infer_space import (
    SpatialIndexingInterface,
    GeometricOperationsInterface,
    SpatialAnalyticsInterface,
)

indexer = SpatialIndexingInterface()
geo_ops = GeometricOperationsInterface()
analytics = SpatialAnalyticsInterface()
```

## Putting It Together: A Minimal Spatial Workflow

This example loads point data, indexes it with H3, buffers a study area, and
runs a hotspot analysis:

```python
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from geo_infer_space import (
    SpatialIndexingInterface,
    SpatialAnalyticsInterface,
    latlng_to_cell,
)

# 1. Create sample point data
np.random.seed(42)
n = 100
lats = np.random.uniform(37.3, 37.9, n)
lngs = np.random.uniform(-122.6, -122.0, n)
values = np.random.exponential(scale=5.0, size=n)

points = gpd.GeoDataFrame({
    "value": values,
    "geometry": [Point(lng, lat) for lng, lat in zip(lngs, lats)]
}, crs="EPSG:4326")

# 2. Assign each point to an H3 cell at resolution 7
points["h3_cell"] = [
    latlng_to_cell(geom.y, geom.x, 7)
    for geom in points.geometry
]

# 3. Aggregate values by H3 cell
cell_stats = points.groupby("h3_cell")["value"].agg(["mean", "count"])
print(f"Unique H3 cells: {len(cell_stats)}")
print(cell_stats.head())

# 4. Run hotspot analysis
analytics = SpatialAnalyticsInterface()
hotspot_data = {
    "cells": cell_stats.index.tolist(),
    "values": cell_stats["mean"].tolist(),
}
hotspots = analytics.analyze_hotspots(hotspot_data)
print(f"Hotspot results: {hotspots}")
```

## Next Steps

- **[Your First Map](first_map.md)** -- Create an interactive map visualization with GEO-INFER-SPACE.
- **[Active Inference Basics](active_inference_basics.md)** -- Learn how spatial data feeds into Active Inference models.
- **[Coordinate Systems](../geospatial/concepts/coordinate_systems.md)** -- Deep dive into CRS transformations.
- **[Spatial Relationships](../geospatial/concepts/spatial_relationships.md)** -- Topological operations in detail.
- **[H3 Geospatial Indexing](../geospatial/data_formats/h3/index.md)** -- Full H3 reference for GEO-INFER.

---

**License**: GEO-INFER is released under the [CC BY-NC-SA 4.0 License](https://creativecommons.org/licenses/by-nc-sa/4.0/).
