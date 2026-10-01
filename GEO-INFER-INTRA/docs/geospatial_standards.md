# GEO-INFER Geospatial Standards

This document defines the geospatial data standards, coordinate conventions,
file formats, and spatial operations used across all GEO-INFER modules. All
modules must conform to these standards.

## H3 v4 API Reference

GEO-INFER uses **H3 version 4** (`h3>=4.5.0,<5`) exclusively. The legacy v3 API
(geo_to_h3, h3_to_geo, h3_to_geo_boundary, etc.) must not be used anywhere
in the codebase.

### Core Indexing Functions

| Function | Parameters | Return Type | Description |
|----------|-----------|-------------|-------------|
| `latlng_to_cell(lat, lng, res)` | float, float, int | str | Convert a latitude/longitude pair to the H3 cell containing it |
| `cell_to_latlng(cell)` | str | tuple[float, float] | Get the center of a cell as (lat, lng) |
| `cell_to_boundary(cell)` | str | tuple[tuple[float, float], ...] | Get cell boundary as sequence of (lat, lng) vertex pairs |
| `get_resolution(cell)` | str | int | Get the resolution of an H3 cell index |

```python
import h3

# Index a point
cell = h3.latlng_to_cell(45.5231, -122.6765, 9)
print(f"Cell: {cell}")  # e.g., "8928308280fffff"

# Get cell center
lat, lng = h3.cell_to_latlng(cell)
print(f"Center: ({lat:.4f}, {lng:.4f})")

# Get cell boundary (for polygon creation)
boundary = h3.cell_to_boundary(cell)
print(f"Vertices: {len(boundary)}")  # 6 for hexagons

# Check resolution
res = h3.get_resolution(cell)
print(f"Resolution: {res}")  # 9
```

### Grid Traversal Functions

| Function | Parameters | Return Type | Description |
|----------|-----------|-------------|-------------|
| `grid_disk(origin, k)` | str, int | frozenset[str] | All cells within k grid steps of origin (inclusive) |
| `grid_ring(origin, k)` | str, int | frozenset[str] | Cells at exactly k grid steps from origin |
| `grid_distance(a, b)` | str, str | int | Minimum grid steps between two cells |
| `grid_path_cells(a, b)` | str, str | list[str] | Ordered list of cells forming shortest path |
| `are_neighbor_cells(a, b)` | str, str | bool | Whether two cells share an edge |

```python
import h3

origin = h3.latlng_to_cell(45.5231, -122.6765, 9)

# Get 2-ring neighborhood (origin + 1 ring + 2 ring)
disk = h3.grid_disk(origin, 2)
print(f"2-disk: {len(disk)} cells")  # 1 + 6 + 12 = 19

# Get only the outer ring
ring = h3.grid_ring(origin, 2)
print(f"2-ring: {len(ring)} cells")  # 12

# Distance between cells
neighbor = list(h3.grid_disk(origin, 1) - {origin})[0]
dist = h3.grid_distance(origin, neighbor)
print(f"Distance to neighbor: {dist}")  # 1

# Check adjacency
is_neighbor = h3.are_neighbor_cells(origin, neighbor)
print(f"Are neighbors: {is_neighbor}")  # True
```

### Hierarchy Functions

| Function | Parameters | Return Type | Description |
|----------|-----------|-------------|-------------|
| `cell_to_parent(cell, res)` | str, int | str | Get parent cell at a coarser resolution |
| `cell_to_children(cell, res)` | str, int | frozenset[str] | Get child cells at a finer resolution |
| `cell_to_center_child(cell, res)` | str, int | str | Get the center child at finer resolution |
| `compact_cells(cells)` | iterable[str] | frozenset[str] | Compact a set of cells to mixed resolutions |
| `uncompact_cells(cells, res)` | iterable[str], int | frozenset[str] | Expand compacted cells to uniform resolution |

```python
import h3

cell = h3.latlng_to_cell(45.5231, -122.6765, 9)

# Move up the hierarchy
parent = h3.cell_to_parent(cell, 7)
print(f"Parent (res 7): {parent}")

# Move down the hierarchy
children = h3.cell_to_children(cell, 10)
print(f"Children (res 10): {len(children)} cells")  # 7

# Compact and uncompact
area_cells = h3.grid_disk(cell, 3)
compacted = h3.compact_cells(area_cells)
print(f"Original: {len(area_cells)}, Compacted: {len(compacted)}")
```

### Measurement Functions

| Function | Parameters | Return Type | Description |
|----------|-----------|-------------|-------------|
| `average_hexagon_area(res, unit)` | int, str | float | Average cell area at resolution |
| `average_hexagon_edge_length(res, unit)` | int, str | float | Average edge length at resolution |
| `cell_area(cell, unit)` | str, str | float | Area of a specific cell |
| `edge_length(edge, unit)` | str, str | float | Length of a specific edge |

Units: `"km^2"`, `"m^2"`, `"rads^2"` for area; `"km"`, `"m"`, `"rads"` for length.

```python
import h3

# Average cell sizes at common resolutions
for res in [6, 8, 9, 10]:
    area = h3.average_hexagon_area(res, "km^2")
    edge = h3.average_hexagon_edge_length(res, "km")
    print(f"Resolution {res:2d}: area={area:.4f} km^2, edge={edge:.4f} km")
```

### Geometry Conversion

| Function | Parameters | Return Type | Description |
|----------|-----------|-------------|-------------|
| `cells_to_geo(cells)` | set[str] | dict | Convert cells to GeoJSON geometry |
| `geo_to_cells(geojson, res)` | dict, int | frozenset[str] | Fill a GeoJSON polygon with cells |

```python
import h3
import json

# Convert a set of cells to a GeoJSON polygon
cells = h3.grid_disk(h3.latlng_to_cell(45.5231, -122.6765, 8), 2)
geojson = h3.cells_to_geo(cells)
print(f"GeoJSON type: {geojson['type']}")

# Fill a polygon with H3 cells
polygon = {
    "type": "Polygon",
    "coordinates": [[
        [-122.70, 45.50], [-122.65, 45.50],
        [-122.65, 45.55], [-122.70, 45.55],
        [-122.70, 45.50]
    ]]
}
filled_cells = h3.geo_to_cells(polygon, 9)
print(f"Cells filling polygon: {len(filled_cells)}")
```

## OGC Standards

### WGS84 (EPSG:4326)

The World Geodetic System 1984 is the default CRS for all GEO-INFER data:
- Coordinates are geographic (latitude, longitude) in decimal degrees.
- Latitude range: -90.0 to 90.0.
- Longitude range: -180.0 to 180.0.
- Used for storage, exchange, and cross-module communication.

### OGC Simple Features

GEO-INFER geometry types follow the OGC Simple Features specification:

| Type | Description | Example |
|------|-------------|---------|
| `Point` | Single coordinate pair | Sensor location, city center |
| `LineString` | Ordered sequence of points | Road segment, river reach |
| `Polygon` | Closed ring with optional holes | Building footprint, watershed |
| `MultiPoint` | Collection of points | Sensor network |
| `MultiLineString` | Collection of line strings | Road network |
| `MultiPolygon` | Collection of polygons | Archipelago, fragmented habitat |
| `GeometryCollection` | Mixed geometry types | Use sparingly |

### OGC Web Services

| Service | Protocol | Purpose | GEO-INFER Usage |
|---------|----------|---------|-----------------|
| WMS | Web Map Service | Raster map image tiles | Display basemaps |
| WFS | Web Feature Service | Vector feature queries | Retrieve spatial features |
| WCS | Web Coverage Service | Raster data access | Download gridded datasets |
| WMTS | Web Map Tile Service | Pre-rendered map tiles | High-performance display |

## CRS Conventions

### EPSG:4326 -- Geographic (Default)

- Coordinate system: latitude (degrees north), longitude (degrees east).
- Use for: data storage, cross-module transfer, H3 indexing, GeoJSON.
- Do not use for: distance calculations, area calculations, display.

### EPSG:3857 -- Web Mercator

- Coordinate system: easting (meters), northing (meters).
- Use for: web map display, tile rendering with Leaflet/Mapbox.
- Do not use for: area calculations (severe distortion at high latitudes),
  distance calculations, data storage.

### When to Use Each

| Operation | CRS | Reason |
|-----------|-----|--------|
| Store data | EPSG:4326 | Universal, interoperable |
| H3 indexing | EPSG:4326 | H3 API takes lat/lng in degrees |
| Web map display | EPSG:3857 | Standard for web mapping |
| Distance calculation | UTM zone or local projection | Preserves distances locally |
| Area calculation | Equal-area projection | Preserves area |
| Buffer (meters) | Local UTM zone | Buffer distance in metric units |

### CRS Conversion

```python
import geopandas as gpd
from shapely.geometry import Point
import pyproj

# Create data in WGS84
gdf = gpd.GeoDataFrame(
    {"name": ["Portland"]},
    geometry=[Point(-122.6765, 45.5231)],
    crs="EPSG:4326"
)

# Convert to Web Mercator for display
gdf_display = gdf.to_crs("EPSG:3857")

# Convert to UTM Zone 10N for distance/area calculations
gdf_utm = gdf.to_crs("EPSG:32610")
print(f"UTM easting: {gdf_utm.geometry.x.iloc[0]:.1f} m")
print(f"UTM northing: {gdf_utm.geometry.y.iloc[0]:.1f} m")

# Using pyproj transformer for individual coordinates
transformer = pyproj.Transformer.from_crs(
    "EPSG:4326", "EPSG:32610", always_xy=True
)
easting, northing = transformer.transform(-122.6765, 45.5231)
print(f"Transformed: ({easting:.1f}, {northing:.1f})")
```

## Coordinate Order

This is the single most common source of bugs in geospatial code. GEO-INFER
follows these conventions:

| Context | Order | Example |
|---------|-------|---------|
| GeoJSON coordinates | (longitude, latitude) | `[-122.6765, 45.5231]` |
| H3 API | (latitude, longitude) | `h3.latlng_to_cell(45.5231, -122.6765, 9)` |
| Shapely Point | (x, y) = (longitude, latitude) | `Point(-122.6765, 45.5231)` |
| GEO-INFER functions | (latitude, longitude) | Document explicitly in docstring |
| pyproj (always_xy=True) | (longitude, latitude) | `transform(-122.6765, 45.5231)` |

**Rule**: always document the coordinate order in every function's docstring.
When a function accepts coordinates, name parameters explicitly:

```python
# CORRECT: explicit parameter names
def analyze_point(lat: float, lng: float, resolution: int = 9):
    """Analyze the area around a geographic point.

    Args:
        lat: Latitude in decimal degrees (WGS84, range: -90 to 90).
        lng: Longitude in decimal degrees (WGS84, range: -180 to 180).
        resolution: H3 resolution for spatial indexing.
    """
    cell = h3.latlng_to_cell(lat, lng, resolution)
    # GeoJSON uses (lng, lat) order
    geojson_point = {"type": "Point", "coordinates": [lng, lat]}
    # Shapely uses (x=lng, y=lat)
    shapely_point = Point(lng, lat)
    return cell, geojson_point, shapely_point
```

## GeoJSON Format

### Specification Summary

GeoJSON (RFC 7946) encodes geographic features as JSON objects.

```json
{
  "type": "Feature",
  "geometry": {
    "type": "Point",
    "coordinates": [-122.6765, 45.5231]
  },
  "properties": {
    "name": "Portland",
    "h3_index": "8928308280fffff",
    "value": 42.0
  }
}
```

### Rules for GEO-INFER

- Coordinates are always (longitude, latitude) per RFC 7946.
- The `crs` member is not part of RFC 7946. GEO-INFER assumes EPSG:4326 for
  all GeoJSON. Do not embed CRS in GeoJSON.
- FeatureCollection is the standard top-level type for data exchange.
- Keep GeoJSON files under 100 MB. For larger datasets, use GeoParquet.

### Creating GeoJSON from GeoDataFrame

```python
import geopandas as gpd
import json

# Ensure WGS84 before exporting to GeoJSON
gdf_4326 = gdf.to_crs("EPSG:4326")
geojson_str = gdf_4326.to_json()
geojson_dict = json.loads(geojson_str)

# Write to file
gdf_4326.to_file("output.geojson", driver="GeoJSON")
```

## GeoParquet

GeoParquet is the preferred format for large geospatial datasets. It provides
columnar compression, predicate pushdown, and standardized geometry metadata.

### Advantages Over GeoJSON

| Property | GeoJSON | GeoParquet |
|----------|---------|------------|
| File size | Large (text) | Small (binary, compressed) |
| Read speed | Slow (full parse) | Fast (columnar access) |
| Schema | Implicit | Explicit |
| Geometry metadata | None | Standardized |
| Partial reads | No | Yes (row groups) |

### Reading and Writing

```python
import geopandas as gpd

# Write GeoParquet
gdf.to_parquet("data.parquet", engine="pyarrow")

# Read GeoParquet
gdf = gpd.read_parquet("data.parquet")

# Read with column selection (only loads requested columns)
gdf_subset = gpd.read_parquet(
    "data.parquet",
    columns=["geometry", "value", "h3_index"]
)
```

### Metadata Conventions

GeoParquet files store geometry metadata in the Parquet file footer under the
`geo` key. This metadata includes:
- Primary geometry column name
- Geometry types present
- CRS in PROJJSON format
- Bounding box of all geometries
- Encoding (WKB)

## Raster Formats

### GeoTIFF

The standard georeferenced raster format. GEO-INFER conventions:

- Use LZW or DEFLATE compression for all GeoTIFF outputs.
- Set the nodata value explicitly using the GeoTIFF nodata tag.
- Use internal tiling (256x256 or 512x512 blocks) for files over 100 MB.
- Band ordering: for multi-band imagery, document band assignments in metadata.

```python
import rasterio
import numpy as np

# Read a GeoTIFF
with rasterio.open("elevation.tif") as src:
    elevation = src.read(1)  # band 1
    transform = src.transform
    crs = src.crs
    nodata = src.nodata
    print(f"Shape: {elevation.shape}")
    print(f"CRS: {crs}")
    print(f"NoData: {nodata}")

# Write a GeoTIFF
with rasterio.open(
    "output.tif",
    "w",
    driver="GTiff",
    height=elevation.shape[0],
    width=elevation.shape[1],
    count=1,
    dtype=elevation.dtype,
    crs=crs,
    transform=transform,
    nodata=-9999,
    compress="lzw",
    tiled=True,
    blockxsize=256,
    blockysize=256,
) as dst:
    dst.write(elevation, 1)
```

### NetCDF

Used for multi-dimensional scientific data, particularly climate and weather.

- Dimension ordering: `(time, y, x)` for 3D spatiotemporal data.
- Coordinate variables must include `units` and `standard_name` attributes.
- Follow CF (Climate and Forecast) conventions for variable naming.

```python
import xarray as xr

# Read NetCDF
ds = xr.open_dataset("climate_data.nc")
print(ds.dims)       # e.g., {'time': 365, 'lat': 180, 'lon': 360}
print(ds.data_vars)  # e.g., {'temperature': ..., 'precipitation': ...}

# Select spatial subset
subset = ds.sel(lat=slice(44, 46), lon=slice(-123, -121))

# Resample temporal dimension
monthly_mean = ds.resample(time="ME").mean()
```

### Nodata Values

| Data Type | Standard Nodata Value |
|-----------|-----------------------|
| float32 | `np.nan` or `-9999.0` |
| float64 | `np.nan` or `-9999.0` |
| int16 | `-9999` |
| int32 | `-9999` |
| uint8 | `255` |
| uint16 | `65535` |

## Shapely Geometry Operations

GEO-INFER uses Shapely for vector geometry operations. These are the standard
operations used across modules:

### Construction

```python
from shapely.geometry import Point, LineString, Polygon, MultiPolygon, box

# Point (x=longitude, y=latitude)
pt = Point(-122.6765, 45.5231)

# LineString
line = LineString([(-122.70, 45.50), (-122.65, 45.55), (-122.60, 45.50)])

# Polygon (exterior ring, optional holes)
poly = Polygon([
    (-122.70, 45.50), (-122.65, 45.50),
    (-122.65, 45.55), (-122.70, 45.55),
    (-122.70, 45.50)  # close the ring
])

# Bounding box
bbox = box(-122.70, 45.50, -122.60, 45.55)  # (minx, miny, maxx, maxy)
```

### Spatial Predicates

```python
# Containment
print(poly.contains(pt))      # True if pt is inside poly
print(pt.within(poly))        # True if pt is inside poly (reverse)

# Intersection
print(poly.intersects(line))  # True if they share any space
print(poly.crosses(line))     # True if line crosses poly boundary

# Distance
dist = pt.distance(line)      # Euclidean distance in CRS units
```

### Spatial Operations

```python
# Buffer (in CRS units -- use projected CRS for metric buffers)
buffer_zone = pt.buffer(0.01)  # ~1.1 km at 45 degrees latitude

# Intersection
clipped = poly.intersection(buffer_zone)

# Union
merged = poly.union(buffer_zone)

# Difference
remaining = poly.difference(buffer_zone)

# Centroid
center = poly.centroid
print(f"Centroid: ({center.x:.4f}, {center.y:.4f})")

# Convex hull
hull = MultiPolygon([poly, buffer_zone]).convex_hull

# Simplify (Douglas-Peucker)
simplified = poly.simplify(tolerance=0.001, preserve_topology=True)
```

### Coordinate Extraction

```python
# From a polygon
exterior_coords = list(poly.exterior.coords)
print(f"Polygon vertices: {len(exterior_coords)}")

# From a point
x, y = pt.x, pt.y

# Bounds
minx, miny, maxx, maxy = poly.bounds
```

## Validation Utilities

### Coordinate Validation

```python
def validate_wgs84_coordinates(lat: float, lng: float) -> None:
    """Validate that coordinates are within WGS84 bounds.

    Args:
        lat: Latitude in decimal degrees.
        lng: Longitude in decimal degrees.

    Raises:
        ValueError: If coordinates are outside valid WGS84 range.
    """
    if not -90.0 <= lat <= 90.0:
        raise ValueError(
            f"Latitude {lat} outside valid range [-90, 90]"
        )
    if not -180.0 <= lng <= 180.0:
        raise ValueError(
            f"Longitude {lng} outside valid range [-180, 180]"
        )
```

### Geometry Validation

```python
from shapely.validation import make_valid

def validate_and_fix_geometry(geom):
    """Validate geometry and attempt to fix if invalid.

    Args:
        geom: Shapely geometry object.

    Returns:
        Valid geometry (possibly modified).
    """
    if geom is None or geom.is_empty:
        raise ValueError("Geometry is null or empty")

    if not geom.is_valid:
        fixed = make_valid(geom)
        return fixed

    return geom
```

## Related Documentation

- [Data Dictionary](data_dictionary.md) -- data structure definitions
- [Installation](installation.md) -- GDAL and H3 setup
- [Terminology](terminology.md) -- geospatial term definitions
- [Active Inference Guide](active_inference_guide.md) -- spatial Active Inference
