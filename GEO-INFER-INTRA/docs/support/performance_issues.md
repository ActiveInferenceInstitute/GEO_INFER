# Performance Issues

This guide covers diagnosing and resolving performance problems in GEO-INFER workloads. It addresses slow spatial queries, memory pressure, and common anti-patterns in geospatial Python code.

## Diagnosing Slow Spatial Queries

### Step 1: Identify the Bottleneck Type

Spatial workloads are constrained by one of three resources. Identify which one before optimizing.

```python
import time
import psutil
import os

def profile_run(func, *args):
    """Measure wall time, CPU time, and peak memory for a function call."""
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss

    cpu_start = time.process_time()
    wall_start = time.perf_counter()

    result = func(*args)

    wall_elapsed = time.perf_counter() - wall_start
    cpu_elapsed = time.process_time() - cpu_start
    mem_after = process.memory_info().rss

    print(f"Wall time:   {wall_elapsed:.3f}s")
    print(f"CPU time:    {cpu_elapsed:.3f}s")
    print(f"Memory delta: {(mem_after - mem_before) / 1e6:.1f} MB")
    print(f"CPU/Wall ratio: {cpu_elapsed / max(wall_elapsed, 1e-9):.2f}")

    return result
```

**Interpreting the CPU/Wall ratio:**

| Ratio | Bottleneck | Action |
|-------|-----------|--------|
| > 0.9 | CPU-bound | Optimize algorithm, use numba, reduce data |
| 0.3 - 0.9 | Mixed | Profile to find the split |
| < 0.3 | I/O-bound | Optimize data loading, use caching, pre-filter |

### Step 2: Profile the Hot Path

Use `cProfile` for function-level profiling:

```bash
python -m cProfile -s cumulative my_analysis.py 2>&1 | head -30
```

Use `line_profiler` for line-level profiling of the slow function:

```bash
uv run --with line_profiler kernprof -l -v my_analysis.py
```

## H3 Resolution vs Performance Tradeoffs

H3 resolution directly controls the number of cells and thus computation time and memory.

| Resolution | Hex Edge (km) | Cells Covering Earth | Typical Use |
|-----------|--------------|---------------------|-------------|
| 0 | 1107 | 122 | Global summary |
| 3 | 59 | 41,162 | Country-level |
| 5 | 8.5 | 2,016,842 | Regional |
| 7 | 1.2 | 98,825,162 | City-level |
| 9 | 0.17 | 4,842,432,842 | Neighborhood |
| 11 | 0.024 | ~237 billion | Building-level |

**Rule of thumb:** Each +2 resolution adds ~49x more cells. If resolution 7 takes 1 second, resolution 9 takes ~49 seconds, and resolution 11 takes ~2,400 seconds.

### Choosing the Right Resolution

```python
import h3

def recommend_resolution(n_points: int, target_cells: int = 100_000) -> int:
    """Suggest an H3 resolution that produces a manageable number of cells.

    Args:
        n_points: Number of data points.
        target_cells: Desired approximate number of unique cells.
    """
    for res in range(16):
        # Rough estimate: unique cells ~ min(n_points, total_cells_at_res)
        total_cells = 122 * (7 ** res)  # approximate
        expected_unique = min(n_points, total_cells)
        if expected_unique >= target_cells:
            return res
    return 15
```

## Memory Profiling for Spatial Data

### Tracking Memory Usage Over Time

```python
import tracemalloc

tracemalloc.start()

# ... your spatial analysis code ...

snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics("lineno")

print("Top 10 memory allocations:")
for stat in top_stats[:10]:
    print(f"  {stat}")
```

### Common Memory Hogs in Geospatial Code

| Operation | Memory Profile | Mitigation |
|-----------|---------------|------------|
| Loading GeoDataFrame | ~10x raw file size | Filter columns/rows before loading |
| Pairwise distance matrix | O(n^2) | Use spatial index, compute only needed pairs |
| GP kernel matrix | O(n^2) | Sparse approximations for n > 5000 |
| Raster to array | width * height * 8 bytes | Windowed reads, tiled processing |
| H3 cell polygons | ~200 bytes per cell | Compute on-demand, do not store all at once |

### Reducing GeoDataFrame Memory

```python
import geopandas as gpd

# Load only needed columns
gdf = gpd.read_file("large_dataset.gpkg", columns=["geometry", "value", "timestamp"])

# Filter spatially before loading (if supported)
gdf = gpd.read_file("large_dataset.gpkg", bbox=(-122.5, 37.5, -122.0, 38.0))

# Downcast numeric types
gdf["value"] = gdf["value"].astype("float32")
```

## Optimizing PostGIS Queries

### Use Spatial Indexes

Always ensure a spatial index exists:

```sql
-- Check if index exists
SELECT indexname FROM pg_indexes WHERE tablename = 'observations';

-- Create if missing
CREATE INDEX IF NOT EXISTS idx_obs_geom ON observations USING GIST (geom);
```

### Avoid ST_Distance for Filtering

```sql
-- Slow: computes distance for every row
SELECT * FROM observations WHERE ST_Distance(geom, ST_MakePoint(-122.4, 37.7)) < 1000;

-- Fast: use ST_DWithin which uses the spatial index
SELECT * FROM observations WHERE ST_DWithin(
    geom,
    ST_SetSRID(ST_MakePoint(-122.4, 37.7), 4326)::geography,
    1000  -- meters
);
```

### Limit Returned Data

```sql
-- Only return needed columns
SELECT id, value, ST_AsText(geom) FROM observations
WHERE ST_Intersects(geom, ST_MakeEnvelope(-122.5, 37.5, -122.0, 38.0, 4326))
LIMIT 10000;
```

### Use EXPLAIN ANALYZE

```sql
EXPLAIN ANALYZE
SELECT count(*) FROM observations
WHERE ST_DWithin(geom, ST_SetSRID(ST_MakePoint(-122.4, 37.7), 4326)::geography, 1000);
```

Check that the output shows "Index Scan" or "Bitmap Index Scan", not "Seq Scan".

## Reducing Data Transfer in APIs

### Return Only Necessary Fields

```python
# In GEO-INFER-API endpoint
@app.get("/v1/observations")
async def get_observations(
    bbox: str,
    fields: str = "id,value",  # let client choose fields
    limit: int = 1000,
):
    field_list = fields.split(",")
    # Query only requested fields from PostGIS
    results = query_observations(bbox, field_list, limit)
    return results
```

### Use GeoJSON Precision Control

```python
import json

def truncate_coordinates(geojson: dict, precision: int = 6) -> dict:
    """Reduce GeoJSON coordinate precision to save bandwidth.

    6 decimal places = ~0.11m precision, sufficient for most applications.
    """
    def _truncate(coords):
        if isinstance(coords[0], (int, float)):
            return [round(c, precision) for c in coords]
        return [_truncate(c) for c in coords]

    geojson["coordinates"] = _truncate(geojson["coordinates"])
    return geojson
```

### Compress Responses

Enable gzip compression for spatial API responses:

```python
from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware

app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=1000)
```

## Profiling with cProfile and line_profiler

### Full Script Profiling

```bash
# Profile and sort by cumulative time
python -m cProfile -s cumulative -o profile.prof my_analysis.py

# View results
python -c "
import pstats
p = pstats.Stats('profile.prof')
p.sort_stats('cumulative')
p.print_stats(20)
"
```

### Targeted Line Profiling

```python
from line_profiler import profile

@profile
def spatial_analysis_pipeline(gdf):
    # Each line gets timing information
    gdf = gdf.to_crs("EPSG:32610")
    buffered = gdf.buffer(100)
    joined = gpd.sjoin(gdf, other_gdf, predicate="intersects")
    result = joined.groupby("region").agg({"value": "mean"})
    return result
```

## Common Anti-Patterns

### Anti-Pattern: Row-Wise Geometry Operations

```python
# SLOW: iterating over GeoDataFrame rows
for idx, row in gdf.iterrows():
    gdf.loc[idx, "area"] = row.geometry.area

# FAST: vectorized operation
gdf["area"] = gdf.geometry.area
```

### Anti-Pattern: Repeated CRS Transformations

```python
# SLOW: transforming CRS inside a loop
for chunk in chunks:
    chunk = chunk.to_crs("EPSG:32610")  # transforms every iteration
    process(chunk)

# FAST: transform once, then chunk
gdf = gdf.to_crs("EPSG:32610")
for chunk in np.array_split(gdf, n_chunks):
    process(chunk)
```

### Anti-Pattern: Building GeoDataFrame Row by Row

```python
# SLOW: appending rows one at a time
gdf = gpd.GeoDataFrame()
for point in points:
    row = gpd.GeoDataFrame({"geometry": [Point(point)]})
    gdf = pd.concat([gdf, row])

# FAST: build all at once
from shapely.geometry import Point
geometries = [Point(x, y) for x, y in points]
gdf = gpd.GeoDataFrame(geometry=geometries, crs="EPSG:4326")
```

### Anti-Pattern: Using Shapely Where H3 Suffices

```python
# SLOW: point-in-polygon test with shapely for millions of points
from shapely.geometry import Point
results = [polygon.contains(Point(x, y)) for x, y in coords]

# FAST: use H3 cell membership instead
import h3
target_cells = set(h3.geo_to_cells(polygon.__geo_interface__, res=9))
results = [h3.latlng_to_cell(lat, lng, 9) in target_cells for lat, lng in coords]
```

## When to Switch from Shapely to H3

Use H3 instead of shapely when:

- You are doing point-in-region tests at scale (> 100K points)
- You need aggregation by area (H3 cells are uniform size)
- You are working with grid data or hexagonal tessellations
- You need fast neighbor lookups

Keep shapely when:

- You need exact geometric operations (intersection, union, difference)
- You are working with irregular polygons that do not align to H3 cells
- You need sub-meter precision for engineering applications

## See Also

- [Performance Optimization](../advanced/performance_optimization.md) -- optimization techniques
- [Scaling Guide](../advanced/scaling_guide.md) -- when to move beyond a single machine
- [Troubleshooting](troubleshooting.md) -- general error diagnosis
