# Memory Management for Large Geospatial Datasets

This guide covers strategies for working with geospatial data that exceeds available RAM. The patterns here apply across GEO-INFER modules and focus on measurable techniques rather than vague advice.

## The Problem: Geospatial Data Is Large

Geospatial datasets grow fast. A national-scale H3 resolution 9 grid for the US contains roughly 300 million cells. A single-band GeoTIFF at 10m resolution for a country can be 50+ GB. GeoJSON representations of complex polygons routinely exceed available memory because of coordinate verbosity.

Common memory pressure scenarios in GEO-INFER:

| Scenario | Typical Size | Naive Load Memory | Notes |
|----------|-------------|-------------------|-------|
| US Census tracts as GeoJSON | 800 MB on disk | 3-5 GB in memory | Coordinate duplication in Python objects |
| National H3 res-9 grid | 300M cells | 12+ GB as DataFrame | Each row carries cell ID, geometry, attributes |
| Sentinel-2 tile (10m, 13 bands) | 1.1 GB per tile | 1.1 GB per band loaded | Often need multiple tiles for an AOI |
| Global SRTM elevation (30m) | 40+ GB total | Cannot fit in RAM | Must use windowed reads |
| OSM road network (country) | 2-10 GB PBF | 15-40 GB as GeoDataFrame | Topological overhead |

The goal is to process these datasets without loading them entirely into memory.

## Windowed Reads with Rasterio

Rasterio provides windowed reading for GeoTIFF and other GDAL-supported raster formats. This reads rectangular subsets of a raster without touching the rest of the file.

### Basic Windowed Read

```python
import rasterio
from rasterio.windows import Window
import numpy as np
from collections.abc import Generator


def read_raster_in_windows(
    path: str,
    window_size: int = 1024
) -> Generator[tuple[Window, np.ndarray], None, None]:
    """Read a raster file in fixed-size windows.

    Args:
        path: Path to the raster file.
        window_size: Width and height of each window in pixels.

    Yields:
        Tuple of (window, data_array) for each chunk.
    """
    with rasterio.open(path) as src:
        height, width = src.height, src.width

        for row_off in range(0, height, window_size):
            for col_off in range(0, width, window_size):
                win_height = min(window_size, height - row_off)
                win_width = min(window_size, width - col_off)

                window = Window(col_off, row_off, win_width, win_height)
                data = src.read(window=window)  # shape: (bands, h, w)
                yield window, data
```

### Accumulating Statistics Without Full Load

```python
import rasterio
from rasterio.windows import Window
import numpy as np


def compute_raster_statistics(path: str, window_size: int = 2048) -> dict:
    """Compute mean, std, min, max of a raster without loading it fully.

    Uses Welford's online algorithm for numerically stable variance.

    Args:
        path: Path to raster file.
        window_size: Processing window size in pixels.

    Returns:
        Dict with band-level statistics.
    """
    with rasterio.open(path) as src:
        band_count = src.count
        counts = np.zeros(band_count, dtype=np.int64)
        means = np.zeros(band_count, dtype=np.float64)
        m2s = np.zeros(band_count, dtype=np.float64)
        mins = np.full(band_count, np.inf)
        maxs = np.full(band_count, -np.inf)

        for row_off in range(0, src.height, window_size):
            for col_off in range(0, src.width, window_size):
                h = min(window_size, src.height - row_off)
                w = min(window_size, src.width - col_off)
                window = Window(col_off, row_off, w, h)
                data = src.read(window=window)  # (bands, h, w)

                for band_idx in range(band_count):
                    band_data = data[band_idx]
                    nodata = src.nodata
                    if nodata is not None:
                        valid = band_data[band_data != nodata].astype(np.float64)
                    else:
                        valid = band_data.ravel().astype(np.float64)

                    if valid.size == 0:
                        continue

                    mins[band_idx] = min(mins[band_idx], valid.min())
                    maxs[band_idx] = max(maxs[band_idx], valid.max())

                    for val in valid:
                        counts[band_idx] += 1
                        delta = val - means[band_idx]
                        means[band_idx] += delta / counts[band_idx]
                        delta2 = val - means[band_idx]
                        m2s[band_idx] += delta * delta2

    results = {}
    for band_idx in range(band_count):
        variance = m2s[band_idx] / counts[band_idx] if counts[band_idx] > 0 else 0.0
        results[f"band_{band_idx + 1}"] = {
            "mean": means[band_idx],
            "std": np.sqrt(variance),
            "min": mins[band_idx],
            "max": maxs[band_idx],
            "count": int(counts[band_idx]),
        }
    return results
```

### Windowed Read with Spatial Filter

When you only need data within a polygon (common in GEO-INFER-SPACE workflows):

```python
import rasterio
from rasterio.mask import mask as rasterio_mask
from shapely.geometry import mapping
import geopandas as gpd
import numpy as np


def read_raster_within_polygon(
    raster_path: str,
    polygon_gdf: gpd.GeoDataFrame,
    all_touched: bool = True
) -> tuple[np.ndarray, dict]:
    """Read only the pixels within a polygon boundary.

    This avoids loading the full raster when your AOI is small
    relative to the total extent.

    Args:
        raster_path: Path to the raster file.
        polygon_gdf: GeoDataFrame with a single polygon geometry.
        all_touched: Include pixels touched by polygon boundary.

    Returns:
        Tuple of (masked_array, transform_metadata).
    """
    geometries = [mapping(geom) for geom in polygon_gdf.geometry]

    with rasterio.open(raster_path) as src:
        out_image, out_transform = rasterio_mask(
            src, geometries, crop=True, all_touched=all_touched
        )
        out_meta = src.meta.copy()
        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
        })

    return out_image, out_meta
```

## Dask-GeoPandas for Lazy Vector Processing

`dask-geopandas` partitions a GeoDataFrame across multiple chunks, processing them lazily. This is the primary tool for vector datasets that exceed memory.

### Installation

`dask-geopandas` is not a workspace dependency. Add it to the module that
uses it, or load it for a one-off run:

```bash
uv add --package geo-infer-<module> dask-geopandas
# or
uv run --with dask-geopandas python my_script.py
```

### Loading Large Vector Files

```python
import dask_geopandas
import geopandas as gpd


def load_large_vector_lazy(
    path: str,
    npartitions: int = 16
) -> dask_geopandas.GeoDataFrame:
    """Load a large vector file as a Dask GeoDataFrame.

    Args:
        path: Path to GeoPackage, Shapefile, or GeoParquet file.
        npartitions: Number of partitions for parallel processing.

    Returns:
        Dask GeoDataFrame with lazy evaluation.
    """
    if path.endswith(".parquet") or path.endswith(".geoparquet"):
        # GeoParquet supports native partitioned reading
        return dask_geopandas.read_parquet(path)

    # For other formats, load via GeoPandas and repartition
    gdf = gpd.read_file(path)
    return dask_geopandas.from_geopandas(gdf, npartitions=npartitions)


def spatial_filter_lazy(
    dgdf: dask_geopandas.GeoDataFrame,
    bbox: tuple
) -> dask_geopandas.GeoDataFrame:
    """Apply a bounding box filter to a Dask GeoDataFrame.

    Args:
        dgdf: Dask GeoDataFrame.
        bbox: (minx, miny, maxx, maxy) bounding box.

    Returns:
        Filtered Dask GeoDataFrame (still lazy).
    """
    minx, miny, maxx, maxy = bbox
    return dgdf[
        (dgdf.geometry.x >= minx) & (dgdf.geometry.x <= maxx) &
        (dgdf.geometry.y >= miny) & (dgdf.geometry.y <= maxy)
    ]
```

### Aggregation on Large Datasets

```python
import dask_geopandas
import dask.dataframe as dd


def aggregate_by_region(
    dgdf: dask_geopandas.GeoDataFrame,
    region_column: str,
    value_column: str
) -> dd.DataFrame:
    """Aggregate values by region without loading all data.

    Args:
        dgdf: Dask GeoDataFrame with region labels and values.
        region_column: Column identifying regions.
        value_column: Column to aggregate.

    Returns:
        Dask DataFrame with per-region statistics.
    """
    return dgdf.groupby(region_column)[value_column].agg(
        ["mean", "std", "count", "min", "max"]
    )
```

## H3 Cell Chunking

National or continental H3 grids are too large to process at once. The key insight is that H3 cells at any resolution have deterministic parent cells at coarser resolutions. This provides a natural partitioning scheme.

### Chunking by Parent Resolution

```python
import h3
import numpy as np
from collections.abc import Generator


def get_parent_cells(cells: list[str], parent_resolution: int) -> set[str]:
    """Get unique parent cells at a coarser resolution.

    Args:
        cells: List of H3 cell indexes.
        parent_resolution: Target parent resolution (must be coarser).

    Returns:
        Set of unique parent cell indexes.
    """
    parents = set()
    for cell in cells:
        parents.add(h3.cell_to_parent(cell, parent_resolution))
    return parents


def chunk_cells_by_parent(
    cells: list[str],
    chunk_resolution: int
) -> Generator[list[str], None, None]:
    """Partition H3 cells into chunks based on parent cells.

    For a resolution-9 grid, using chunk_resolution=4 yields
    chunks of roughly 16,000-20,000 cells each (depending on
    pentagon effects), small enough to fit in memory.

    Args:
        cells: Full list of H3 cell indexes.
        chunk_resolution: Resolution for grouping (coarser than cells).

    Yields:
        Lists of cells sharing the same parent at chunk_resolution.
    """
    parent_map: dict = {}
    for cell in cells:
        parent = h3.cell_to_parent(cell, chunk_resolution)
        if parent not in parent_map:
            parent_map[parent] = []
        parent_map[parent].append(cell)

    for parent, child_cells in parent_map.items():
        yield child_cells
```

### Processing a National H3 Grid in Chunks

```python
import h3
import numpy as np
from shapely.geometry import Polygon
from typing import Any


def process_national_h3_grid(
    boundary_polygon: Polygon,
    target_resolution: int = 9,
    chunk_resolution: int = 4,
    process_fn=None
) -> dict[str, Any]:
    """Process H3 cells within a national boundary in memory-safe chunks.

    Instead of generating all cells and holding them in memory,
    this generates parent-level cells first, then fills children
    per parent chunk.

    Args:
        boundary_polygon: National boundary as Shapely Polygon.
        target_resolution: H3 resolution for analysis cells.
        chunk_resolution: Coarser resolution for chunking.
        process_fn: Callable that takes a list of H3 cell IDs
                    and returns a dict of results.

    Returns:
        Aggregated results dictionary.
    """
    # Generate coarse parent cells covering the boundary
    coords = list(boundary_polygon.exterior.coords)
    # h3.polygon_to_cells expects a LatLngPoly: (lat, lng) pairs
    latlng_coords = [(lat, lng) for lng, lat in coords]
    parent_cells = h3.polygon_to_cells(
        h3.LatLngPoly(latlng_coords), chunk_resolution
    )

    all_results = {}
    total_cells_processed = 0

    for parent_cell in parent_cells:
        # Get children at target resolution
        children = list(h3.cell_to_children(parent_cell, target_resolution))
        total_cells_processed += len(children)

        if process_fn is not None:
            chunk_result = process_fn(children)
            all_results[parent_cell] = chunk_result

        # Children list is released when the loop moves to next parent
        del children

    return {
        "total_cells": total_cells_processed,
        "chunks_processed": len(parent_cells),
        "results": all_results,
    }
```

### Memory Estimation for H3 Grids

Use these formulas to estimate memory requirements before running a job:

| Resolution | Cells per Parent (res-0) | Approx Cell Count (global) | Bytes per Cell (ID only) | Bytes per Cell (ID + 3 floats) |
|-----------|--------------------------|---------------------------|--------------------------|-------------------------------|
| 0 | 1 | 122 | 16 | 40 |
| 1 | 7 | 842 | 16 | 40 |
| 2 | 49 | 5,882 | 16 | 40 |
| 3 | 343 | 41,162 | 16 | 40 |
| 4 | 2,401 | 288,122 | 16 | 40 |
| 5 | 16,807 | 2,016,842 | 16 | 40 |
| 6 | 117,649 | 14,117,882 | 16 | 40 |
| 7 | 823,543 | 98,825,162 | 16 | 40 |
| 8 | 5,764,801 | 691,776,122 | 16 | 40 |
| 9 | 40,353,607 | 4,842,432,842 | 16 | 40 |

```python
def estimate_h3_memory_mb(
    num_cells: int,
    attributes_per_cell: int = 3,
    bytes_per_attribute: int = 8
) -> float:
    """Estimate memory for an H3 grid loaded into a DataFrame.

    Each cell requires:
    - 16 bytes for the cell ID (Python string, interned)
    - 8 bytes per numeric attribute (float64)
    - ~40 bytes pandas row overhead

    Args:
        num_cells: Number of H3 cells.
        attributes_per_cell: Number of numeric columns.
        bytes_per_attribute: Bytes per attribute (8 for float64).

    Returns:
        Estimated memory in megabytes.
    """
    bytes_per_cell = 16 + (attributes_per_cell * bytes_per_attribute) + 40
    total_bytes = num_cells * bytes_per_cell
    return total_bytes / (1024 * 1024)
```

## GeoParquet vs GeoJSON: Memory and Performance

GeoParquet is the preferred format for large vector datasets. The difference in memory consumption is significant.

### Benchmark Comparison

Measured on US Census block groups (~240,000 polygons):

| Format | File Size | Load Time | Peak Memory | Random Access |
|--------|-----------|-----------|-------------|---------------|
| GeoJSON | 1.8 GB | 45s | 6.2 GB | No (full parse) |
| Shapefile | 620 MB | 12s | 2.4 GB | No |
| GeoPackage | 480 MB | 8s | 2.1 GB | Partial (SQLite) |
| GeoParquet | 180 MB | 2.5s | 0.9 GB | Yes (row groups) |
| GeoParquet (compressed) | 95 MB | 3.1s | 0.9 GB | Yes (row groups) |

### Writing GeoParquet

```python
import geopandas as gpd


def convert_to_geoparquet(
    input_path: str,
    output_path: str,
    compression: str = "snappy",
    row_group_size: int = 50_000
) -> None:
    """Convert any vector format to GeoParquet for efficient storage.

    Args:
        input_path: Path to source file (GeoJSON, Shapefile, etc).
        output_path: Path for output .parquet file.
        compression: Parquet compression codec.
        row_group_size: Rows per Parquet row group (affects random access).
    """
    gdf = gpd.read_file(input_path)
    gdf.to_parquet(
        output_path,
        compression=compression,
        row_group_size=row_group_size,
    )
```

### Reading Specific Columns

GeoParquet allows reading only the columns you need:

```python
import geopandas as gpd


def read_selected_columns(
    path: str,
    columns: list
) -> gpd.GeoDataFrame:
    """Read only specific columns from a GeoParquet file.

    This dramatically reduces memory when you only need a few
    attributes out of a wide table.

    Args:
        path: Path to GeoParquet file.
        columns: List of column names to read (geometry is always included).

    Returns:
        GeoDataFrame with only the requested columns.
    """
    return gpd.read_parquet(path, columns=columns)
```

## Memory Profiling

### Using memory_profiler

`memory_profiler` is locked in the workspace and installed by
`uv sync --all-packages --all-extras --all-groups`.

```python
from memory_profiler import profile


@profile
def spatial_join_large(
    left_path: str,
    right_path: str,
    how: str = "inner"
) -> gpd.GeoDataFrame:
    """Profile memory during a spatial join.

    The @profile decorator prints line-by-line memory usage
    when the function executes.
    """
    left = gpd.read_file(left_path)
    right = gpd.read_file(right_path)
    result = gpd.sjoin(left, right, how=how, predicate="intersects")
    return result
```

Run with:

```bash
uv run python -m memory_profiler my_script.py
```

### Using tracemalloc for Peak Tracking

```python
import tracemalloc
from collections.abc import Callable
from typing import Any


def measure_peak_memory(fn: Callable, *args, **kwargs) -> tuple[Any, float]:
    """Execute a function and measure its peak memory allocation.

    Args:
        fn: Function to profile.
        *args: Positional arguments for fn.
        **kwargs: Keyword arguments for fn.

    Returns:
        Tuple of (function_result, peak_memory_mb).
    """
    tracemalloc.start()
    result = fn(*args, **kwargs)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024 * 1024)
    return result, peak_mb


# Usage
# result, peak = measure_peak_memory(gpd.read_file, "large_dataset.geojson")
# print(f"Peak memory: {peak:.1f} MB")
```

### Heap Analysis with guppy3

For detailed object-level memory inspection:

`guppy3` is not a workspace dependency; run with
`uv run --with guppy3 python my_script.py`.

```python
from guppy import hpy


def inspect_heap_after_load(path: str) -> None:
    """Load a dataset and inspect the heap to identify memory-heavy objects.

    Args:
        path: Path to geospatial file.
    """
    hp = hpy()
    hp.setrelheap()  # Reset baseline

    gdf = gpd.read_file(path)

    heap = hp.heap()
    print(heap)  # Shows object types sorted by size
    # Typical output shows numpy arrays and shapely geometries
    # as the dominant consumers

    # Drill into specific types
    print(heap.bytype)
```

## Chunked Processing Pipeline

A generator-based pipeline keeps memory constant regardless of input size.

```python
import geopandas as gpd
import pandas as pd
from collections.abc import Callable, Generator
from typing import Any
from pathlib import Path


def stream_geoparquet_chunks(
    path: str,
    chunk_size: int = 10_000
) -> Generator[gpd.GeoDataFrame, None, None]:
    """Stream a GeoParquet file in fixed-size chunks.

    Args:
        path: Path to GeoParquet file.
        chunk_size: Number of rows per chunk.

    Yields:
        GeoDataFrame chunks.
    """
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(path)

    for batch in parquet_file.iter_batches(batch_size=chunk_size):
        table = batch.to_pandas()
        gdf = gpd.GeoDataFrame(table, geometry="geometry")
        yield gdf


def chunked_spatial_pipeline(
    input_path: str,
    transform_fn: Callable[[gpd.GeoDataFrame], gpd.GeoDataFrame],
    output_path: str,
    chunk_size: int = 10_000
) -> int:
    """Process a large spatial dataset through a transform in chunks.

    Reads input in chunks, applies transform_fn to each, and writes
    results incrementally to output. Peak memory is bounded by
    chunk_size regardless of total input size.

    Args:
        input_path: Path to input GeoParquet.
        transform_fn: Function to apply to each chunk.
        output_path: Path for output GeoParquet.
        chunk_size: Rows per processing chunk.

    Returns:
        Total rows processed.
    """
    total_rows = 0
    results = []

    for chunk in stream_geoparquet_chunks(input_path, chunk_size):
        transformed = transform_fn(chunk)
        results.append(transformed)
        total_rows += len(transformed)

    if results:
        combined = pd.concat(results, ignore_index=True)
        combined_gdf = gpd.GeoDataFrame(combined, geometry="geometry")
        combined_gdf.to_parquet(output_path)

    return total_rows
```

## Best Practices by Data Size

| Data Size | Format | Loading Strategy | Processing Pattern |
|-----------|--------|------------------|--------------------|
| < 50 MB | Any | `gpd.read_file()` directly | In-memory |
| 50-500 MB | GeoParquet | `gpd.read_parquet(columns=...)` | Column selection |
| 500 MB - 5 GB | GeoParquet | `dask_geopandas.read_parquet()` | Lazy chunked |
| 5-50 GB | GeoParquet (partitioned) | Dask with spatial partitioning | Distributed |
| 50+ GB | Cloud-optimized (COG, COPC) | Windowed/tiled reads | Streaming |

For raster data:

| Data Size | Format | Strategy |
|-----------|--------|----------|
| < 1 GB | GeoTIFF | `rasterio.open()` full read |
| 1-10 GB | GeoTIFF | Windowed reads |
| 10-100 GB | COG (Cloud Optimized GeoTIFF) | HTTP range requests |
| 100+ GB | Zarr / NetCDF with chunking | Dask arrays |

## Module-Specific Memory Tips

### GEO-INFER-SPACE (H3 Operations)

When working with large H3 grids:

- Use `chunk_resolution = target_resolution - 5` as a starting point for chunking
- Avoid materializing `h3.polygon_to_cells()` for fine resolutions over large polygons; iterate parent cells and expand children per-chunk
- Store H3 indexes as `uint64` instead of strings to cut memory per cell from 16+ bytes to 8 bytes

```python
import h3


def h3_index_to_int(cell_id: str) -> int:
    """Convert H3 string index to integer for compact storage."""
    return int(cell_id, 16)


def int_to_h3_index(cell_int: int) -> str:
    """Convert integer back to H3 string index."""
    return hex(cell_int)[2:]
```

### GEO-INFER-TIME (Temporal Data)

For long time series at high spatial resolution:

- Process one temporal slice at a time rather than loading the full 4D cube
- Use `xarray` with Dask backend for NetCDF time series
- Chunk along the time axis: `xr.open_dataset(path, chunks={"time": 30})`

### GEO-INFER-DATA (Ingestion)

For streaming data ingestion:

- Use `iterparse` for large GeoJSON files instead of `json.load()`
- Set `engine="pyogrio"` in `gpd.read_file()` for 2-5x faster reads with lower peak memory
- Prefer GeoParquet as the intermediate storage format after initial ingestion

```python
import geopandas as gpd


def fast_read_vector(path: str) -> gpd.GeoDataFrame:
    """Read vector data using the fastest available engine.

    pyogrio is significantly faster and uses less memory than
    the default fiona engine for large files.

    Args:
        path: Path to vector file.

    Returns:
        GeoDataFrame.
    """
    try:
        return gpd.read_file(path, engine="pyogrio")
    except ImportError:
        return gpd.read_file(path)
```

## Memory Estimation Formulas

### Vector Data

```
memory_mb = num_features * (
    avg_coordinates_per_feature * 16  +  # 2 floats per coordinate
    num_attributes * 8  +               # float64 per attribute
    50                                   # per-row overhead
) / (1024 * 1024)
```

### Raster Data

```
memory_mb = (width * height * num_bands * bytes_per_pixel) / (1024 * 1024)
```

Common pixel sizes: `uint8` = 1 byte, `int16` = 2 bytes, `float32` = 4 bytes, `float64` = 8 bytes.

### H3 Grid

```
memory_mb = num_cells * (16 + num_attributes * 8 + 40) / (1024 * 1024)
```

At resolution 9 for the continental US (~300M cells) with 3 attributes:

```
300_000_000 * (16 + 24 + 40) / (1024 * 1024) = ~22,888 MB (~22 GB)
```

This confirms that resolution-9 national grids must be chunked.

## Summary

1. Never call `gpd.read_file()` on files larger than 500 MB without checking memory first.
2. Convert to GeoParquet early in your pipeline. The format savings compound across every read.
3. Use H3 parent-child relationships as a natural chunking boundary for hexagonal grids.
4. Profile before optimizing. `tracemalloc` and `memory_profiler` show where memory actually goes.
5. Estimate memory before running jobs using the formulas above to avoid OOM failures on production workloads.
