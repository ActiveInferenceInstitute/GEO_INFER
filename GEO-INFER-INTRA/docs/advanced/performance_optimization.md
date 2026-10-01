# Performance Optimization

> **Illustrative example notice.** This page contains historical or
> conceptual integration sketches. Names such as `SpatialAnalyzer` and
> domain-specific facade classes are not public GEO-INFER exports in the
> current checkout; verify imports against each module's `src/` package
> and use the module README/tests for executable examples.

This guide covers profiling, bottleneck identification, and optimization techniques for geospatial Python code within the GEO-INFER framework. The focus is on practical patterns that yield measurable gains, not speculative micro-optimization.

## Profiling Geospatial Code

Before optimizing, measure. Profile at the function level first, then drill into hot lines.

### cProfile for Function-Level Profiling

```python
import cProfile
import pstats

# Profile a spatial analysis run
profiler = cProfile.Profile()
profiler.enable()

from geo_infer_space import SpatialAnalyzer
analyzer = SpatialAnalyzer()
results = analyzer.cluster_points(data, resolution=7)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats("cumulative")
stats.print_stats(20)  # top 20 by cumulative time
```

### line_profiler for Hot Loops

`line_profiler` is not a workspace dependency; load it for one run with
`uv run --with line_profiler` (see below).

```python
# Decorate the function you suspect is slow
from line_profiler import profile

@profile
def compute_spatial_weights(coordinates: np.ndarray, bandwidth: float) -> np.ndarray:
    distances = np.sqrt(np.sum((coordinates[:, None] - coordinates[None, :]) ** 2, axis=-1))
    weights = np.exp(-0.5 * (distances / bandwidth) ** 2)
    np.fill_diagonal(weights, 0.0)
    return weights / weights.sum(axis=1, keepdims=True)
```

Run with:

```bash
uv run --with line_profiler kernprof -l -v my_script.py
```

### memory_profiler for Spatial Data

Geospatial workloads often hit memory limits before CPU limits. Profile memory to catch unexpected allocations.

```python
from memory_profiler import profile

@profile
def load_and_index_raster(path: str) -> np.ndarray:
    import rasterio
    with rasterio.open(path) as src:
        data = src.read(1)
    return data
```

## H3 Vectorized Operations

GEO-INFER-SPACE uses H3 v4. The single-cell functions (`h3.latlng_to_cell`) are Python-call-heavy. Batch operations reduce overhead.

### Vectorized Indexing

```python
import h3
import numpy as np

# Slow: one Python call per point
def index_points_slow(lats: np.ndarray, lngs: np.ndarray, res: int) -> list:
    return [h3.latlng_to_cell(lat, lng, res) for lat, lng in zip(lats, lngs)]

# Faster: use numpy to reduce Python-level iteration
def index_points_fast(lats: np.ndarray, lngs: np.ndarray, res: int) -> np.ndarray:
    """Vectorize H3 indexing with numpy string array output."""
    coords = np.column_stack([lats, lngs])
    cells = np.empty(len(coords), dtype="U16")
    for i, (lat, lng) in enumerate(coords):
        cells[i] = h3.latlng_to_cell(lat, lng, res)
    return cells
```

For genuinely large datasets (10M+ points), push the loop into C via `h3ronpy` or pre-bin with numpy before calling H3.

### Resolution Selection

Higher H3 resolutions produce more cells and more memory. Choose resolution based on data density, not maximum precision.

| Resolution | Avg Hex Area | Use Case |
|-----------|-------------|----------|
| 3 | ~12,000 km^2 | Country-level aggregation |
| 5 | ~253 km^2 | Regional analysis |
| 7 | ~5.2 km^2 | City-level analysis |
| 9 | ~0.1 km^2 | Neighborhood-level |
| 11 | ~0.002 km^2 | Building-level |

## NumPy and Pandas Optimization

### Avoid Row-Wise Iteration

```python
import pandas as pd
import numpy as np

# Anti-pattern: iterrows for spatial distance
def distances_slow(gdf):
    results = []
    for idx, row in gdf.iterrows():
        d = np.sqrt((gdf.geometry.x - row.geometry.x)**2 +
                     (gdf.geometry.y - row.geometry.y)**2)
        results.append(d.values)
    return np.array(results)

# Better: vectorized pairwise distance
def distances_fast(gdf):
    x = gdf.geometry.x.values
    y = gdf.geometry.y.values
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    return np.sqrt(dx**2 + dy**2)
```

### Chunked Processing for Large DataFrames

```python
def process_spatial_chunks(df: pd.DataFrame, chunk_size: int = 50_000):
    """Process a large spatial DataFrame in chunks to limit peak memory."""
    results = []
    for start in range(0, len(df), chunk_size):
        chunk = df.iloc[start:start + chunk_size]
        processed = _analyze_chunk(chunk)
        results.append(processed)
    return pd.concat(results, ignore_index=True)
```

## Lazy Evaluation with Dask

For datasets that exceed memory, use Dask to process partitions lazily.

```python
import dask.dataframe as dd

# Load a large CSV of spatial observations
ddf = dd.read_csv("observations_*.csv", blocksize="128MB")

# Filter and aggregate lazily
result = (
    ddf[ddf["quality_flag"] == "good"]
    .groupby("h3_cell")
    .agg({"value": "mean", "timestamp": "count"})
    .compute()  # triggers execution
)
```

Dask works well when the computation is partition-friendly (aggregations, filters, maps). It does not help with operations requiring the full dataset in memory (global spatial joins, full covariance matrices).

## Spatial Indexing

### R-tree for Geometry Queries

```python
from shapely import STRtree
from shapely.geometry import Point

# Build spatial index once
tree = STRtree([Point(x, y) for x, y in coordinates])

# Query: find all points within a polygon
query_polygon = some_polygon
indices = tree.query(query_polygon)
```

### H3 as a Spatial Index

H3 cells are a natural spatial index. Instead of R-tree queries, aggregate data by cell, then query neighboring cells.

```python
import h3

def get_neighborhood_data(cell: str, data_by_cell: dict, k: int = 1) -> list:
    """Retrieve data from a cell and its k-ring neighbors."""
    neighbors = h3.grid_disk(cell, k)
    return [data_by_cell[n] for n in neighbors if n in data_by_cell]
```

## Caching Strategies

### functools.lru_cache for Repeated Computations

```python
from functools import lru_cache

@lru_cache(maxsize=4096)
def cell_to_polygon_wkt(cell: str) -> str:
    """Cache H3 cell boundary conversion."""
    boundary = h3.cell_to_boundary(cell)
    coords = ", ".join(f"{lng} {lat}" for lat, lng in boundary)
    first = boundary[0]
    return f"POLYGON(({coords}, {first[1]} {first[0]}))"
```

### Disk Caching for Expensive Computations

```python
import hashlib
import pickle
from pathlib import Path

CACHE_DIR = Path(".cache/geo_infer")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def cached_computation(key: str, compute_fn, *args, **kwargs):
    """Cache computation results to disk."""
    cache_key = hashlib.sha256(key.encode()).hexdigest()[:16]
    cache_path = CACHE_DIR / f"{cache_key}.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    result = compute_fn(*args, **kwargs)
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    return result
```

## Memory-Mapped Files for Large Rasters

For raster data that exceeds available RAM, use memory-mapped access.

```python
import rasterio
import numpy as np

def read_raster_window(path: str, row_off: int, col_off: int,
                       height: int, width: int) -> np.ndarray:
    """Read a window from a raster without loading the full file."""
    with rasterio.open(path) as src:
        window = rasterio.windows.Window(col_off, row_off, width, height)
        return src.read(1, window=window)

# Process a large raster in tiles
def process_raster_tiled(path: str, tile_size: int = 1024):
    with rasterio.open(path) as src:
        for row in range(0, src.height, tile_size):
            for col in range(0, src.width, tile_size):
                h = min(tile_size, src.height - row)
                w = min(tile_size, src.width - col)
                tile = src.read(1, window=rasterio.windows.Window(col, row, w, h))
                yield tile, row, col
```

## JIT Compilation with Numba

For custom spatial kernels where numpy broadcasting is insufficient, use numba to compile Python loops to machine code.

```python
from numba import njit

@njit
def inverse_distance_weights(x_query: float, y_query: float,
                              x_pts: np.ndarray, y_pts: np.ndarray,
                              power: float = 2.0) -> np.ndarray:
    """Compute IDW weights using numba-compiled loop."""
    n = len(x_pts)
    weights = np.empty(n)
    for i in range(n):
        dx = x_query - x_pts[i]
        dy = y_query - y_pts[i]
        dist = np.sqrt(dx * dx + dy * dy)
        if dist < 1e-12:
            weights[:] = 0.0
            weights[i] = 1.0
            return weights
        weights[i] = 1.0 / (dist ** power)
    total = weights.sum()
    for i in range(n):
        weights[i] /= total
    return weights
```

Numba works well for numerical loops over arrays. It does not support arbitrary Python objects or most library calls inside `@njit` functions.

## Benchmarking Methodology

Use `pytest-benchmark` for repeatable benchmarks.

```python
# tests/performance/test_spatial_perf.py
import pytest
import numpy as np

@pytest.mark.performance
def test_h3_indexing_throughput(benchmark):
    lats = np.random.uniform(30, 50, size=10_000)
    lngs = np.random.uniform(-120, -80, size=10_000)

    def index_all():
        import h3
        return [h3.latlng_to_cell(lat, lng, 7) for lat, lng in zip(lats, lngs)]

    result = benchmark(index_all)
    assert len(result) == 10_000
```

Run with:

```bash
uv run python -m pytest tests/performance/ -v --benchmark-only
```

## Module-Specific Tips

| Module | Optimization Target | Technique |
|--------|-------------------|-----------|
| GEO-INFER-SPACE | H3 cell generation | Batch processing, precomputed grids |
| GEO-INFER-BAYES | Cholesky decomposition | Sparse approximations for n > 5000 |
| GEO-INFER-ACT | Free energy computation | Vectorized belief arrays, avoid per-step allocation |
| GEO-INFER-TIME | Sliding window analysis | Deque-based buffers, incremental statistics |
| GEO-INFER-RISK | Monte Carlo simulation | Numba-compiled inner loops, parallel seeds |
| GEO-INFER-TRANSPORT | Routing algorithms | Graph precomputation, bidirectional search |

## See Also

- [Performance Issues](../support/performance_issues.md) -- diagnosing slow runs
- [Scaling Guide](scaling_guide.md) -- when single-machine optimization is insufficient
- [Custom Models](custom_models.md) -- performance considerations for model design
