# Scaling Guide

This guide addresses when and how to scale GEO-INFER workloads beyond a single machine. It covers data size thresholds, tiling strategies, distributed computation patterns, and cost optimization for large-scale geospatial analysis.

## When to Scale

Not every workload needs distributed computing. Scaling adds operational complexity and debugging difficulty. Use this decision framework:

| Data Size | Points/Cells | Recommendation |
|-----------|-------------|----------------|
| < 100 MB | < 1M rows | Single process, in-memory |
| 100 MB - 10 GB | 1M - 100M rows | Single machine, chunked processing |
| 10 GB - 1 TB | 100M - 10B rows | Distributed processing (Dask, Spark) |
| > 1 TB | > 10B rows | Cloud-native pipelines |

### Signals That You Need to Scale

- Single analysis runs exceed 1 hour wall-clock time
- Memory usage exceeds 80% of available RAM
- Multiple users need concurrent access to the same pipeline
- Data arrives faster than a single process can consume it
- You need results at multiple H3 resolutions simultaneously

## Tiling Strategies for Global Datasets

### Fixed-Size Tiles

Partition the spatial extent into regular tiles for parallel processing:

```python
import numpy as np

def generate_tiles(bbox: tuple[float, float, float, float],
                   tile_size_deg: float = 1.0) -> list[tuple[float, float, float, float]]:
    """Generate non-overlapping tiles covering a bounding box.

    Args:
        bbox: (min_lon, min_lat, max_lon, max_lat)
        tile_size_deg: Tile dimension in degrees.

    Returns:
        List of (min_lon, min_lat, max_lon, max_lat) tiles.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    tiles = []
    lat = min_lat
    while lat < max_lat:
        lon = min_lon
        while lon < max_lon:
            tiles.append((
                lon,
                lat,
                min(lon + tile_size_deg, max_lon),
                min(lat + tile_size_deg, max_lat),
            ))
            lon += tile_size_deg
        lat += tile_size_deg
    return tiles

# Global coverage at 1-degree resolution = ~65,000 tiles
global_tiles = generate_tiles((-180, -90, 180, 90), tile_size_deg=1.0)
```

### H3-Based Tiling

Use coarse H3 cells as tile boundaries. This ensures tiles align with the H3 hierarchy and simplifies aggregation.

```python
import h3

def h3_tile_partition(bbox: tuple[float, float, float, float],
                       tile_resolution: int = 2) -> list[str]:
    """Partition a bounding box into H3 cells at a coarse resolution."""
    min_lon, min_lat, max_lon, max_lat = bbox
    center_lat = (min_lat + max_lat) / 2
    center_lng = (min_lon + max_lon) / 2

    # Get cells covering the area
    center_cell = h3.latlng_to_cell(center_lat, center_lng, tile_resolution)
    # Expand outward until we cover the bbox
    k = max(
        int(abs(max_lat - min_lat) / 2),  # rough estimate
        int(abs(max_lon - min_lon) / 2),
        1,
    )
    return list(h3.grid_disk(center_cell, k))
```

### Resolution Hierarchy for Multi-Scale Analysis

Process data at coarse resolution first, then drill into areas of interest at finer resolution:

```python
def hierarchical_analysis(data_by_cell: dict,
                           coarse_res: int = 5,
                           fine_res: int = 9,
                           threshold: float = 0.8) -> dict:
    """Two-pass analysis: coarse screening then fine-grained where needed.

    Args:
        data_by_cell: Dict mapping H3 cell -> observation value at fine_res.
        coarse_res: Resolution for initial screening pass.
        fine_res: Resolution for detailed analysis.
        threshold: Value threshold triggering fine-grained analysis.
    """
    # Pass 1: aggregate to coarse resolution
    coarse_agg = {}
    for cell, value in data_by_cell.items():
        parent = h3.cell_to_parent(cell, coarse_res)
        coarse_agg.setdefault(parent, []).append(value)

    coarse_means = {cell: np.mean(vals) for cell, vals in coarse_agg.items()}

    # Pass 2: detailed analysis only where coarse mean exceeds threshold
    results = {}
    for coarse_cell, mean_val in coarse_means.items():
        if mean_val > threshold:
            children = h3.cell_to_children(coarse_cell, fine_res)
            for child in children:
                if child in data_by_cell:
                    results[child] = data_by_cell[child]

    return results
```

## Distributed Active Inference

Active Inference belief propagation can be partitioned spatially. Each spatial partition maintains local beliefs, with periodic synchronization at boundaries.

### Spatial Belief Partitioning

```python
import numpy as np

def partition_beliefs(beliefs: dict[str, np.ndarray],
                      partitions: list[list[str]]) -> list[dict[str, np.ndarray]]:
    """Split a belief dictionary into spatial partitions."""
    return [
        {cell: beliefs[cell] for cell in part if cell in beliefs}
        for part in partitions
    ]

def synchronize_boundary_beliefs(
    partition_a: dict[str, np.ndarray],
    partition_b: dict[str, np.ndarray],
    boundary_cells: list[str],
    mixing_weight: float = 0.5,
) -> None:
    """Average beliefs at partition boundaries for consistency."""
    for cell in boundary_cells:
        if cell in partition_a and cell in partition_b:
            mixed = (
                mixing_weight * partition_a[cell]
                + (1 - mixing_weight) * partition_b[cell]
            )
            partition_a[cell] = mixed
            partition_b[cell] = mixed
```

## Batch vs Stream Processing

### When to Use Batch

- Historical analysis over fixed datasets
- Model training and hyperparameter tuning
- Report generation
- Workloads where latency is not critical

```python
# Batch processing pattern with Dask
import dask.dataframe as dd

ddf = dd.read_parquet("s3://bucket/observations/year=2025/")
result = (
    ddf.groupby("h3_cell")
    .agg({"temperature": ["mean", "std"], "precipitation": "sum"})
    .compute()
)
```

### When to Use Stream

- Real-time sensor data from GEO-INFER-IOT
- Alerting systems (fire, flood, pollution)
- Live dashboards
- Workloads where results must be available within seconds

```python
from geo_infer_time.core.stream_processing import StreamProcessor
from datetime import timedelta

processor = StreamProcessor(
    window_size=timedelta(minutes=5),
    slide_interval=timedelta(minutes=1),
)

# Process incoming sensor data
for reading in sensor_stream:
    processor.add_data_point(
        timestamp=reading.timestamp,
        value=reading.temperature,
        metadata={"sensor_id": reading.sensor_id},
    )
```

## Cloud-Native Patterns

### Serverless Spatial Functions

For sporadic workloads, package analysis as serverless functions:

```python
# Example: AWS Lambda handler for point-in-polygon lookup
import json
import h3

def handler(event, context):
    lat = event["lat"]
    lng = event["lng"]
    resolution = event.get("resolution", 7)

    cell = h3.latlng_to_cell(lat, lng, resolution)
    neighbors = list(h3.grid_disk(cell, 1))

    return {
        "statusCode": 200,
        "body": json.dumps({
            "cell": cell,
            "neighbors": neighbors,
            "resolution": resolution,
        }),
    }
```

### Container-Based Scaling

For sustained workloads, deploy GEO-INFER modules as containers:

```yaml
# docker-compose.yml for a GEO-INFER analysis cluster
services:
  api:
    image: geo-infer-api:latest
    ports:
      - "8000:8000"
    environment:
      - WORKERS=4

  spatial-worker:
    image: geo-infer-space:latest
    deploy:
      replicas: 4
    environment:
      - H3_RESOLUTION=7

  temporal-worker:
    image: geo-infer-time:latest
    deploy:
      replicas: 2
```

## Cost Optimization

### Compute Cost Reduction

1. **Right-size H3 resolution** -- resolution 7 produces ~100x fewer cells than resolution 9
2. **Pre-filter before analysis** -- apply spatial and temporal filters before expensive computations
3. **Cache intermediate results** -- store GP fitted models rather than refitting
4. **Use spot/preemptible instances** -- batch analysis workloads tolerate interruption

### Storage Cost Reduction

1. **Use Cloud Optimized GeoTIFF (COG)** for raster data -- enables range requests
2. **Partition Parquet files by H3 cell** -- read only relevant partitions
3. **Compress with zstd** for columnar spatial data
4. **Set TTL on cache entries** -- avoid unbounded cache growth

### Example: Cost-Aware Resolution Selection

```python
def estimate_cost(n_points: int, resolution: int) -> dict:
    """Estimate processing cost for a given H3 resolution.

    Returns approximate compute hours and storage bytes.
    """
    # Number of unique H3 cells scales with resolution
    approx_cells = min(n_points, int(5.0 * (7 ** resolution)))
    compute_hours = approx_cells * 0.001 / 3600  # 1ms per cell
    storage_bytes = approx_cells * 128  # 128 bytes per cell record

    return {
        "resolution": resolution,
        "estimated_cells": approx_cells,
        "compute_hours": round(compute_hours, 2),
        "storage_mb": round(storage_bytes / 1e6, 2),
    }
```

## Practical Examples

### Scaling GEO-INFER-RISK: Monte Carlo Simulation

```python
from concurrent.futures import ProcessPoolExecutor
import numpy as np

def run_simulation_batch(seed: int, n_sims: int,
                          hazard_params: dict) -> np.ndarray:
    """Run a batch of Monte Carlo simulations with a fixed seed."""
    rng = np.random.default_rng(seed)
    losses = np.empty(n_sims)
    for i in range(n_sims):
        magnitude = rng.exponential(hazard_params["scale"])
        exposure = rng.uniform(0, hazard_params["max_exposure"])
        vulnerability = 1.0 - np.exp(-magnitude / hazard_params["resistance"])
        losses[i] = exposure * vulnerability
    return losses

# Distribute across cores
total_sims = 1_000_000
batch_size = 100_000
seeds = range(0, total_sims, batch_size)

with ProcessPoolExecutor() as pool:
    futures = [
        pool.submit(run_simulation_batch, seed, batch_size, {"scale": 2.0, "max_exposure": 1e6, "resistance": 5.0})
        for seed in seeds
    ]
    all_losses = np.concatenate([f.result() for f in futures])
```

## See Also

- [Performance Optimization](performance_optimization.md) -- single-machine optimization
- [Production Architecture](production_architecture.md) -- deployment patterns
- [Performance Issues](../support/performance_issues.md) -- diagnosing bottlenecks
