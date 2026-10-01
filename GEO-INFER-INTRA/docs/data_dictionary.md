# GEO-INFER Data Dictionary

This document defines the data structures, formats, naming conventions, and
exchange protocols used across all 45 GEO-INFER modules. Adherence to these
standards ensures interoperability between modules and predictable behavior
for developers working across the framework.

## GeoDataFrame Conventions

All vector geospatial data in GEO-INFER is represented using GeoPandas
GeoDataFrames. The following conventions apply universally.

### Geometry Column

- The geometry column must be named `geometry` (GeoPandas default).
- Valid geometry types: `Point`, `LineString`, `Polygon`, `MultiPoint`,
  `MultiLineString`, `MultiPolygon`, `GeometryCollection`.
- All geometries in a single GeoDataFrame should be of the same type unless
  the use case specifically requires mixed types.
- Empty geometries (`GEOMETRYCOLLECTION EMPTY`) must be handled explicitly;
  do not silently drop rows with null geometry.

### Coordinate Reference System

- Default CRS: **EPSG:4326** (WGS84 geographic coordinates, latitude/longitude
  in degrees).
- All GeoDataFrames must have a CRS set. Never create a GeoDataFrame without
  calling `gdf.set_crs()` or passing `crs=` to the constructor.
- When performing spatial operations between GeoDataFrames, verify CRS match
  first. Use `gdf.to_crs()` to reproject if necessary.

```python
import geopandas as gpd
from shapely.geometry import Point

# Correct: always set CRS
gdf = gpd.GeoDataFrame(
    {"name": ["Portland"], "value": [42.0]},
    geometry=[Point(-122.6765, 45.5231)],
    crs="EPSG:4326"
)

# Reproject to Web Mercator for distance calculations in meters
gdf_mercator = gdf.to_crs("EPSG:3857")
```

### Standard Column Names

GEO-INFER modules use consistent column names for common fields:

| Column Name | Type | Description | Required |
|-------------|------|-------------|----------|
| `geometry` | Shapely geometry | Spatial geometry object | Yes |
| `lat` | float64 | Latitude in decimal degrees (WGS84) | No |
| `lng` | float64 | Longitude in decimal degrees (WGS84) | No |
| `h3_index` | str | H3 cell index (hex string) | No |
| `h3_resolution` | int | H3 resolution level (0-15) | No |
| `timestamp` | datetime64[ns, UTC] | Observation timestamp | No |
| `value` | float64 | Primary measured value | No |
| `category` | str or category | Classification label | No |
| `confidence` | float64 | Confidence score [0.0, 1.0] | No |
| `source` | str | Data source identifier | No |
| `crs` | str | CRS identifier (when stored as attribute) | No |

### Example: Creating a Standard GeoDataFrame

```python
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
import h3

# Sample sensor readings
data = {
    "lat": [45.5231, 45.5150, 45.5300],
    "lng": [-122.6765, -122.6800, -122.6700],
    "timestamp": pd.to_datetime([
        "2025-06-15T10:00:00Z",
        "2025-06-15T10:05:00Z",
        "2025-06-15T10:10:00Z",
    ]),
    "value": [23.5, 24.1, 22.8],
    "category": ["temperature", "temperature", "temperature"],
    "confidence": [0.95, 0.92, 0.88],
    "source": ["sensor_A", "sensor_B", "sensor_C"],
}

geometry = [Point(lng, lat) for lat, lng in zip(data["lat"], data["lng"])]

gdf = gpd.GeoDataFrame(data, geometry=geometry, crs="EPSG:4326")

# Add H3 indices
gdf["h3_index"] = [
    h3.latlng_to_cell(lat, lng, 9)
    for lat, lng in zip(gdf["lat"], gdf["lng"])
]
gdf["h3_resolution"] = 9

print(gdf[["h3_index", "value", "category", "confidence"]].to_string())
```

## H3 Cell Conventions

GEO-INFER uses H3 v4 (`h3>=4.5.0,<5`) as its primary spatial indexing system.
The v3 API (geo_to_h3, h3_to_geo) is deprecated and must not be used.

### Resolution Guide

| Resolution | Avg Cell Area | Avg Edge Length | Typical Use Case |
|------------|--------------|-----------------|------------------|
| 0 | 4,357,449 km^2 | 1,108 km | Continental analysis |
| 1 | 609,788 km^2 | 419 km | Sub-continental regions |
| 2 | 86,802 km^2 | 158 km | Country-level analysis |
| 3 | 12,393 km^2 | 60 km | State/province analysis |
| 4 | 1,770 km^2 | 23 km | Regional planning |
| 5 | 252.9 km^2 | 8.5 km | Metropolitan area analysis |
| 6 | 36.13 km^2 | 3.2 km | City district analysis |
| 7 | 5.161 km^2 | 1.2 km | Neighborhood analysis |
| 8 | 0.7373 km^2 | 461 m | Urban block analysis |
| 9 | 0.1053 km^2 | 174 m | Building cluster analysis |
| 10 | 0.01505 km^2 | 66 m | Individual building scale |
| 11 | 0.002149 km^2 | 25 m | Parcel-level analysis |
| 12 | 0.0003071 km^2 | 9.4 m | Sub-parcel / room scale |
| 13 | 0.00004388 km^2 | 3.6 m | Precision positioning |
| 14 | 0.000006267 km^2 | 1.3 m | Sub-meter analysis |
| 15 | 0.0000008953 km^2 | 0.51 m | Centimeter-scale |

### Cell ID Format

H3 cell IDs are 64-bit unsigned integers represented as 15-character hexadecimal
strings (e.g., `"8928308280fffff"`). In Python, they are handled as `str` type.

### H3 v4 API Summary

| Function | Parameters | Return Type | Description |
|----------|-----------|-------------|-------------|
| `latlng_to_cell(lat, lng, res)` | float, float, int | str | Convert coordinates to H3 cell |
| `cell_to_latlng(cell)` | str | tuple[float, float] | Get cell center as (lat, lng) |
| `cell_to_boundary(cell)` | str | tuple[tuple[float, float], ...] | Get cell boundary vertices |
| `get_resolution(cell)` | str | int | Get resolution of a cell |
| `grid_disk(origin, k)` | str, int | frozenset[str] | Get cells within k rings |
| `grid_ring(origin, k)` | str, int | frozenset[str] | Get cells at exactly k rings |
| `grid_distance(a, b)` | str, str | int | Grid distance between cells |
| `grid_path_cells(a, b)` | str, str | list[str] | Shortest path between cells |
| `cell_to_parent(cell, res)` | str, int | str | Get parent cell at coarser resolution |
| `cell_to_children(cell, res)` | str, int | frozenset[str] | Get children at finer resolution |
| `are_neighbor_cells(a, b)` | str, str | bool | Check if cells are adjacent |
| `cells_to_multi_polygon(cells)` | set[str] | ... | Convert cells to GeoJSON polygon |
| `average_hexagon_area(res, unit)` | int, str | float | Average cell area at resolution |
| `average_hexagon_edge_length(res, unit)` | int, str | float | Average edge length at resolution |

```python
import h3

# Basic cell operations
cell = h3.latlng_to_cell(45.5231, -122.6765, 9)
print(f"Cell: {cell}")                          # e.g., "8928308280fffff"
print(f"Resolution: {h3.get_resolution(cell)}") # 9

center = h3.cell_to_latlng(cell)
print(f"Center: lat={center[0]:.4f}, lng={center[1]:.4f}")

boundary = h3.cell_to_boundary(cell)
print(f"Boundary vertices: {len(boundary)}")    # 6 (hexagon)

# Neighborhood operations
neighbors = h3.grid_disk(cell, 1)
print(f"1-ring neighborhood: {len(neighbors)} cells")  # 7

ring_only = h3.grid_ring(cell, 2)
print(f"2nd ring only: {len(ring_only)} cells")         # 12

# Hierarchy
parent = h3.cell_to_parent(cell, 7)
children = h3.cell_to_children(cell, 10)
print(f"Parent (res 7): {parent}")
print(f"Children (res 10): {len(children)} cells")
```

## Temporal Series Formats

### DatetimeIndex Standards

- All timestamps must be timezone-aware with UTC as the default timezone.
- Use `pd.Timestamp` or `datetime64[ns, UTC]` dtype.
- Store timestamps as ISO 8601 strings in serialized formats:
  `"2025-06-15T10:30:00Z"`.

```python
import pandas as pd

# Correct: timezone-aware UTC DatetimeIndex
index = pd.DatetimeIndex(
    pd.date_range("2025-01-01", periods=365, freq="D", tz="UTC")
)

# Create a time series with proper index
ts = pd.Series(
    data=np.random.randn(365),
    index=index,
    name="temperature_anomaly"
)

# Resample to monthly means
monthly = ts.resample("ME").mean()
```

### Period Conventions

| Period Code | Meaning | Use Case |
|-------------|---------|----------|
| `T` or `min` | Minute | Sensor data, IoT |
| `H` or `h` | Hour | Traffic, weather |
| `D` | Day | Daily aggregates |
| `W` | Week | Weekly reports |
| `ME` | Month end | Monthly statistics |
| `QE` | Quarter end | Seasonal analysis |
| `YE` | Year end | Annual summaries |

### Resampling Standards

When resampling spatial time series, specify the aggregation method explicitly:

```python
# Spatial time series: aggregate temperature readings
daily_mean = hourly_readings.resample("D").mean()
daily_max = hourly_readings.resample("D").max()
daily_count = hourly_readings.resample("D").count()

# Preserve spatial information during resampling
def resample_spatial_ts(gdf, time_col, value_col, freq, agg_func="mean"):
    """Resample a spatial time series while preserving geometry."""
    grouped = gdf.set_index(time_col).groupby("h3_index")
    resampled = grouped[value_col].resample(freq).agg(agg_func).reset_index()
    # Rejoin geometry
    cell_geom = gdf[["h3_index", "geometry"]].drop_duplicates("h3_index")
    result = resampled.merge(cell_geom, on="h3_index")
    return gpd.GeoDataFrame(result, geometry="geometry", crs=gdf.crs)
```

## Active Inference State Vectors

Active Inference models in GEO-INFER use a standardized parameterization
following the notation in `GEO-INFER-ACT`.

### State Space Components

| Symbol | Name | Shape | Description | Variable Name |
|--------|------|-------|-------------|---------------|
| `o` | Observations | `(num_obs,)` or `(num_obs_modalities, ...)` | Sensory input vector | `observations` |
| `s` | Hidden states | `(num_states,)` or `(num_state_factors, ...)` | Inferred hidden states | `states` or `beliefs` |
| `a` | Actions | `(num_actions,)` | Selected actions | `actions` |
| `D` | Prior beliefs | `(num_states,)` | Prior distribution over initial states | `state_prior` |
| `A` | Likelihood | `(num_obs, num_states)` | Observation model P(o\|s) | `observation_model` or `likelihood` |
| `B` | Transitions | `(num_states, num_states)` or `(num_states, num_states, num_actions)` | State transition model P(s'\|s,a) | `transition_model` |
| `C` | Preferences | `(num_obs,)` | Log-preferences over observations | `preferences` |
| `E` | Policy prior | `(num_policies,)` | Prior over policies | `policy_prior` |

### Array Conventions

- All probability distributions are stored as numpy arrays with dtype `float64`.
- Probability vectors must sum to 1.0 (within floating-point tolerance of 1e-10).
- Log-probabilities use natural logarithm (`np.log`, not `np.log2` or `np.log10`).
- The likelihood matrix `A` has observations on rows and states on columns:
  `A[o, s] = P(o | s)`.
- The transition matrix `B` follows the convention `B[s', s] = P(s' | s)` for
  the action-free case.

```python
import numpy as np

# Standard Active Inference state vector setup
num_states = 5
num_obs = 4
num_actions = 3

# Likelihood matrix: P(observation | hidden_state)
A = np.random.dirichlet(np.ones(num_obs), size=num_states).T
assert A.shape == (num_obs, num_states)
assert np.allclose(A.sum(axis=0), 1.0)

# Transition matrix: P(next_state | current_state, action)
B = np.zeros((num_states, num_states, num_actions))
for a in range(num_actions):
    B[:, :, a] = np.random.dirichlet(np.ones(num_states), size=num_states).T
    assert np.allclose(B[:, :, a].sum(axis=0), 1.0)

# Prior beliefs over initial states
D = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
assert np.isclose(D.sum(), 1.0)

# Preferences over observations (log-scale; higher = more preferred)
C = np.array([0.0, 1.0, 2.0, 3.0])  # prefers observation index 3
```

## Cross-Module Exchange Formats

### JSON Schema for Spatial Features

When exchanging spatial features between modules via JSON, use GeoJSON format
with GEO-INFER metadata extensions:

```json
{
  "type": "FeatureCollection",
  "crs": "EPSG:4326",
  "geo_infer_metadata": {
    "source_module": "GEO-INFER-SPACE",
    "version": "0.1.0",
    "created_at": "2025-06-15T10:30:00Z",
    "h3_resolution": 9
  },
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Point",
        "coordinates": [-122.6765, 45.5231]
      },
      "properties": {
        "h3_index": "8928308280fffff",
        "value": 23.5,
        "category": "temperature",
        "confidence": 0.95,
        "timestamp": "2025-06-15T10:00:00Z"
      }
    }
  ]
}
```

Note: GeoJSON coordinates use **(longitude, latitude)** order, which is the
opposite of H3's `(lat, lng)` convention. Always convert explicitly.

### GeoParquet Column Metadata

GeoParquet files must include the following metadata in the `geo` key of the
Parquet file metadata:

```python
import geopandas as gpd

# Write GeoParquet with standard metadata
gdf.to_parquet(
    "output.parquet",
    engine="pyarrow",
    index=False,
)

# Read GeoParquet
gdf_loaded = gpd.read_parquet("output.parquet")
assert gdf_loaded.crs is not None
```

### Inter-Module Data Transfer

When passing data between GEO-INFER modules in-process, use these conventions:

| Transfer Type | Format | Example |
|---------------|--------|---------|
| Vector spatial data | GeoDataFrame | `gdf = module_a.process(input_gdf)` |
| Raster data | xarray.DataArray | `da = module_b.load_raster(path)` |
| Time series | pandas.Series or DataFrame with DatetimeIndex | `ts = module_c.extract_series(gdf)` |
| Active Inference states | numpy.ndarray | `beliefs = agent.perceive(observation)` |
| Configuration | dict | `config = {"resolution": 9, "crs": "EPSG:4326"}` |
| Model parameters | dict[str, numpy.ndarray] | `params = {"A": A, "B": B, "D": D}` |

## Standard Type Annotations

GEO-INFER uses strict type annotations throughout. These are the standard types
used across modules:

```python
from collections.abc import Sequence
from typing import Any
import numpy as np
import numpy.typing as npt
import geopandas as gpd
import pandas as pd
from shapely.geometry import (
    Point, LineString, Polygon, MultiPolygon, MultiPoint
)
from shapely.geometry.base import BaseGeometry

# Standard type aliases used across GEO-INFER
Coordinate = tuple[float, float]              # (latitude, longitude)
BoundingBox = tuple[float, float, float, float]  # (min_lng, min_lat, max_lng, max_lat)
H3CellId = str                                 # H3 hex string
H3Resolution = int                              # 0-15
CRSType = str | int                       # "EPSG:4326" or 4326
TimestampType = str | pd.Timestamp        # ISO 8601 string or Timestamp
ProbabilityVector = npt.NDArray[np.float64]      # sums to 1.0
StateVector = npt.NDArray[np.float64]            # arbitrary float vector
TransitionMatrix = npt.NDArray[np.float64]       # square stochastic matrix
LikelihoodMatrix = npt.NDArray[np.float64]       # column-stochastic matrix

# Function signature examples
def analyze_region(
    center: Coordinate,
    radius_km: float,
    resolution: H3Resolution = 9,
    crs: CRSType = "EPSG:4326",
    start_time: TimestampType | None = None,
    end_time: TimestampType | None = None,
) -> gpd.GeoDataFrame:
    """Analyze a circular region around a center point."""
    ...

def update_beliefs(
    prior: ProbabilityVector,
    likelihood: LikelihoodMatrix,
    observation: ProbabilityVector,
) -> ProbabilityVector:
    """Bayesian belief update."""
    ...
```

## Data Validation Patterns

All modules should validate incoming data using these patterns:

```python
import numpy as np
import geopandas as gpd


def validate_geodataframe(gdf: gpd.GeoDataFrame) -> None:
    """Validate a GeoDataFrame meets GEO-INFER standards."""
    if gdf.crs is None:
        raise ValueError("GeoDataFrame must have a CRS set")

    if gdf.geometry.isna().any():
        null_count = gdf.geometry.isna().sum()
        raise ValueError(f"GeoDataFrame contains {null_count} null geometries")

    if not gdf.geometry.is_valid.all():
        invalid_count = (~gdf.geometry.is_valid).sum()
        raise ValueError(
            f"GeoDataFrame contains {invalid_count} invalid geometries"
        )


def validate_probability_vector(vec: np.ndarray, name: str = "vector") -> None:
    """Validate that an array is a proper probability distribution."""
    if vec.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional, got {vec.ndim}D")

    if not np.all(vec >= 0):
        raise ValueError(f"{name} contains negative values")

    total = vec.sum()
    if not np.isclose(total, 1.0, atol=1e-10):
        raise ValueError(
            f"{name} does not sum to 1.0 (sum={total:.10f})"
        )


def validate_transition_matrix(B: np.ndarray, name: str = "B") -> None:
    """Validate a column-stochastic transition matrix."""
    if B.ndim < 2:
        raise ValueError(f"{name} must be at least 2-dimensional")

    col_sums = B.sum(axis=0)
    if not np.allclose(col_sums, 1.0, atol=1e-10):
        raise ValueError(
            f"{name} columns do not sum to 1.0: {col_sums}"
        )
```

## Related Documentation

- [Geospatial Standards](geospatial_standards.md) -- CRS, H3, and format details
- [Active Inference Guide](active_inference_guide.md) -- mathematical foundations
- [Terminology](terminology.md) -- definitions of all terms used here
- [Installation](installation.md) -- dependency setup for data libraries
